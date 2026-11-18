from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath, trunc_normal_
except Exception:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    def trunc_normal_(tensor, std=0.02):
        return nn.init.trunc_normal_(tensor, std=std)


layer_scale = True
init_value = 1e-6


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        num_heads = max(1, min(int(num_heads), int(dim)))
        while dim % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


def block(x, win_h, win_w):
    b, h, w, c = x.shape
    pad_h = (win_h - h % win_h) % win_h
    pad_w = (win_w - w % win_w) % win_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.reshape(b, hp // win_h, win_h, wp // win_w, win_w, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, h, w, hp, wp, c


def unblock(x, out_h, out_w):
    b, gh, gw, win_h, win_w, c = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(b, gh * win_h, gw * win_w, c)
    return x[:, :out_h, :out_w, :].contiguous()


def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    grid_h = max(1, math.ceil(x.shape[1] / sparse_size))
    grid_w = max(1, math.ceil(x.shape[2] / sparse_size))
    out, h, w, hp, wp, c = block(x, grid_h, grid_w)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, c)
    out = out.permute(0, 3, 1, 2)
    return out, h, w, hp, wp, c


def alter_unsparse(x, h, w, hp, wp, c, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, hp // sparse_size, wp // sparse_size, sparse_size, sparse_size, c)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, h, w)
    return out.permute(0, 3, 1, 2)


class SABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        sparse_size=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls = layer_scale
        self.sparse_size = int(sparse_size)
        if self.ls:
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.pos_embed(x)
        pad_h = (self.sparse_size - h % self.sparse_size) % self.sparse_size
        pad_w = (self.sparse_size - w % self.sparse_size) % self.sparse_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        _, _, hp, wp = x.shape
        shortcut = x.flatten(2).transpose(1, 2)

        xs, ho, wo, hps, wps, cs = alter_sparse(x, self.sparse_size)
        bf, cf, hf, wf = xs.shape
        xs = xs.flatten(2).transpose(1, 2)
        xs = self.attn(self.norm1(xs))
        xs = xs.transpose(1, 2).reshape(bf, cf, hf, wf)
        xs = alter_unsparse(xs, ho, wo, hps, wps, cs, self.sparse_size)
        xs = xs.flatten(2).transpose(1, 2)

        if self.ls:
            g1 = self.gamma_1.to(dtype=xs.dtype, device=xs.device)
            g2 = self.gamma_2.to(dtype=xs.dtype, device=xs.device)
            x_tokens = shortcut + self.drop_path(g1 * xs)
            x_tokens = x_tokens + self.drop_path(g2 * self.mlp(self.norm2(x_tokens), hp, wp))
        else:
            x_tokens = shortcut + self.drop_path(xs)
            x_tokens = x_tokens + self.drop_path(self.mlp(self.norm2(x_tokens), hp, wp))

        x = x_tokens.transpose(1, 2).reshape(b, c, hp, wp)
        return x[:, :, :h, :w] if pad_h > 0 or pad_w > 0 else x


class StructuredSparseAttentionResidual(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, sparse_size=2):
        super().__init__()
        self.attn = SABlock(c, num_heads=num_heads, sparse_size=sparse_size)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class StructuredSparseSelfAttentionBlock(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, sparse_size=2):
        super().__init__()
        if c1 != c2:
            raise ValueError("StructuredSparseSelfAttentionBlock requires c1 == c2")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        heads = max(1, self.c // 64)
        while self.c % heads != 0:
            heads -= 1
        self.m = nn.Sequential(
            *(StructuredSparseAttentionResidual(self.c, attn_ratio=0.5, num_heads=heads, sparse_size=sparse_size) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
