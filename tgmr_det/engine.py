import torch
from tqdm import tqdm
from .utils import move_targets_to_device


def train_one_epoch(
    model, criterion, loader, optimizer, scaler, device, epoch,
    amp=True, accumulate=1, grad_clip=10.0,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = {k: 0.0 for k in ("loss", "loss_cls", "loss_ctr", "loss_giou", "loss_nwd", "loss_align")}
    steps = 0
    pbar = tqdm(loader, desc=f"Train {epoch}", dynamic_ncols=True)

    for step, (visible, thermal, targets) in enumerate(pbar):
        visible = visible.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            outputs = model(visible, thermal)
            loss_dict = criterion(model, outputs, targets)
            loss = loss_dict["loss"] / accumulate

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        if (step + 1) % accumulate == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        steps += 1
        for key in running:
            running[key] += float(loss_dict[key].detach())
        pbar.set_postfix(loss=f"{running['loss'] / steps:.4f}", pos=int(loss_dict["num_pos"]))

    return {k: v / max(steps, 1) for k, v in running.items()}
