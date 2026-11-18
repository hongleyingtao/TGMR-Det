from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Save an experiment configuration without runtime-only metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_cfg = {
        key: value
        for key, value in cfg.items()
        if not str(key).startswith("_")
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            serializable_cfg,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )