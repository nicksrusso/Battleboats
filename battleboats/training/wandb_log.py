"""Optional Weights & Biases logging that never aborts a run.

Off unless the config flag is set or ``BATTLEBOATS_WANDB`` is ``1`` / ``true``
/ ``yes`` / ``on``. A missing package, a missing API key, or an init/log
error is printed once and ignored.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def wandb_requested(config_flag: bool) -> bool:
    env = os.environ.get("BATTLEBOATS_WANDB", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    return bool(config_flag)


class WandbLog:
    """Thin logger. ``enabled`` is false when logging was skipped or failed."""

    def __init__(self, module: Any = None) -> None:
        self._mod = module

    @property
    def enabled(self) -> bool:
        return self._mod is not None

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._mod is None:
            return
        try:
            self._mod.log(data, step=step)
        except Exception as exc:
            print(f"[wandb] log failed ({exc}); continuing without logging")
            self.finish()

    def finish(self) -> None:
        mod = self._mod
        self._mod = None
        if mod is None:
            return
        try:
            mod.finish()
        except Exception as exc:
            print(f"[wandb] finish failed ({exc})")


def _has_credentials(wandb: Any) -> bool:
    mode = os.environ.get("WANDB_MODE", "online")
    if mode in ("offline", "disabled"):
        return True
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        api = getattr(wandb, "api", None)
        return bool(getattr(api, "api_key", None))
    except Exception:
        return False


def init_wandb(config_flag: bool, *, project: str, config: Dict[str, Any]) -> WandbLog:
    """Start a run, or return a no-op logger."""
    if not wandb_requested(config_flag):
        return WandbLog()
    try:
        import wandb
    except ImportError:
        print("[wandb] package is not installed; continuing without logging")
        return WandbLog()
    if not _has_credentials(wandb):
        print("[wandb] not logged in (no API key); continuing without logging")
        return WandbLog()
    try:
        wandb.init(project=project, config=config, reinit=True)
    except Exception as exc:
        print(f"[wandb] init failed ({exc}); continuing without logging")
        return WandbLog()
    return WandbLog(wandb)
