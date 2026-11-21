"""Optional Weights & Biases logging that never aborts a run.

Off unless the config flag is set or ``BATTLEBOATS_WANDB`` is ``1`` / ``true``
/ ``yes`` / ``on``. A missing package, a missing API key, or an init/log
error is printed once and ignored.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def wandb_source(config_flag: bool) -> Optional[str]:
    """Why logging was requested, or None when it stays off.

    ``BATTLEBOATS_WANDB`` wins over the config flag so a shell override is
    visible in the log line.
    """
    env = os.environ.get("BATTLEBOATS_WANDB", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return "BATTLEBOATS_WANDB"
    if config_flag:
        return "config"
    return None


def wandb_requested(config_flag: bool) -> bool:
    return wandb_source(config_flag) is not None


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
    """Start a run, or return a no-op logger.

    Always prints one line: ``[wandb] on ...`` or ``[wandb] off ...``. A
    missing package, a missing key, or an init error is off, with the reason
    on that same line.
    """
    source = wandb_source(config_flag)
    if source is None:
        print("[wandb] off (config wandb=false, BATTLEBOATS_WANDB unset)")
        return WandbLog()
    try:
        import wandb
    except ImportError:
        print("[wandb] off (package is not installed); continuing without logging")
        return WandbLog()
    if not _has_credentials(wandb):
        print("[wandb] off (not logged in, no API key); continuing without logging")
        return WandbLog()
    try:
        wandb.init(project=project, config=config, reinit=True)
    except Exception as exc:
        print(f"[wandb] off (init failed: {exc}); continuing without logging")
        return WandbLog()
    print(f"[wandb] on project={project} via {source}")
    return WandbLog(wandb)
