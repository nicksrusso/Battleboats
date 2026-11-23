"""Training-run manifests: write, finish, and append evals to run.json.

Each trainer calls :func:`start_run` (or :class:`ManifestSession`) after
creating its run directory and :func:`finish_run` when it exits. Eval
scripts call :func:`append_eval` with the checkpoint they just scored;
if that checkpoint belongs to a run that already has a ``run.json``,
the summary is appended to the ``evals`` list. Unknown fields stay
``null``. Secrets in the command line or config are redacted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

SCHEMA_VERSION = 1
MANIFEST_NAME = "run.json"
CHECKPOINTS_DIRNAME = "checkpoints"

# Top-level names under runs/ that are not training runs.
SKIP_RUN_DIR_NAMES = frozenset(
    {
        "checkpoints",
        "eval",
        "pgdata",
        "maps",
        "snapshots",
        "logs",
    }
)

KIND_SELFPLAY = "selfplay"
KIND_SANDBOX = "sandbox"
KIND_BC = "bc"
KIND_BUYDIV = "buydiv"
KIND_CURRICULUM = "curriculum"
KIND_SKILL_PIPELINE = "skill_pipeline"
KIND_TAGGED = "tagged"

KNOWN_KINDS = (
    KIND_SELFPLAY,
    KIND_SANDBOX,
    KIND_BC,
    KIND_BUYDIV,
    KIND_CURRICULUM,
    KIND_SKILL_PIPELINE,
    KIND_TAGGED,
)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|authorization|openrouter)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)((?:OPENROUTER_API_KEY|API_KEY|SECRET|TOKEN|PASSWORD)\s*[=:]\s*)(\S+)"
)
_REDACTED = "********"

PathLike = Union[str, Path]


def manifest_path(run_dir: PathLike) -> Path:
    return Path(run_dir) / MANIFEST_NAME


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_jsonable(obj: Any, *, _key: str = "") -> Any:
    """Convert a config object to JSON, redacting secret-looking keys."""
    if _key and _SECRET_KEY_RE.search(str(_key)):
        if obj in (None, ""):
            return obj
        return _REDACTED
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str):
            return _SECRET_VALUE_RE.sub(lambda m: m.group(1) + _REDACTED, obj)
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v, _key=k) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, _key=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return to_jsonable(vars(obj))
        except TypeError:
            pass
    return str(obj)


def command_line(argv: Optional[Sequence[str]] = None) -> str:
    args = list(sys.argv if argv is None else argv)
    cleaned: List[str] = []
    hide_next = False
    for arg in args:
        if hide_next:
            cleaned.append(_REDACTED)
            hide_next = False
            continue
        if arg.startswith("-") and _SECRET_KEY_RE.search(arg.split("=", 1)[0]):
            if "=" in arg:
                key, _val = arg.split("=", 1)
                cleaned.append(f"{key}={_REDACTED}")
            else:
                cleaned.append(arg)
                hide_next = True
            continue
        cleaned.append(_SECRET_VALUE_RE.sub(lambda m: m.group(1) + _REDACTED, arg))
    return shlex.join(cleaned)


def git_info(repo_root: Optional[PathLike] = None) -> Dict[str, Any]:
    """Return ``{"commit": sha or None, "dirty": bool or None}``."""
    root = _git_root(repo_root)
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = _git(root, ["rev-parse", "HEAD"])
        porcelain = _git(root, ["status", "--porcelain"])
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": bool(porcelain)}


def sha256_file(path: PathLike) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_rewards(config: Any) -> Optional[Dict[str, Any]]:
    """Pull the reward-related subset out of a resolved config object or dict."""
    if config is None:
        return None
    data = to_jsonable(config)
    if not isinstance(data, dict):
        return None
    out: Dict[str, Any] = {}
    if "reward_shaping" in data:
        out["reward_shaping"] = data["reward_shaping"]
    if "reward" in data:
        out["reward"] = data["reward"]
    for key in (
        "draw_reward",
        "coverage_coef",
        "anchor_coef",
        "vf_coef",
        "ent_coef",
        "hull_ent_coef",
    ):
        if key in data:
            out[key] = data[key]
    return out or None


def infer_kind(run_name: str) -> Optional[str]:
    """Best-effort kind from a run directory name. Returns None if unknown."""
    n = run_name.lower()
    if n.startswith("sandbox_buydiv") or "buydiv" in n:
        return KIND_BUYDIV
    if n.startswith("sandboxes") or n.startswith("sandbox"):
        return KIND_SANDBOX
    if n.startswith("selfplay"):
        return KIND_SELFPLAY
    if n.startswith("bc_") or n.endswith("_bc") or "tactical_bc" in n:
        return KIND_BC
    if n.startswith("curriculum"):
        return KIND_CURRICULUM
    if "skill_pipeline" in n:
        return KIND_SKILL_PIPELINE
    return None


def empty_manifest() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": None,
        "kind": None,
        "backfilled": False,
        "started_at": None,
        "ended_at": None,
        "status": None,
        "config": None,
        "git": {"commit": None, "dirty": None},
        "command": None,
        "parent": None,
        "rewards": None,
        "updates_completed": None,
        "final_checkpoint": None,
        "evals": [],
    }


def load_manifest(run_dir: PathLike) -> Optional[Dict[str, Any]]:
    path = manifest_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_manifest(run_dir: PathLike, data: Mapping[str, Any]) -> Path:
    dest = manifest_path(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    text = json.dumps(data, indent=2, sort_keys=False, default=str) + "\n"
    tmp.write_text(text)
    os.replace(tmp, dest)
    return dest


def describe_checkpoint(
    path: Optional[PathLike],
    *,
    checkpoints_dir: Optional[PathLike] = None,
) -> Optional[Dict[str, Any]]:
    """``{path, sha256, tag}`` for a checkpoint, or None if ``path`` is empty."""
    if path is None or str(path) == "":
        return None
    p = Path(path)
    info: Dict[str, Any] = {
        "path": str(p),
        "sha256": sha256_file(p) if p.is_file() else None,
        "tag": None,
    }
    info["tag"] = match_checkpoint_tag(
        p, sha256=info["sha256"], checkpoints_dir=checkpoints_dir
    )
    return info


def match_checkpoint_tag(
    path: PathLike,
    *,
    sha256: Optional[str] = None,
    checkpoints_dir: Optional[PathLike] = None,
) -> Optional[str]:
    """Return the version tag if this file matches ``runs/checkpoints/vX.Y.Z_*``."""
    p = Path(path)
    name_tag = _tag_from_filename(p.name)
    ckpt_dir = _checkpoints_dir(p, checkpoints_dir)
    if ckpt_dir is None:
        return name_tag
    digest = sha256 or (sha256_file(p) if p.is_file() else None)
    resolved = _safe_resolve(p)
    for sidecar in sorted(ckpt_dir.glob("v*.json")):
        meta = _read_json(sidecar)
        if not isinstance(meta, dict):
            continue
        tag = meta.get("version") or _tag_from_filename(sidecar.stem)
        weights = meta.get("weights")
        sidecar_sha = meta.get("sha256")
        if digest and sidecar_sha and digest == sidecar_sha:
            return str(tag) if tag else name_tag
        if weights:
            weight_path = _resolve_maybe(weights, start=ckpt_dir.parent.parent)
            if resolved and weight_path and resolved == weight_path:
                return str(tag) if tag else name_tag
            if Path(weights).name == p.name and name_tag:
                return name_tag
    if name_tag and (p.parent == ckpt_dir or p.name.startswith("v")):
        return name_tag
    return None


def canonical_run_dir(path: PathLike) -> Path:
    """The run root for a save path (``.../bc_tactical_v040/train`` -> parent)."""
    p = Path(path)
    if p.name in {"train", "harvest"}:
        return p.parent
    return p


def start_run(
    run_dir: PathLike,
    *,
    kind: str,
    config: Any = None,
    init_checkpoint: Optional[PathLike] = None,
    rewards: Any = None,
    run_name: Optional[str] = None,
    repo_root: Optional[PathLike] = None,
    checkpoints_dir: Optional[PathLike] = None,
    command: Optional[str] = None,
    argv: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write (or resume) ``run.json`` at the start of a training run."""
    dest_dir = Path(run_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(dest_dir)
    data = empty_manifest()
    started_at = now_iso()
    evals: List[Any] = []
    if existing:
        # Resume: keep the original start time and any evals already recorded.
        if existing.get("started_at"):
            started_at = existing["started_at"]
        if isinstance(existing.get("evals"), list):
            evals = list(existing["evals"])
        if existing.get("command") and command is None and argv is None:
            command = existing["command"]
        if existing.get("git") and existing["git"].get("commit"):
            data["git"] = existing["git"]
        else:
            data["git"] = git_info(repo_root)
    else:
        data["git"] = git_info(repo_root)
    data["run_name"] = run_name or dest_dir.name
    data["kind"] = kind
    data["backfilled"] = False
    data["started_at"] = started_at
    data["ended_at"] = None
    data["status"] = "running"
    data["config"] = to_jsonable(config)
    if command is None:
        command = command_line(argv)
    data["command"] = command
    parent = describe_checkpoint(init_checkpoint, checkpoints_dir=checkpoints_dir)
    data["parent"] = parent
    data["rewards"] = to_jsonable(rewards) if rewards is not None else extract_rewards(config)
    data["updates_completed"] = existing.get("updates_completed") if existing else None
    data["final_checkpoint"] = existing.get("final_checkpoint") if existing else None
    data["evals"] = evals
    write_manifest(dest_dir, data)
    return data


def finish_run(
    run_dir: PathLike,
    *,
    final_checkpoint: Optional[PathLike] = None,
    updates_completed: Optional[int] = None,
    status: str = "completed",
    checkpoints_dir: Optional[PathLike] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Update ``run.json`` at the end of a training run."""
    dest_dir = Path(run_dir)
    data = load_manifest(dest_dir) or empty_manifest()
    if not data.get("run_name"):
        data["run_name"] = dest_dir.name
    data["ended_at"] = now_iso()
    data["status"] = status
    if updates_completed is not None:
        data["updates_completed"] = int(updates_completed)
    if final_checkpoint is not None:
        data["final_checkpoint"] = describe_checkpoint(
            final_checkpoint, checkpoints_dir=checkpoints_dir
        )
    if extra:
        # Only merge keys the caller names; do not invent.
        for key, value in extra.items():
            data[key] = to_jsonable(value)
    write_manifest(dest_dir, data)
    return data


def append_eval(
    summary: Mapping[str, Any],
    *,
    checkpoint: Optional[PathLike] = None,
    run_dir: Optional[PathLike] = None,
    name: Optional[str] = None,
    checkpoints_dir: Optional[PathLike] = None,
    runs_root: Optional[PathLike] = None,
) -> Optional[Path]:
    """Append a compact eval summary to the checkpoint's run.json, if any.

    Returns the path of the updated manifest, or None if no run.json exists.
    """
    dest = Path(run_dir) if run_dir is not None else None
    if dest is None and checkpoint is not None:
        dest = run_dir_for_checkpoint(
            checkpoint, checkpoints_dir=checkpoints_dir, runs_root=runs_root
        )
    if dest is None:
        return None
    data = load_manifest(dest)
    if data is None:
        return None
    entry: Dict[str, Any] = {
        "recorded_at": now_iso(),
        "name": name,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "summary": compact_eval(summary),
    }
    evals = data.get("evals")
    if not isinstance(evals, list):
        evals = []
    evals.append(entry)
    data["evals"] = evals
    write_manifest(dest, data)
    return manifest_path(dest)


def compact_eval(summary: Mapping[str, Any]) -> Any:
    """Drop bulky per-game blobs so run.json stays a short record."""
    drop = {"games", "episodes_raw", "curve"}
    return _drop_keys(to_jsonable(summary), drop)


def run_dir_for_checkpoint(
    checkpoint: PathLike,
    *,
    checkpoints_dir: Optional[PathLike] = None,
    runs_root: Optional[PathLike] = None,
) -> Optional[Path]:
    """Locate the training run directory that owns this checkpoint, if any."""
    p = Path(checkpoint)
    # Tagged official copy: the sidecar names source_run.
    ckpt_dir = _checkpoints_dir(p, checkpoints_dir)
    if ckpt_dir is not None and _safe_resolve(p) and (
        p.parent == ckpt_dir or _safe_resolve(p.parent) == _safe_resolve(ckpt_dir)
    ):
        sidecar = _sidecar_for_weights(p, ckpt_dir)
        if sidecar:
            meta = _read_json(sidecar)
            source = (meta or {}).get("source_run") if isinstance(meta, dict) else None
            if source:
                root = runs_root or ckpt_dir.parent
                candidate = _resolve_maybe(source, start=Path(root).parent)
                if candidate and manifest_path(candidate).is_file():
                    return candidate
                # source_run is often "runs/foo" relative to the repo.
                rel = Path(str(source).split()[0])
                if runs_root is not None:
                    alt = Path(runs_root) / rel.name
                    if manifest_path(alt).is_file():
                        return alt
                if rel.name and ckpt_dir is not None:
                    alt = ckpt_dir.parent / rel.name
                    if manifest_path(alt).is_file():
                        return alt
    # Walk up from the file looking for a run.json, but not past runs/.
    root = Path(runs_root) if runs_root is not None else _runs_root_from(p)
    for folder in [p.parent] + list(p.parent.parents):
        if manifest_path(folder).is_file():
            return folder
        if root is not None and _safe_resolve(folder) == _safe_resolve(root):
            break
        if folder.name == "runs":
            break
    return None


def is_run_in_progress(run_dir: PathLike) -> bool:
    """True when ``train.pid`` exists and that process is still alive."""
    pid_path = Path(run_dir) / "train.pid"
    if not pid_path.is_file():
        return False
    try:
        text = pid_path.read_text().strip()
    except OSError:
        return False
    if not text:
        return False
    try:
        pid = int(text.split()[0])
    except ValueError:
        return False
    return pid_is_alive(pid)


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, but we cannot signal it.
        return True
    except OSError:
        return False
    return True


class ManifestSession:
    """Start a manifest on enter; finish or fail on exit."""

    def __init__(
        self,
        run_dir: PathLike,
        *,
        kind: str,
        config: Any = None,
        init_checkpoint: Optional[PathLike] = None,
        rewards: Any = None,
        run_name: Optional[str] = None,
        repo_root: Optional[PathLike] = None,
        checkpoints_dir: Optional[PathLike] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.kind = kind
        self.config = config
        self.init_checkpoint = init_checkpoint
        self.rewards = rewards
        self.run_name = run_name
        self.repo_root = repo_root
        self.checkpoints_dir = checkpoints_dir
        self._finished = False
        self.data: Optional[Dict[str, Any]] = None

    def start(self) -> "ManifestSession":
        self.data = start_run(
            self.run_dir,
            kind=self.kind,
            config=self.config,
            init_checkpoint=self.init_checkpoint,
            rewards=self.rewards,
            run_name=self.run_name,
            repo_root=self.repo_root,
            checkpoints_dir=self.checkpoints_dir,
        )
        return self

    def complete(
        self,
        *,
        final_checkpoint: Optional[PathLike] = None,
        updates_completed: Optional[int] = None,
        status: str = "completed",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = finish_run(
            self.run_dir,
            final_checkpoint=final_checkpoint,
            updates_completed=updates_completed,
            status=status,
            checkpoints_dir=self.checkpoints_dir,
            extra=extra,
        )
        self._finished = True
        self.data = data
        return data

    def fail(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return self.complete(status="failed", extra=extra)

    def __enter__(self) -> "ManifestSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._finished:
            return None
        if exc_type is not None:
            self.fail()
        else:
            self.complete(status="completed")
        return None


# ---------------------------------------------------------------------------
# Backfill helpers (best-effort; never invent a value)
# ---------------------------------------------------------------------------


def backfill_run(
    run_dir: PathLike,
    *,
    runs_root: Optional[PathLike] = None,
    checkpoints_dir: Optional[PathLike] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Write a backfilled run.json from artifacts already in ``run_dir``.

    Returns a status dict with keys ``status`` (written / skipped / error)
    and ``reason``. Does not modify any file other than ``run.json``.
    """
    dest = Path(run_dir)
    if is_run_in_progress(dest):
        return {
            "status": "skipped",
            "reason": "in_progress",
            "run_dir": str(dest),
        }
    existing = load_manifest(dest)
    if existing is not None and not overwrite:
        return {
            "status": "skipped",
            "reason": "exists",
            "run_dir": str(dest),
        }
    if existing is not None and overwrite and not existing.get("backfilled"):
        return {
            "status": "skipped",
            "reason": "live_manifest",
            "run_dir": str(dest),
        }
    data, missing = reconstruct_from_artifacts(
        dest, runs_root=runs_root, checkpoints_dir=checkpoints_dir
    )
    write_manifest(dest, data)
    return {
        "status": "written",
        "reason": "backfilled",
        "run_dir": str(dest),
        "missing": missing,
    }


def reconstruct_from_artifacts(
    run_dir: PathLike,
    *,
    runs_root: Optional[PathLike] = None,
    checkpoints_dir: Optional[PathLike] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Build a manifest from files on disk. Missing fields stay null."""
    dest = Path(run_dir)
    data = empty_manifest()
    data["backfilled"] = True
    data["run_name"] = dest.name
    data["kind"] = infer_kind(dest.name)
    missing: List[str] = []

    config = _load_yaml(dest / "config.yaml")
    if config is None:
        # BC runs store hyperparams in train/metrics.json, not config.yaml.
        metrics = _read_json(dest / "train" / "metrics.json")
        if isinstance(metrics, dict):
            keep = (
                "harvest",
                "split",
                "epochs",
                "batch_size",
                "lr",
                "vf_coef",
                "seed",
                "max_train_rows",
                "max_val_rows",
                "n_train_games",
                "n_val_games",
            )
            config = {k: metrics[k] for k in keep if k in metrics}
    data["config"] = config
    if config is None:
        missing.append("config")

    data["rewards"] = extract_rewards(config) if config else None
    if data["rewards"] is None:
        missing.append("rewards")

    parent_path = None
    if isinstance(config, dict):
        parent_path = config.get("init_checkpoint") or config.get("init")
    summary = _read_json(dest / "summary.json")
    if parent_path is None and isinstance(summary, dict):
        parent_path = summary.get("init_checkpoint")
    sidecar = _sidecar_for_source_run(dest, checkpoints_dir, runs_root)
    if parent_path is None and sidecar:
        warm = sidecar.get("warm_start") or sidecar.get("init")
        if isinstance(warm, str):
            # "runs/checkpoints/v0.1.0_cb0a0c7.pt (v0.1.0)" — path is the first token
            # that looks like a file, not a prose sentence.
            parent_path = _path_from_prose(warm)
    if parent_path:
        data["parent"] = describe_checkpoint(
            parent_path, checkpoints_dir=checkpoints_dir
        )
    else:
        data["parent"] = None
        missing.append("parent")

    updates = _updates_from_artifacts(dest, summary, sidecar)
    data["updates_completed"] = updates
    if updates is None:
        missing.append("updates_completed")

    final_path = _final_checkpoint_path(dest)
    if final_path is not None:
        data["final_checkpoint"] = describe_checkpoint(
            final_path, checkpoints_dir=checkpoints_dir
        )
    else:
        missing.append("final_checkpoint")

    started, ended, commit = _times_and_commit_from_artifacts(dest, sidecar)
    data["started_at"] = started
    data["ended_at"] = ended
    if started is None:
        missing.append("started_at")
    if ended is None:
        missing.append("ended_at")
    if commit is not None:
        data["git"] = {"commit": commit, "dirty": None}
    else:
        missing.append("git.commit")
    # command is almost never recorded in historical artifacts.
    data["command"] = None
    missing.append("command")
    data["status"] = "completed" if (ended or updates is not None or final_path) else None
    if data["status"] is None:
        missing.append("status")

    evals = _evals_from_run_dir(dest)
    data["evals"] = evals
    return data, missing


def iter_run_dirs(runs_root: PathLike) -> List[Path]:
    """Training run directories under ``runs/``, skipping non-run folders."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    found: List[Path] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        if entry.name.startswith(".") or entry.name in SKIP_RUN_DIR_NAMES:
            continue
        if _looks_like_run_dir(entry):
            found.append(entry)
    return found


def _looks_like_run_dir(path: Path) -> bool:
    markers = (
        MANIFEST_NAME,
        "config.yaml",
        "history.jsonl",
        "state.json",
        "summary.json",
        "train.log",
        "latest.pt",
        "final.pt",
        "train",
    )
    for name in markers:
        if (path / name).exists():
            return True
    try:
        if any(path.glob("update_*.pt")):
            return True
    except OSError:
        return False
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_root(start: Optional[PathLike]) -> Optional[Path]:
    if start is not None:
        p = Path(start)
        if (p / ".git").exists():
            return p
        try:
            out = _git(p, ["rev-parse", "--show-toplevel"])
            return Path(out) if out else None
        except (OSError, subprocess.CalledProcessError):
            return None
    # Prefer the caller's cwd (the checkout they launched from).
    try:
        out = _git(Path.cwd(), ["rev-parse", "--show-toplevel"])
        return Path(out) if out else None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git(cwd: Path, args: List[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _safe_resolve(path: PathLike) -> Optional[Path]:
    try:
        return Path(path).resolve()
    except OSError:
        return None


def _resolve_maybe(path: PathLike, start: Optional[Path] = None) -> Optional[Path]:
    p = Path(str(path).split()[0])
    if p.is_file() or p.is_dir():
        return _safe_resolve(p)
    if start is not None:
        alt = start / p
        if alt.exists():
            return _safe_resolve(alt)
        # "runs/foo" relative to a repo root.
        if p.parts and p.parts[0] == "runs":
            alt = start / p
            if alt.exists():
                return _safe_resolve(alt)
    return _safe_resolve(p) if p.exists() else None


def _checkpoints_dir(
    hint: Optional[PathLike], explicit: Optional[PathLike]
) -> Optional[Path]:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_dir() else None
    if hint is not None:
        p = Path(hint)
        for folder in [p] + (list(p.parents) if p.exists() or p.parent.exists() else []):
            if folder.name == CHECKPOINTS_DIRNAME and folder.is_dir():
                return folder
            cand = folder / "runs" / CHECKPOINTS_DIRNAME
            if cand.is_dir():
                return cand
            cand = folder / CHECKPOINTS_DIRNAME
            if cand.is_dir():
                return cand
    cwd_cand = Path.cwd() / "runs" / CHECKPOINTS_DIRNAME
    if cwd_cand.is_dir():
        return cwd_cand
    return None


def _runs_root_from(path: Path) -> Optional[Path]:
    for folder in [path] + list(path.parents):
        if folder.name == "runs":
            return folder
    return None


def _tag_from_filename(name: str) -> Optional[str]:
    # v0.4.0_8ad2530 or v0.4.0_8ad2530.pt or v0.4.0
    stem = name[:-3] if name.endswith(".pt") else name
    match = re.match(r"(v\d+\.\d+\.\d+)", stem)
    return match.group(1) if match else None


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_yaml(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        return None
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    return raw if isinstance(raw, dict) else None


def _sidecar_for_weights(weights: Path, ckpt_dir: Path) -> Optional[Path]:
    stem = weights.stem
    direct = ckpt_dir / f"{stem}.json"
    if direct.is_file():
        return direct
    for sidecar in ckpt_dir.glob("*.json"):
        meta = _read_json(sidecar)
        if not isinstance(meta, dict):
            continue
        named = meta.get("weights")
        if named and Path(str(named)).name == weights.name:
            return sidecar
        if meta.get("sha256") and weights.is_file():
            if sha256_file(weights) == meta.get("sha256"):
                return sidecar
    return None


def _sidecar_for_source_run(
    run_dir: Path,
    checkpoints_dir: Optional[PathLike],
    runs_root: Optional[PathLike],
) -> Optional[Dict[str, Any]]:
    ckpt_dir = _checkpoints_dir(run_dir, checkpoints_dir)
    if ckpt_dir is None:
        return None
    run_name = run_dir.name
    resolved = _safe_resolve(run_dir)
    for sidecar in sorted(ckpt_dir.glob("v*.json")):
        meta = _read_json(sidecar)
        if not isinstance(meta, dict):
            continue
        source = meta.get("source_run")
        if not source:
            continue
        token = str(source).split()[0]
        if Path(token).name == run_name:
            return meta
        if resolved and _resolve_maybe(token, start=run_dir.parent.parent):
            if _safe_resolve(_resolve_maybe(token, start=run_dir.parent.parent) or token) == resolved:
                return meta
    return None


def _path_from_prose(text: str) -> Optional[str]:
    """First path-like token in a sidecar blurb, or None if none looks like a file."""
    for token in text.replace("(", " ").replace(")", " ").split():
        if token.endswith(".pt") or token.startswith("runs/"):
            return token
    return None


def _updates_from_artifacts(
    run_dir: Path,
    summary: Optional[Any],
    sidecar: Optional[Mapping[str, Any]],
) -> Optional[int]:
    state = _read_json(run_dir / "state.json")
    if isinstance(state, dict) and state.get("update") is not None:
        try:
            return int(state["update"])
        except (TypeError, ValueError):
            pass
    if isinstance(summary, dict):
        for key in ("updates_run", "updates", "update"):
            if summary.get(key) is not None:
                try:
                    return int(summary[key])
                except (TypeError, ValueError):
                    pass
    last = _last_history_row(run_dir / "history.jsonl")
    if isinstance(last, dict):
        for key in ("update", "updates", "update_end"):
            if last.get(key) is not None:
                try:
                    return int(last[key])
                except (TypeError, ValueError):
                    pass
    highest = _highest_update_pt(run_dir)
    if highest is not None:
        return highest[0]
    return None


def _last_history_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            return row
    return None


def _highest_update_pt(run_dir: Path) -> Optional[tuple[int, Path]]:
    best: Optional[tuple[int, Path]] = None
    try:
        files = list(run_dir.glob("update_*.pt"))
        files += list((run_dir / "train").glob("update_*.pt")) if (run_dir / "train").is_dir() else []
    except OSError:
        return None
    for path in files:
        match = re.search(r"update_(\d+)\.pt$", path.name)
        if not match:
            continue
        n = int(match.group(1))
        if best is None or n > best[0]:
            best = (n, path)
    return best


def _final_checkpoint_path(run_dir: Path) -> Optional[Path]:
    for name in ("latest.pt", "final.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    train_best = run_dir / "train" / "best.pt"
    if train_best.is_file():
        return train_best
    train_latest = run_dir / "train" / "latest.pt"
    if train_latest.is_file():
        return train_latest
    highest = _highest_update_pt(run_dir)
    if highest is not None:
        return highest[1]
    return None


def _times_and_commit_from_artifacts(
    run_dir: Path, sidecar: Optional[Mapping[str, Any]]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    started = ended = commit = None
    if sidecar:
        trained = sidecar.get("trained")
        if isinstance(trained, str):
            started, ended = _parse_trained_range(trained)
        raw_commit = sidecar.get("commit_full") or sidecar.get("commit")
        if isinstance(raw_commit, str) and raw_commit.strip():
            commit = raw_commit.strip()
    # metrics.json may record elapsed_sec but not a wall-clock start; skip.
    return started, ended, commit


def _parse_trained_range(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse '2026-11-23 08:43-09:15 CST' or '2026-11-23 06:21:56-06:45:32 CST'."""
    text = text.strip()
    match = re.match(
        r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}(?::\d{2})?)\s*-\s*"
        r"(\d{2}:\d{2}(?::\d{2})?)\s*([A-Za-z0-9_+\-]+)?",
        text,
    )
    if not match:
        return None, None
    day, start, end, zone = match.groups()
    # Keep the original zone label; do not invent an offset.
    suffix = f" {zone}" if zone else ""
    return f"{day} {start}{suffix}", f"{day} {end}{suffix}"


def _evals_from_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    evals: List[Dict[str, Any]] = []
    for name in ("eval_final", "eval_selfplay", "eval_unmasked", "eval_u350", "eval_u500"):
        summary = _read_json(run_dir / name / "summary.json")
        if isinstance(summary, dict):
            evals.append(
                {
                    "recorded_at": None,
                    "name": name,
                    "checkpoint": None,
                    "summary": compact_eval(summary),
                    "source": str(run_dir / name / "summary.json"),
                }
            )
    return evals


def _drop_keys(obj: Any, drop: Iterable[str]) -> Any:
    drop_set = set(drop)
    if isinstance(obj, dict):
        return {k: _drop_keys(v, drop_set) for k, v in obj.items() if k not in drop_set}
    if isinstance(obj, list):
        return [_drop_keys(v, drop_set) for v in obj]
    return obj
