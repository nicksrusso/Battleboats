"""Build runs/index.csv and runs/index.sqlite from run.json manifests.

    poetry run python scripts/runs_index.py
    poetry run python scripts/runs_index.py --runs-root /path/to/Battleboats/runs
    poetry run python scripts/runs_index.py --backfill
    poetry run python scripts/runs_index.py --format mermaid

Scans ``runs/*/run.json`` and the tagged sidecars in ``runs/checkpoints/``.
Prints a lineage tree from parent links (markdown by default). ``--backfill``
writes a best-effort ``run.json`` for existing run directories that do not
already have one. A run whose ``train.pid`` process is still alive is skipped.
Backfill never modifies any file other than ``run.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from battleboats.training.manifest import (
    KIND_TAGGED,
    backfill_run,
    iter_run_dirs,
    load_manifest,
    match_checkpoint_tag,
)

INDEX_CSV = "index.csv"
INDEX_SQLITE = "index.sqlite"

CSV_COLUMNS = (
    "run_name",
    "kind",
    "status",
    "backfilled",
    "started_at",
    "ended_at",
    "git_commit",
    "git_dirty",
    "parent_path",
    "parent_sha256",
    "parent_tag",
    "updates_completed",
    "final_checkpoint_path",
    "final_checkpoint_sha256",
    "final_checkpoint_tag",
    "tag",
    "source_run",
    "run_dir",
    "command",
)


def default_runs_root() -> Path:
    here = Path(__file__).resolve()
    repo = here.parents[1]
    cand = repo / "runs"
    if cand.is_dir():
        return cand
    return Path.cwd() / "runs"


def load_sidecar_rows(checkpoints_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not checkpoints_dir.is_dir():
        return rows
    for sidecar in sorted(checkpoints_dir.glob("v*.json")):
        try:
            meta = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        version = meta.get("version") or sidecar.stem
        weights = meta.get("weights")
        sha = meta.get("sha256")
        parent_path = None
        for key in ("warm_start", "init"):
            raw = meta.get(key)
            if isinstance(raw, str) and (".pt" in raw or raw.startswith("runs/")):
                # First path-like token.
                for token in raw.replace("(", " ").replace(")", " ").split():
                    if token.endswith(".pt") or token.startswith("runs/"):
                        parent_path = token
                        break
            if parent_path:
                break
        parent_tag = (
            match_checkpoint_tag(parent_path, checkpoints_dir=checkpoints_dir)
            if parent_path
            else None
        )
        source = meta.get("source_run")
        source_run = str(source).split()[0] if source else None
        rows.append(
            {
                "run_name": str(version),
                "kind": KIND_TAGGED,
                "status": "tagged",
                "backfilled": False,
                "started_at": _trained_start(meta.get("trained")),
                "ended_at": _trained_end(meta.get("trained")),
                "git_commit": meta.get("commit_full") or meta.get("commit"),
                "git_dirty": None,
                "parent_path": parent_path,
                "parent_sha256": None,
                "parent_tag": parent_tag,
                "updates_completed": meta.get("update"),
                "final_checkpoint_path": weights,
                "final_checkpoint_sha256": sha,
                "final_checkpoint_tag": str(version),
                "tag": str(version),
                "source_run": source_run,
                "run_dir": str(sidecar),
                "command": None,
                "_node_id": f"tag:{version}",
            }
        )
    return rows


def load_run_rows(runs_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in iter_run_dirs(runs_root):
        data = load_manifest(run_dir)
        if data is None:
            continue
        parent = data.get("parent") or {}
        if not isinstance(parent, dict):
            parent = {}
        final = data.get("final_checkpoint") or {}
        if not isinstance(final, dict):
            final = {}
        git = data.get("git") or {}
        if not isinstance(git, dict):
            git = {}
        rows.append(
            {
                "run_name": data.get("run_name") or run_dir.name,
                "kind": data.get("kind"),
                "status": data.get("status"),
                "backfilled": bool(data.get("backfilled")),
                "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"),
                "git_commit": git.get("commit"),
                "git_dirty": git.get("dirty"),
                "parent_path": parent.get("path"),
                "parent_sha256": parent.get("sha256"),
                "parent_tag": parent.get("tag"),
                "updates_completed": data.get("updates_completed"),
                "final_checkpoint_path": final.get("path"),
                "final_checkpoint_sha256": final.get("sha256"),
                "final_checkpoint_tag": final.get("tag"),
                "tag": final.get("tag"),
                "source_run": None,
                "run_dir": str(run_dir),
                "command": data.get("command"),
                "_node_id": f"run:{run_dir.name}",
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: _csv_value(row.get(k)) for k in CSV_COLUMNS}
            writer.writerow(out)


def write_sqlite(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        cols = ", ".join(f"{name} TEXT" for name in CSV_COLUMNS)
        conn.execute(f"CREATE TABLE runs ({cols})")
        placeholders = ", ".join("?" for _ in CSV_COLUMNS)
        for row in rows:
            values = [_csv_value(row.get(k)) for k in CSV_COLUMNS]
            conn.execute(
                f"INSERT INTO runs ({', '.join(CSV_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(run_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_tag ON runs(tag)")
        conn.commit()
    finally:
        conn.close()


GENERIC_CHECKPOINT_NAMES = frozenset(
    {"latest.pt", "final.pt", "best.pt", "in_progress.pt"}
)


def lineage_edges(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """Parent -> child edges using tags, source_run, and checkpoint paths."""
    by_id = {row["_node_id"]: row for row in rows if "_node_id" in row}
    by_name: Dict[str, List[str]] = {}
    by_tag: Dict[str, List[str]] = {}
    by_ckpt: Dict[str, List[str]] = {}
    by_sha: Dict[str, List[str]] = {}
    for node_id, row in by_id.items():
        by_name.setdefault(str(row["run_name"]), []).append(node_id)
        if row.get("tag"):
            by_tag.setdefault(str(row["tag"]), []).append(node_id)
        if row.get("final_checkpoint_path"):
            by_ckpt.setdefault(_norm_ckpt(row["final_checkpoint_path"]), []).append(node_id)
        if row.get("final_checkpoint_sha256"):
            by_sha.setdefault(str(row["final_checkpoint_sha256"]), []).append(node_id)
        # Tagged nodes also register their sidecar weights under the tag.
        if row.get("kind") == KIND_TAGGED and row.get("run_name"):
            by_tag.setdefault(str(row["run_name"]), []).append(node_id)

    def resolve_parent(row: Mapping[str, Any]) -> Optional[str]:
        # A tagged checkpoint is produced by its source run; that link is the
        # primary parent. warm_start is already the source run's parent.
        if row.get("kind") == KIND_TAGGED and row.get("source_run"):
            source_name = Path(str(row["source_run"])).name
            hits = by_name.get(source_name) or []
            if hits:
                return hits[0]
        # Prefer a tagged official parent when the parent tag is known.
        if row.get("parent_tag"):
            hits = by_tag.get(str(row["parent_tag"])) or []
            tagged = [n for n in hits if by_id[n].get("kind") == KIND_TAGGED]
            if tagged:
                return tagged[0]
            if hits:
                return hits[0]
        if row.get("parent_sha256"):
            hits = by_sha.get(str(row["parent_sha256"])) or []
            if hits:
                return hits[0]
        if row.get("parent_path"):
            key = _norm_ckpt(row["parent_path"])
            hits = by_ckpt.get(key) or []
            if hits:
                return hits[0]
            # Distinctive filename only (v0.4.0_8ad2530.pt), never latest.pt.
            name = Path(str(row["parent_path"])).name
            if name not in GENERIC_CHECKPOINT_NAMES:
                tagged = [
                    n
                    for n, other in by_id.items()
                    if Path(str(other.get("final_checkpoint_path") or "")).name == name
                    and other.get("kind") == KIND_TAGGED
                ]
                if tagged:
                    return tagged[0]
        return None

    edges: List[Tuple[str, str]] = []
    seen = set()
    for node_id, row in by_id.items():
        parent_id = resolve_parent(row)
        if parent_id and parent_id != node_id:
            pair = (parent_id, node_id)
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
    return edges


def format_lineage_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    edges = lineage_edges(rows)
    by_id = {row["_node_id"]: row for row in rows if "_node_id" in row}
    children: Dict[str, List[str]] = {}
    has_parent = set()
    for parent, child in edges:
        children.setdefault(parent, []).append(child)
        has_parent.add(child)
    roots = [n for n in by_id if n not in has_parent]
    roots.sort(key=lambda n: _label(by_id[n]))
    for kids in children.values():
        kids.sort(key=lambda n: _label(by_id[n]))

    lines = ["# Training lineage", ""]
    if not by_id:
        lines.append("No runs or tagged checkpoints found.")
        return "\n".join(lines) + "\n"

    def walk(node_id: str, prefix: str, is_last: bool) -> None:
        branch = "└─ " if is_last else "├─ "
        if prefix == "":
            lines.append(f"- {_label(by_id[node_id])}")
        else:
            lines.append(f"{prefix}{branch}{_label(by_id[node_id])}")
        kids = children.get(node_id, [])
        next_prefix = "" if prefix == "" else prefix + ("   " if is_last else "│  ")
        if prefix == "":
            next_prefix = "  "
        for i, kid in enumerate(kids):
            walk(kid, next_prefix, i == len(kids) - 1)

    for i, root in enumerate(roots):
        walk(root, "", True)
        if i != len(roots) - 1:
            lines.append("")
    return "\n".join(lines) + "\n"


def format_lineage_mermaid(rows: Sequence[Mapping[str, Any]]) -> str:
    edges = lineage_edges(rows)
    by_id = {row["_node_id"]: row for row in rows if "_node_id" in row}
    lines = ["```mermaid", "flowchart LR"]
    used = set()
    for parent, child in edges:
        used.add(parent)
        used.add(child)
        lines.append(
            f"    {_mermaid_id(parent)}[{_mermaid_label(by_id[parent])}] --> "
            f"{_mermaid_id(child)}[{_mermaid_label(by_id[child])}]"
        )
    # Isolated nodes, if any.
    for node_id, row in by_id.items():
        if node_id not in used:
            lines.append(f"    {_mermaid_id(node_id)}[{_mermaid_label(row)}]")
    lines.append("```")
    return "\n".join(lines) + "\n"


def backfill_all(
    runs_root: Path,
    *,
    overwrite: bool = False,
    checkpoints_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    results = []
    ckpt = checkpoints_dir or (runs_root / "checkpoints")
    for run_dir in iter_run_dirs(runs_root):
        results.append(
            backfill_run(
                run_dir,
                runs_root=runs_root,
                checkpoints_dir=ckpt if ckpt.is_dir() else None,
                overwrite=overwrite,
            )
        )
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory that holds run folders (default: <repo>/runs).",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Write missing run.json files from existing artifacts.",
    )
    parser.add_argument(
        "--overwrite-backfilled",
        action="store_true",
        help="Rewrite run.json files that were previously backfilled.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "mermaid", "none"),
        default="markdown",
        help="Lineage tree format printed to stdout (default: markdown).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Where to write index.csv and index.sqlite (default: runs-root).",
    )
    args = parser.parse_args(argv)

    runs_root = (args.runs_root or default_runs_root()).resolve()
    if not runs_root.is_dir():
        print(f"runs root not found: {runs_root}", file=sys.stderr)
        return 2
    checkpoints_dir = runs_root / "checkpoints"
    index_dir = args.index_dir or runs_root

    if args.backfill:
        results = backfill_all(
            runs_root,
            overwrite=args.overwrite_backfilled,
            checkpoints_dir=checkpoints_dir if checkpoints_dir.is_dir() else None,
        )
        written = sum(1 for r in results if r["status"] == "written")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        in_progress = [
            r["run_dir"] for r in results if r.get("reason") == "in_progress"
        ]
        print(
            f"backfill: wrote {written}, skipped {skipped}, scanned {len(results)}",
            flush=True,
        )
        if in_progress:
            print("skipped in-progress:", ", ".join(in_progress), flush=True)
        # Coverage: how often fields were missing on written manifests.
        missing_counts: Dict[str, int] = {}
        for rec in results:
            if rec["status"] != "written":
                continue
            for field in rec.get("missing") or []:
                missing_counts[field] = missing_counts.get(field, 0) + 1
        if missing_counts:
            print("backfill missing fields (written runs):", flush=True)
            for field, count in sorted(missing_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {field}: {count}", flush=True)

    run_rows = load_run_rows(runs_root)
    sidecar_rows = load_sidecar_rows(checkpoints_dir)
    rows = run_rows + sidecar_rows
    write_csv(index_dir / INDEX_CSV, rows)
    write_sqlite(index_dir / INDEX_SQLITE, rows)
    print(
        f"index: {len(run_rows)} runs, {len(sidecar_rows)} tagged checkpoints -> "
        f"{index_dir / INDEX_CSV} and {index_dir / INDEX_SQLITE}",
        flush=True,
    )
    if args.format == "markdown":
        print(format_lineage_markdown(rows), end="")
    elif args.format == "mermaid":
        print(format_lineage_mermaid(rows), end="")
    return 0


def _csv_value(value: Any) -> Optional[str]:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _norm_ckpt(path: Any) -> str:
    """Stable path key from ``runs/`` onward. Basename alone is not unique."""
    p = Path(str(path))
    parts = p.parts
    if "runs" in parts:
        return "/".join(parts[parts.index("runs"):]).lower()
    if "checkpoints" in parts:
        return "runs/" + "/".join(parts[parts.index("checkpoints"):]).lower()
    if len(parts) >= 2:
        return "/".join(parts[-2:]).lower()
    return p.name.lower()


def _label(row: Mapping[str, Any]) -> str:
    name = str(row.get("run_name") or row.get("_node_id"))
    kind = row.get("kind") or "?"
    extras = []
    if row.get("tag") and row.get("kind") != KIND_TAGGED:
        extras.append(str(row["tag"]))
    if row.get("updates_completed") is not None and row.get("kind") != KIND_TAGGED:
        extras.append(f"u{row['updates_completed']}")
    extra = f" ({', '.join(extras)})" if extras else ""
    return f"{name} [{kind}]{extra}"


def _mermaid_id(node_id: str) -> str:
    return (
        node_id.replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _mermaid_label(row: Mapping[str, Any]) -> str:
    # Mermaid node text: quote if needed, keep short.
    text = _label(row).replace('"', "'")
    return f'"{text}"'


def _trained_start(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    from battleboats.training.manifest import _parse_trained_range

    start, _end = _parse_trained_range(text)
    return start


def _trained_end(text: Any) -> Optional[str]:
    if not isinstance(text, str):
        return None
    from battleboats.training.manifest import _parse_trained_range

    _start, end = _parse_trained_range(text)
    return end


if __name__ == "__main__":
    raise SystemExit(main())
