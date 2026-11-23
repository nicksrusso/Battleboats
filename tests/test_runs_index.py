"""Index CSV/SQLite writer and lineage tree from parent links."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from battleboats.training.manifest import sha256_file, start_run
from scripts.runs_index import (
    backfill_all,
    format_lineage_markdown,
    format_lineage_mermaid,
    lineage_edges,
    load_run_rows,
    load_sidecar_rows,
    main,
    write_csv,
    write_sqlite,
)


def _ckpt(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _seed_lineage(tmp_path: Path) -> Path:
    """v0.4.0 <- sandbox_buydiv_v0.4.1 <- selfplay_v0.4.1_winloss."""
    runs = tmp_path / "runs"
    ckpt_dir = runs / "checkpoints"
    v040 = _ckpt(ckpt_dir / "v0.4.0_8ad2530.pt", b"v040-weights")
    (ckpt_dir / "v0.4.0_8ad2530.json").write_text(
        json.dumps(
            {
                "version": "v0.4.0",
                "weights": "runs/checkpoints/v0.4.0_8ad2530.pt",
                "sha256": sha256_file(v040),
                "source_run": "runs/bc_tactical_v040",
                "commit": "8ad2530",
                "trained": "2026-11-23 08:43-09:15 CST",
            }
        )
    )
    bc = runs / "bc_tactical_v040"
    start_run(bc, kind="bc", config={"epochs": 8}, argv=["train_tactical_bc"])
    buydiv = runs / "sandbox_buydiv_v0.4.1"
    buydiv_ckpt = _ckpt(buydiv / "update_300.pt", b"buydiv-final")
    start_run(
        buydiv,
        kind="buydiv",
        config={"init_checkpoint": str(v040)},
        init_checkpoint=v040,
        argv=["train_buydiv"],
    )
    from battleboats.training.manifest import finish_run

    finish_run(buydiv, final_checkpoint=buydiv_ckpt, updates_completed=300)
    winloss = runs / "selfplay_v0.4.1_winloss"
    start_run(
        winloss,
        kind="selfplay",
        config={"init_checkpoint": str(buydiv_ckpt)},
        init_checkpoint=buydiv_ckpt,
        argv=["train_selfplay"],
    )
    finish_run(winloss, final_checkpoint=_ckpt(winloss / "latest.pt", b"sp"), updates_completed=12)
    return runs


def test_index_csv_and_sqlite(tmp_path: Path) -> None:
    runs = _seed_lineage(tmp_path)
    rows = load_run_rows(runs) + load_sidecar_rows(runs / "checkpoints")
    csv_path = tmp_path / "index.csv"
    sqlite_path = tmp_path / "index.sqlite"
    write_csv(csv_path, rows)
    write_sqlite(sqlite_path, rows)
    with csv_path.open() as fh:
        table = list(csv.DictReader(fh))
    names = {row["run_name"] for row in table}
    assert "v0.4.0" in names
    assert "sandbox_buydiv_v0.4.1" in names
    assert "selfplay_v0.4.1_winloss" in names
    assert "bc_tactical_v040" in names
    buydiv = next(r for r in table if r["run_name"] == "sandbox_buydiv_v0.4.1")
    assert buydiv["parent_tag"] == "v0.4.0"
    assert buydiv["updates_completed"] == "300"
    conn = sqlite3.connect(str(sqlite_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert count == len(table)
        tag = conn.execute(
            "SELECT parent_tag FROM runs WHERE run_name = ?",
            ("sandbox_buydiv_v0.4.1",),
        ).fetchone()
        assert tag == ("v0.4.0",)
    finally:
        conn.close()


def test_lineage_tree_follows_parent_links(tmp_path: Path) -> None:
    runs = _seed_lineage(tmp_path)
    rows = load_run_rows(runs) + load_sidecar_rows(runs / "checkpoints")
    edges = lineage_edges(rows)
    labels = {(a.split(":")[-1], b.split(":")[-1]) for a, b in edges}
    # bc_tactical_v040 -> v0.4.0 (tagged, via source_run)
    # v0.4.0 -> sandbox_buydiv_v0.4.1 (parent tag)
    # sandbox_buydiv_v0.4.1 -> selfplay_v0.4.1_winloss (parent path)
    assert ("bc_tactical_v040", "v0.4.0") in labels
    assert ("v0.4.0", "sandbox_buydiv_v0.4.1") in labels
    assert ("sandbox_buydiv_v0.4.1", "selfplay_v0.4.1_winloss") in labels
    md = format_lineage_markdown(rows)
    assert "v0.4.0 [tagged]" in md
    assert "sandbox_buydiv_v0.4.1 [buydiv]" in md
    assert "selfplay_v0.4.1_winloss [selfplay]" in md
    mermaid = format_lineage_mermaid(rows)
    assert mermaid.startswith("```mermaid")
    assert "-->" in mermaid
    assert "selfplay_v0.4.1_winloss" in mermaid


def test_main_writes_index_and_prints_tree(tmp_path: Path, capsys) -> None:
    runs = _seed_lineage(tmp_path)
    rc = main(["--runs-root", str(runs), "--format", "markdown"])
    assert rc == 0
    assert (runs / "index.csv").is_file()
    assert (runs / "index.sqlite").is_file()
    out = capsys.readouterr().out
    assert "Training lineage" in out
    assert "v0.4.0" in out


def test_backfill_all_skips_in_progress(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    done = runs / "selfplay_done"
    done.mkdir(parents=True)
    (done / "config.yaml").write_text("updates: 2\n")
    (done / "state.json").write_text(json.dumps({"update": 2}))
    live = runs / "selfplay_live"
    live.mkdir()
    (live / "config.yaml").write_text("updates: 9\n")
    (live / "train.pid").write_text(f"{os_get_live_pid()}\n")
    try:
        results = backfill_all(runs)
    finally:
        # leave the live pid file; process may be this pytest worker (alive).
        pass
    statuses = {Path(r["run_dir"]).name: r["status"] for r in results}
    assert statuses["selfplay_done"] == "written"
    assert statuses["selfplay_live"] == "skipped"
    assert (done / "run.json").is_file()
    assert not (live / "run.json").exists()


def os_get_live_pid() -> int:
    return os_getpid()


def os_getpid() -> int:
    import os

    return os.getpid()


def test_latest_pt_parent_does_not_cross_link(tmp_path: Path) -> None:
    """Two runs both ending in latest.pt must not be joined by basename."""
    from battleboats.training.manifest import finish_run, start_run

    runs = tmp_path / "runs"
    a = runs / "sandboxes_v002"
    b = runs / "sandboxes_v006"
    child = runs / "selfplay_from_v006"
    a_ckpt = _ckpt(a / "latest.pt", b"aaa")
    b_ckpt = _ckpt(b / "latest.pt", b"bbb")
    start_run(a, kind="sandbox", config={}, argv=["a"])
    finish_run(a, final_checkpoint=a_ckpt, updates_completed=2)
    start_run(b, kind="sandbox", config={}, argv=["b"])
    finish_run(b, final_checkpoint=b_ckpt, updates_completed=6)
    start_run(
        child,
        kind="selfplay",
        config={"init_checkpoint": "runs/sandboxes_v006/latest.pt"},
        init_checkpoint=b_ckpt,
        argv=["c"],
    )
    # Force the stored parent path to the relative form used in real configs,
    # not the absolute tmp path start_run recorded.
    import json
    man = json.loads((child / "run.json").read_text())
    man["parent"] = {"path": "runs/sandboxes_v006/latest.pt", "sha256": None, "tag": None}
    (child / "run.json").write_text(json.dumps(man, indent=2))
    rows = load_run_rows(runs)
    labels = {(x.split(":")[-1], y.split(":")[-1]) for x, y in lineage_edges(rows)}
    assert ("sandboxes_v006", "selfplay_from_v006") in labels
    assert ("sandboxes_v002", "selfplay_from_v006") not in labels
    # v0.1.0-style: a tagged node whose warm_start is runs/sandboxes_v006/latest.pt
    # plus a source_run should parent to the source run, not the warm start.
    ckpt_dir = runs / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "v0.1.0_abc.json").write_text(
        json.dumps(
            {
                "version": "v0.1.0",
                "weights": "runs/checkpoints/v0.1.0_abc.pt",
                "source_run": "runs/selfplay_from_v006",
                "warm_start": "runs/sandboxes_v006/latest.pt",
            }
        )
    )
    rows = load_run_rows(runs) + load_sidecar_rows(ckpt_dir)
    labels = {(x.split(":")[-1], y.split(":")[-1]) for x, y in lineage_edges(rows)}
    assert ("selfplay_from_v006", "v0.1.0") in labels
    assert ("sandboxes_v002", "v0.1.0") not in labels
