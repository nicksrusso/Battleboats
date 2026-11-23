"""Manifest writer, eval append, and best-effort backfill."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import pytest
import yaml

from battleboats.training.manifest import (
    MANIFEST_NAME,
    ManifestSession,
    append_eval,
    backfill_run,
    command_line,
    compact_eval,
    describe_checkpoint,
    extract_rewards,
    finish_run,
    infer_kind,
    is_run_in_progress,
    load_manifest,
    match_checkpoint_tag,
    pid_is_alive,
    reconstruct_from_artifacts,
    sha256_file,
    start_run,
    to_jsonable,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _tiny_checkpoint(path: Path, payload: bytes = b"ckpt-v1") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_start_and_finish_write_expected_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "selfplay_demo"
    init = _tiny_checkpoint(tmp_path / "init.pt", b"parent-weights")
    cfg = {
        "run_dir": str(run_dir),
        "updates": 10,
        "init_checkpoint": str(init),
        "reward": {"draw": -0.15},
        "reward_shaping": {"enabled": False},
    }
    data = start_run(
        run_dir,
        kind="selfplay",
        config=cfg,
        init_checkpoint=init,
        argv=["scripts/train_selfplay.py", "--config", "configs/selfplay.yaml"],
        repo_root=tmp_path,
    )
    assert data["run_name"] == "selfplay_demo"
    assert data["kind"] == "selfplay"
    assert data["backfilled"] is False
    assert data["status"] == "running"
    assert data["started_at"]
    assert data["ended_at"] is None
    assert data["config"]["updates"] == 10
    assert data["command"] == "scripts/train_selfplay.py --config configs/selfplay.yaml"
    assert data["parent"]["path"] == str(init)
    assert data["parent"]["sha256"] == sha256_file(init)
    assert data["rewards"]["reward"] == {"draw": -0.15}
    assert data["evals"] == []
    # git.commit is null here: tmp_path is not a git repo.
    assert data["git"]["commit"] is None

    final = _tiny_checkpoint(run_dir / "latest.pt", b"final-weights")
    finished = finish_run(run_dir, final_checkpoint=final, updates_completed=10)
    assert finished["status"] == "completed"
    assert finished["ended_at"]
    assert finished["updates_completed"] == 10
    assert finished["final_checkpoint"]["path"] == str(final)
    assert finished["final_checkpoint"]["sha256"] == sha256_file(final)
    assert finished["started_at"] == data["started_at"]
    on_disk = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert on_disk["updates_completed"] == 10


def test_resume_preserves_started_at_and_evals(tmp_path: Path) -> None:
    run_dir = tmp_path / "selfplay_resume"
    start_run(run_dir, kind="selfplay", config={"updates": 1}, argv=["train"])
    first = load_manifest(run_dir)
    assert first is not None
    append_eval({"score": 0.5}, run_dir=run_dir, name="mid")
    again = start_run(
        run_dir, kind="selfplay", config={"updates": 2}, argv=["train", "--resume"]
    )
    assert again["started_at"] == first["started_at"]
    assert again["status"] == "running"
    assert len(again["evals"]) == 1
    assert again["evals"][0]["summary"]["score"] == 0.5
    assert again["config"]["updates"] == 2


def test_append_eval_is_noop_without_manifest(tmp_path: Path) -> None:
    ckpt = _tiny_checkpoint(tmp_path / "orphan" / "latest.pt")
    assert append_eval({"wins": 1}, checkpoint=ckpt) is None
    assert not (tmp_path / "orphan" / MANIFEST_NAME).exists()


def test_append_eval_to_checkpoint_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "buydiv_demo"
    ckpt = _tiny_checkpoint(run_dir / "update_300.pt")
    start_run(run_dir, kind="buydiv", config={}, argv=["train"])
    dest = append_eval(
        {"diversity": 0.48, "games": [{"id": 1}], "episodes_raw": [1, 2]},
        checkpoint=ckpt,
        name="unmasked",
        runs_root=tmp_path / "runs",
    )
    assert dest == run_dir / MANIFEST_NAME
    data = load_manifest(run_dir)
    assert data is not None
    entry = data["evals"][0]
    assert entry["name"] == "unmasked"
    assert entry["checkpoint"] == str(ckpt)
    assert entry["summary"]["diversity"] == 0.48
    assert "games" not in entry["summary"]
    assert "episodes_raw" not in entry["summary"]


def test_append_eval_follows_tagged_sidecar_source_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    source = runs / "bc_tactical_v040"
    source.mkdir(parents=True)
    start_run(source, kind="bc", config={"epochs": 8}, argv=["train_tactical_bc"])
    weights = _tiny_checkpoint(runs / "checkpoints" / "v0.4.0_abc1234.pt", b"bc-weights")
    (runs / "checkpoints" / "v0.4.0_abc1234.json").write_text(
        json.dumps(
            {
                "version": "v0.4.0",
                "weights": "runs/checkpoints/v0.4.0_abc1234.pt",
                "sha256": sha256_file(weights),
                "source_run": "runs/bc_tactical_v040",
            }
        )
    )
    dest = append_eval(
        {"vs_tactical": {"wld": [1, 2, 3]}},
        checkpoint=weights,
        name="sampled",
        checkpoints_dir=runs / "checkpoints",
        runs_root=runs,
    )
    assert dest == source / MANIFEST_NAME
    data = load_manifest(source)
    assert data is not None
    assert data["evals"][0]["name"] == "sampled"


def test_parent_tag_matches_checkpoint_sidecar(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    weights = _tiny_checkpoint(runs / "checkpoints" / "v0.4.0_abc1234.pt", b"tag-w")
    digest = sha256_file(weights)
    (runs / "checkpoints" / "v0.4.0_abc1234.json").write_text(
        json.dumps(
            {
                "version": "v0.4.0",
                "weights": "runs/checkpoints/v0.4.0_abc1234.pt",
                "sha256": digest,
            }
        )
    )
    # Match by sha256 even from a copy in another folder.
    copy = _tiny_checkpoint(tmp_path / "copy.pt", b"tag-w")
    assert match_checkpoint_tag(copy, checkpoints_dir=runs / "checkpoints") == "v0.4.0"
    info = describe_checkpoint(copy, checkpoints_dir=runs / "checkpoints")
    assert info is not None
    assert info["tag"] == "v0.4.0"
    assert info["sha256"] == digest


def test_session_marks_failed_on_exception(tmp_path: Path) -> None:
    run_dir = tmp_path / "boom"
    with pytest.raises(RuntimeError):
        with ManifestSession(run_dir, kind="selfplay", config={"x": 1}):
            raise RuntimeError("training crashed")
    data = load_manifest(run_dir)
    assert data is not None
    assert data["status"] == "failed"
    assert data["ended_at"]


def test_session_complete_on_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "ok"
    final = _tiny_checkpoint(run_dir / "latest.pt")
    with ManifestSession(run_dir, kind="sandbox", config={"n": 1}) as session:
        session.complete(final_checkpoint=final, updates_completed=3)
    data = load_manifest(run_dir)
    assert data is not None
    assert data["status"] == "completed"
    assert data["updates_completed"] == 3


def test_command_line_and_config_redact_secrets() -> None:
    line = command_line(
        [
            "train.py",
            "--api-key",
            "super-secret",
            "--openrouter-api-key=abcd",
            "OPENROUTER_API_KEY=sk-live-123",
        ]
    )
    assert "super-secret" not in line
    assert "abcd" not in line
    assert "sk-live-123" not in line
    assert "********" in line
    dumped = to_jsonable(
        {"lr": 1e-4, "openrouter_api_key": "sk-live-123", "nested": {"token": "xyz"}}
    )
    assert dumped["openrouter_api_key"] == "********"
    assert dumped["nested"]["token"] == "********"
    assert dumped["lr"] == 1e-4


def test_extract_rewards_and_infer_kind() -> None:
    assert infer_kind("selfplay_v0.4.1_winloss") == "selfplay"
    assert infer_kind("sandbox_buydiv_v0.4.1") == "buydiv"
    assert infer_kind("sandboxes_v006") == "sandbox"
    assert infer_kind("bc_tactical_v040") == "bc"
    assert infer_kind("skill_pipeline") == "skill_pipeline"
    assert infer_kind("curriculum_smoke") == "curriculum"
    assert infer_kind("random_notes") is None
    rewards = extract_rewards(
        {
            "reward": {"draw": -0.15},
            "coverage_coef": 0.25,
            "anchor_coef": 0.15,
            "updates": 300,
        }
    )
    assert rewards is not None
    assert rewards["reward"]["draw"] == -0.15
    assert rewards["coverage_coef"] == 0.25
    assert "updates" not in rewards


def test_compact_eval_drops_bulky_keys() -> None:
    out = compact_eval({"score": 1, "games": [1, 2, 3], "curve": [0.1], "ok": True})
    assert out == {"score": 1, "ok": True}


def test_backfill_from_artifacts_leaves_unknown_null(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "selfplay_demo"
    init = _tiny_checkpoint(tmp_path / "runs" / "checkpoints" / "v0.4.0_abc1234.pt", b"p")
    _write(
        tmp_path / "runs" / "checkpoints" / "v0.4.0_abc1234.json",
        json.dumps(
            {
                "version": "v0.4.0",
                "weights": "runs/checkpoints/v0.4.0_abc1234.pt",
                "sha256": sha256_file(init),
            }
        ),
    )
    cfg = {
        "run_dir": str(run_dir),
        "init_checkpoint": str(init),
        "updates": 200,
        "reward": {"draw": -0.15},
    }
    _write(run_dir / "config.yaml", yaml.safe_dump(cfg))
    _write(run_dir / "state.json", json.dumps({"update": 150, "episode": 9}))
    _write(
        run_dir / "history.jsonl",
        json.dumps({"update": 149}) + "\n" + json.dumps({"update": 150}) + "\n",
    )
    latest = _tiny_checkpoint(run_dir / "latest.pt", b"final")
    result = backfill_run(
        run_dir,
        runs_root=tmp_path / "runs",
        checkpoints_dir=tmp_path / "runs" / "checkpoints",
    )
    assert result["status"] == "written"
    data = load_manifest(run_dir)
    assert data is not None
    assert data["backfilled"] is True
    assert data["run_name"] == "selfplay_demo"
    assert data["kind"] == "selfplay"
    assert data["config"]["updates"] == 200
    assert data["rewards"]["reward"]["draw"] == -0.15
    assert data["parent"]["sha256"] == sha256_file(init)
    assert data["parent"]["tag"] == "v0.4.0"
    assert data["updates_completed"] == 150
    assert data["final_checkpoint"]["sha256"] == sha256_file(latest)
    # No evidenced wall-clock or command — must stay null, not invented.
    assert data["started_at"] is None
    assert data["ended_at"] is None
    assert data["command"] is None
    assert data["git"]["commit"] is None
    assert "started_at" in result["missing"]
    assert "command" in result["missing"]
    # Second pass does not overwrite.
    again = backfill_run(run_dir)
    assert again["status"] == "skipped"
    assert again["reason"] == "exists"


def test_backfill_skips_live_pid_and_does_not_touch_other_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "selfplay_live"
    config = _write(run_dir / "config.yaml", "updates: 3\n")
    history = _write(run_dir / "history.jsonl", json.dumps({"update": 1}) + "\n")
    before_config = config.read_bytes()
    before_history = history.read_bytes()
    proc = subprocess.Popen(["sleep", "30"])
    try:
        _write(run_dir / "train.pid", f"{proc.pid}\n")
        assert is_run_in_progress(run_dir)
        result = backfill_run(run_dir)
        assert result["status"] == "skipped"
        assert result["reason"] == "in_progress"
        assert not (run_dir / MANIFEST_NAME).exists()
        assert config.read_bytes() == before_config
        assert history.read_bytes() == before_history
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    # Stale pid (the one we just reaped) is not in progress.
    assert not pid_is_alive(proc.pid)
    assert not is_run_in_progress(run_dir)
    written = backfill_run(run_dir)
    assert written["status"] == "written"
    assert (run_dir / MANIFEST_NAME).is_file()
    assert config.read_bytes() == before_config
    assert history.read_bytes() == before_history


def test_backfill_does_not_overwrite_live_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "live"
    start_run(run_dir, kind="selfplay", config={"updates": 1}, argv=["train"])
    result = backfill_run(run_dir, overwrite=True)
    assert result["status"] == "skipped"
    assert result["reason"] == "live_manifest"
    data = load_manifest(run_dir)
    assert data is not None
    assert data["backfilled"] is False


def test_reconstruct_uses_sidecar_trained_range_and_commit(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "selfplay_mixed_v0.2.2"
    run_dir.mkdir(parents=True)
    _write(run_dir / "state.json", json.dumps({"update": 1000}))
    _tiny_checkpoint(run_dir / "update_800.pt", b"u800")
    _tiny_checkpoint(run_dir / "latest.pt", b"latest")
    (runs / "checkpoints").mkdir()
    (runs / "checkpoints" / "v0.3.0_abc.json").write_text(
        json.dumps(
            {
                "version": "v0.3.0",
                "source_run": "runs/selfplay_mixed_v0.2.2",
                "source_file": "runs/selfplay_mixed_v0.2.2/update_800.pt",
                "warm_start": "runs/checkpoints/v0.2.0_955f221.pt (v0.2.0)",
                "commit": "26348ee",
                "trained": "2026-11-23 06:21:56-06:45:32 CST",
                "update": 800,
            }
        )
    )
    data, missing = reconstruct_from_artifacts(
        run_dir, runs_root=runs, checkpoints_dir=runs / "checkpoints"
    )
    assert data["started_at"] == "2026-11-23 06:21:56 CST"
    assert data["ended_at"] == "2026-11-23 06:45:32 CST"
    assert data["git"]["commit"] == "26348ee"
    assert data["parent"]["path"] == "runs/checkpoints/v0.2.0_955f221.pt"
    # The run's last checkpoint is latest.pt, not the tagged snapshot.
    assert data["final_checkpoint"]["path"].endswith("latest.pt")
    assert data["updates_completed"] == 1000
    assert "started_at" not in missing
    assert "command" in missing
