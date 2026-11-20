"""Curriculum config, a one-update handoff, resume, and the snapshot ring."""

from pathlib import Path

import yaml

from battleboats.training.checkpoints import save_checkpoint
from battleboats.training.curriculum import (
    ArchSpec,
    CurriculumConfig,
    PPOHyper,
    StageSpec,
    load_curriculum_config,
    run_curriculum,
)
from battleboats.training.policy import PolicyNetwork
from battleboats.training.selfplay import RecentSnapshots, load_selfplay_config


def _tiny_stage(skill: str, width: int, height: int, **kwargs) -> dict:
    base = {
        "skill": skill,
        "width": width,
        "height": height,
        "success_threshold": 0.0,
        "max_updates": 1,
        "metric": "win_rate" if skill == "full" else "success_rate",
        "eval_every": 1,
        "eval_episodes": 1,
        "max_turns": 6,
        "max_steps": 16,
        "rollout_decisions": 4,
        "train_opponent": "random" if skill == "full" else "none",
    }
    base.update(kwargs)
    return base


def test_load_real_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    curriculum = load_curriculum_config(root / "configs" / "curriculum.yaml")
    assert [s.skill for s in curriculum.stages[:4]] == [
        "merchant",
        "merchant",
        "merchant",
        "landing",
    ]
    assert curriculum.stages[-1].skill == "full"
    assert curriculum.stages[-1].train_opponent == "snapshot"
    selfplay = load_selfplay_config(root / "configs" / "selfplay.yaml")
    assert selfplay.snapshot_pool == 4
    assert selfplay.random_opponent_frac == 0.15


def test_unknown_skill_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"stages": [_tiny_stage("league", 2, 3)]}))
    try:
        load_curriculum_config(path)
    except ValueError as exc:
        assert "league" in str(exc)
    else:
        raise AssertionError("unknown skill should be rejected")


def test_tiny_curriculum_handoff_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "cur.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run_dir": str(tmp_path / "run"),
                "device": "cpu",
                "allow_unmet": True,
                "arch": {"d_model": 32, "nhead": 2, "num_layers": 1, "dim_feedforward": 64, "dropout": 0.0},
                "ppo": {"epochs": 1, "minibatch_size": 8},
                "stages": [
                    _tiny_stage("merchant", 2, 3),
                    _tiny_stage("full", 4, 4),
                ],
            }
        )
    )
    cfg = load_curriculum_config(path)
    first = run_curriculum(cfg, resume=False)
    assert first["ok"]
    assert (tmp_path / "run" / "latest.pt").is_file()
    history = (tmp_path / "run" / "history.jsonl").read_text().strip().splitlines()
    assert len(history) >= 2
    second = run_curriculum(cfg, resume=True)
    assert second["ok"]
    again = (tmp_path / "run" / "history.jsonl").read_text().strip().splitlines()
    assert again == history


def test_missed_gate_stops(tmp_path: Path) -> None:
    cfg = CurriculumConfig(
        run_dir=tmp_path / "stop",
        device="cpu",
        allow_unmet=False,
        arch=ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0),
        ppo=PPOHyper(epochs=1, minibatch_size=8),
        stages=[
            StageSpec(
                skill="merchant",
                width=2,
                height=3,
                success_threshold=1.1,
                max_updates=1,
                eval_every=1,
                eval_episodes=1,
                max_turns=4,
                max_steps=8,
                rollout_decisions=4,
            )
        ],
    )
    result = run_curriculum(cfg, resume=False)
    assert result["ok"] is False
    assert "merchant_2x3" in result["reason"]


def test_snapshot_ring_drops_oldest(tmp_path: Path) -> None:
    ring = RecentSnapshots(tmp_path / "snaps", pool_size=2, device="cpu")
    for i in range(3):
        policy = PolicyNetwork(
            token_dim=28,
            d_model=32,
            nhead=2,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
        )
        ring.add(policy, tag=f"update_{i:05d}")
    assert ring.names() == ["update_00001.pt", "update_00002.pt"]
    assert (tmp_path / "snaps" / "update_00000.pt").is_file()
    fresh = RecentSnapshots(tmp_path / "snaps", pool_size=2, device="cpu")
    assert fresh.load_from_disk() == 2
    assert fresh.names() == ["update_00001.pt", "update_00002.pt"]
    sampled = fresh.sample(__import__("random").Random(0))
    assert sampled is not None
    save_checkpoint(sampled, tmp_path / "roundtrip.pt")
