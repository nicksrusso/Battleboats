"""Skill-pipeline config, a missed gate, resume, and optional wandb."""

import sys
from pathlib import Path

import yaml

from battleboats.training.curriculum import ArchSpec
from battleboats.training.loop import PPOHyper
from battleboats.training.skill_pipeline import (
    PipelineConfig,
    StageSpec,
    TaskSpec,
    final_checkpoint_path,
    load_pipeline_config,
    run_pipeline,
)
from battleboats.training.wandb_log import WandbLog, init_wandb, wandb_requested


def _tiny_task(skill: str, **kwargs) -> TaskSpec:
    base = dict(
        skill=skill,
        width=2,
        height=3,
        max_turns=4,
        max_steps=12,
        randomize=False,
        approach_coef=0.0,
        success_threshold=0.0,
    )
    base.update(kwargs)
    return TaskSpec(**base)


def _tiny_arch() -> ArchSpec:
    return ArchSpec(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, dropout=0.0)


def test_load_real_and_smoke_configs() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_pipeline_config(root / "configs" / "skill_pipeline.yaml")
    assert [t.skill for t in cfg.stages[0].tasks] == [
        "move",
        "merchant",
        "shipyard",
        "combat",
        "landing",
        "builder",
    ]
    assert cfg.stages[0].kind == "interleave"
    assert cfg.stages[-1].kind == "full_mix"
    assert cfg.stages[-1].scripted_threshold > 0
    assert cfg.wandb is False
    assert final_checkpoint_path(cfg) == Path("runs/skill_pipeline/final.pt")
    smoke = load_pipeline_config(root / "configs" / "skill_pipeline_smoke.yaml")
    assert smoke.allow_unmet is True
    assert {t.skill for t in smoke.stages[0].tasks} == {
        "move",
        "merchant",
        "shipyard",
        "combat",
        "landing",
        "builder",
        "opening",
    }
    assert any(s.name == "opening_6x6" for s in cfg.stages)
    assert smoke.stages[1].kind == "full_mix"


def test_unknown_sandbox_rejected() -> None:
    try:
        TaskSpec(skill="league", width=2, height=3)
    except ValueError as exc:
        assert "league" in str(exc)
    else:
        raise AssertionError("unknown sandbox should be rejected")


def test_tiny_pipeline_writes_final_and_resumes(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        run_dir=tmp_path / "run",
        device="cpu",
        allow_unmet=True,
        arch=_tiny_arch(),
        ppo=PPOHyper(epochs=1, minibatch_size=4, ent_coef=0.01),
        stages=[
            StageSpec(
                name="skills",
                kind="interleave",
                max_updates=1,
                eval_every=1,
                eval_episodes=1,
                decisions_per_update=4,
                tasks=[_tiny_task("move"), _tiny_task("shipyard")],
            ),
            StageSpec(
                name="full",
                kind="full_mix",
                max_updates=1,
                eval_every=1,
                eval_episodes=1,
                decisions_per_update=4,
                width=4,
                height=4,
                max_turns=4,
                max_steps=12,
                map_count=1,
                skill_frac=0.5,
                scripted_frac=0.5,
                win_threshold=0.0,
                scripted_threshold=0.0,
                tasks=[_tiny_task("combat")],
            ),
        ],
    )
    first = run_pipeline(cfg, resume=False)
    assert first["ok"]
    assert final_checkpoint_path(cfg).is_file()
    assert (tmp_path / "run" / "latest.pt").is_file()
    history = (tmp_path / "run" / "history.jsonl").read_text().strip().splitlines()
    assert len(history) >= 2
    second = run_pipeline(cfg, resume=True)
    assert second["ok"]
    again = (tmp_path / "run" / "history.jsonl").read_text().strip().splitlines()
    assert again == history


def test_missed_gate_stops(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        run_dir=tmp_path / "stop",
        device="cpu",
        allow_unmet=False,
        arch=_tiny_arch(),
        ppo=PPOHyper(epochs=1, minibatch_size=4),
        stages=[
            StageSpec(
                name="skills",
                kind="interleave",
                max_updates=1,
                eval_every=1,
                eval_episodes=1,
                decisions_per_update=4,
                tasks=[_tiny_task("move", success_threshold=1.1)],
            )
        ],
    )
    result = run_pipeline(cfg, resume=False)
    assert result["ok"] is False
    assert "skills" in result["reason"]
    assert final_checkpoint_path(cfg).is_file()


def test_wandb_stays_off_and_survives_a_missing_package(monkeypatch) -> None:
    monkeypatch.delenv("BATTLEBOATS_WANDB", raising=False)
    assert wandb_requested(False) is False
    assert init_wandb(False, project="battleboats-skill-pipeline", config={}).enabled is False

    monkeypatch.setenv("BATTLEBOATS_WANDB", "1")
    assert wandb_requested(False) is True
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = init_wandb(False, project="battleboats-skill-pipeline", config={})
    assert logger.enabled is False


def test_wandb_skips_when_logged_out_and_logs_when_configured(monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class LoggedOut:
        api = type("Api", (), {"api_key": None})()

        @staticmethod
        def init(**kwargs):
            raise AssertionError("init should not run without a key")

    monkeypatch.setitem(sys.modules, "wandb", LoggedOut)
    quiet = init_wandb(True, project="battleboats-skill-pipeline", config={})
    assert quiet.enabled is False

    class Ready:
        api = type("Api", (), {"api_key": "secret"})()
        rows = []

        @staticmethod
        def init(**kwargs):
            Ready.rows.append(("init", kwargs["project"]))

        @staticmethod
        def log(data, step=None):
            Ready.rows.append(("log", step))

        @staticmethod
        def finish():
            Ready.rows.append(("finish", None))

    monkeypatch.setitem(sys.modules, "wandb", Ready)
    logger = init_wandb(True, project="battleboats-skill-pipeline", config={"run_dir": "runs"})
    assert isinstance(logger, WandbLog)
    assert logger.enabled
    logger.log({"eval/move": 1.0}, step=1)
    logger.finish()
    assert Ready.rows[0] == ("init", "battleboats-skill-pipeline")
    assert ("log", 1) in Ready.rows
    assert Ready.rows[-1][0] == "finish"


def test_smoke_yaml_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "configs" / "skill_pipeline_smoke.yaml").read_text())
    assert raw["wandb"] is False
    assert raw["stages"][1]["scripted_frac"] == 0.5
