"""Memorization probe: does a trained landing policy generalize to a moved target?

The learner never sees the enemy home port until its ship is within visibility
range (fog of war — enemy ports come from port_sightings, not engine.ports). So a
policy trained on a fixed map can succeed purely by memorizing a route to the
home port's fixed location. This probe loads a checkpoint and evaluates it (no
training) on maps with the enemy home moved to various tiles. If capture-rate is
high at the training location but collapses elsewhere, the policy memorized.

Usage:
    poetry run python3 scripts/landing_generalization_probe.py \
        --checkpoint runs/landing_checkpoints_5x5/landing_latest.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from battleboats.training.policy import PolicyNetwork
from battleboats.training.rollout import collect_trajectory

LEARNER_PID = 0
P0_HOME = (0, 0)  # fixed, so the ship always spawns at the same tile (~(1,0))


def write_map(path: Path, p1_home, width=5, height=5) -> None:
    data = {
        "width": width,
        "height": height,
        "seed": 0,
        "land": [list(P0_HOME), list(p1_home)],
        "ports": {"0": [list(P0_HOME)], "1": [list(p1_home)]},
    }
    path.write_text(json.dumps(data))


def load_policy(path: Path, device) -> PolicyNetwork:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = PolicyNetwork(**ckpt["model_config"])
    net.load_state_dict(ckpt["model_state"])
    return net.to(device).eval()


def eval_target(policy, tmpdir: Path, p1_home, episodes, rng, max_turns, max_steps) -> float:
    from battleboats.envs.landing_skill_aec import LandingSkillAEC

    map_path = tmpdir / f"map_{p1_home[0]}_{p1_home[1]}.json"
    write_map(map_path, p1_home)
    env = LandingSkillAEC(str(map_path), max_turns=max_turns)
    caps = 0
    for ep in range(episodes):
        env.reset(seed=ep)
        _, info = collect_trajectory(policy, None, env, LEARNER_PID, rng, max_steps=max_steps)
        caps += int(info["winner"] == LEARNER_PID)
    return caps / episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tmpdir", type=Path, default=Path("runs/_probe_maps"))
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = random.Random(0)
    torch.manual_seed(0)
    policy = load_policy(args.checkpoint, device)
    args.tmpdir.mkdir(parents=True, exist_ok=True)

    # (4,4) is the training target (control). The rest move it around the board.
    targets = [(4, 4), (4, 0), (0, 4), (2, 4), (4, 2), (2, 2), (3, 3), (1, 3)]
    print(f"checkpoint={args.checkpoint}  episodes={args.episodes}")
    print(f"{'target':>8}  {'capture_rate':>12}")
    for t in targets:
        rate = eval_target(policy, args.tmpdir, t, args.episodes, rng, args.max_turns, args.max_steps)
        tag = "  <- TRAIN" if t == (4, 4) else ""
        print(f"{str(t):>8}  {rate:>12.3f}{tag}")


if __name__ == "__main__":
    main()
