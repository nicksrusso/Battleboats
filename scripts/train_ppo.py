"""PPO self-play training orchestration.

Wires ``collect_trajectory`` + ``compute_gae`` (per-game rollouts), a
``RolloutBuffer`` (pooling decisions across games), and ``ppo_update`` (the
K-epoch clipped update) into the training loop. Each update clears the buffer,
collects games until it holds ``rollout_decisions`` decisions (alternating which
seat the learner plays), runs the update, refreshes the frozen opponent every N
updates, and logs / checkpoints.

The opponent is a frozen snapshot of a recent policy (initially the BC checkpoint,
or a random-init copy), refreshed periodically for a slowly-improving curriculum
rather than the instability of training against a live opponent.

Example:
    poetry run python scripts/train_ppo.py \\
        --scenarios runs/scenarios/scenarios_500.json --updates 50 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.training.policy import PolicyNetwork
from battleboats.training.ppo_buffer import RolloutBuffer
from battleboats.training.ppo_update import ppo_update
from battleboats.training.rollout import collect_trajectory, compute_gae

DEFAULT_TOKEN_DIM = 28  # 64x32 harvest w/ cash-on-home-port token (see observation.py)


def load_policy(
    bc_checkpoint: Optional[Path],
    device: torch.device,
    *,
    token_dim: int = DEFAULT_TOKEN_DIM,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
) -> PolicyNetwork:
    """Build the learning policy, warm-starting from a BC checkpoint if given.

    A BC checkpoint is a self-describing ``{"model_state", "model_config"}`` dict,
    so warm-start is a clean round-trip: rebuild from ``model_config`` and load the
    weights. With no checkpoint, the policy is randomly initialized at the
    architecture given by the keyword arguments. The checkpoint's ``model_config``
    is authoritative over the arch flags (the state dict only fits that arch).
    """
    if bc_checkpoint is not None:
        ckpt = torch.load(bc_checkpoint, map_location=device, weights_only=False)
        net = PolicyNetwork(**ckpt["model_config"])
        net.load_state_dict(ckpt["model_state"])
        if ckpt["model_config"].get("d_model", d_model) != d_model:
            print(
                f"[load_policy] note: --d-model={d_model} ignored; checkpoint "
                f"d_model={ckpt['model_config']['d_model']} is authoritative."
            )
    else:
        net = PolicyNetwork(
            token_dim=token_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    return net.to(device)


def make_frozen_opponent(policy: PolicyNetwork) -> PolicyNetwork:
    """Snapshot ``policy`` into a frozen opponent (deep copy, eval, no grad).

    Also used to refresh the opponent every N updates by re-snapshotting the
    current policy. The deep copy keeps later training of ``policy`` from mutating
    the frozen copy through shared tensors.
    """
    opp = copy.deepcopy(policy)
    opp.eval()
    for param in opp.parameters():
        param.requires_grad_(False)
    return opp


def collect_rollouts(
    policy: PolicyNetwork,
    opponent: PolicyNetwork,
    env: BattleboatsAEC,
    scenarios: List[dict],
    buffer: RolloutBuffer,
    rng: random.Random,
    *,
    target_decisions: int,
    gamma: float,
    lam: float,
    max_steps: int,
    game_counter: int,
) -> Tuple[int, Dict[str, float]]:
    """Collect games into the buffer until it holds ``target_decisions``.

    Cycles through ``scenarios``, alternating the learner's seat by game parity so
    it learns both sides of the first-player advantage, and runs ``compute_gae`` on
    each trajectory before adding it (GAE is per-game and must not cross game
    boundaries). Collecting by decisions rather than games keeps the update batch
    size stable across wildly varying game lengths.

    Returns:
        ``(game_counter, stats)`` where ``stats`` reports games, decisions, and
        win / loss / draw rates and mean game length for this rollout.
    """
    n_games = 0
    outcomes: List[float] = []
    lengths: List[int] = []
    while len(buffer) < target_decisions:
        scenario = scenarios[game_counter % len(scenarios)]
        env.reset(options={"scenario": scenario})

        # Alternate the learner's seat each game so it learns both sides (p0 has a
        # first-move advantage). Parity on the global game counter.
        agent_pid = game_counter % 2

        traj, info = collect_trajectory(
            policy, opponent, env, agent_pid, rng, max_steps=max_steps
        )
        compute_gae(traj, gamma=gamma, lam=lam)  # per-game, in place, before add
        buffer.add(traj)

        outcomes.append(info["outcome"])
        lengths.append(info["steps"])
        n_games += 1
        game_counter += 1

    stats = {
        "rollout/games": float(n_games),
        "rollout/decisions": float(len(buffer)),
        "rollout/win_rate": sum(o > 0 for o in outcomes) / n_games,
        "rollout/loss_rate": sum(o < 0 for o in outcomes) / n_games,
        "rollout/draw_rate": sum(o == 0 for o in outcomes) / n_games,
        "rollout/mean_len": sum(lengths) / n_games,
    }
    return game_counter, stats


def main() -> None:
    p = argparse.ArgumentParser(description="PPO self-play training for Battleboats.")
    # Data / env
    p.add_argument("--scenarios", type=Path, default=Path("runs/scenarios/scenarios_500.json"),
                   help="JSON list of scenarios to cycle through for game starts.")
    p.add_argument("--bc-checkpoint", type=Path,
                   default=Path("runs/checkpoints/bc_policy_latest.pt"),
                   help="Warm-start policy + initial opponent from this BC checkpoint. "
                        "Pass --bc-checkpoint '' for random init.")
    p.add_argument("--max-steps", type=int, default=400, help="Engine-step budget per game.")
    # PPO loop
    p.add_argument("--updates", type=int, default=50, help="Number of PPO updates (outer loop).")
    p.add_argument("--rollout-decisions", type=int, default=2048,
                   help="Collect at least this many decisions per update before training.")
    p.add_argument("--opponent-refresh", type=int, default=5,
                   help="Refresh the frozen opponent every N updates.")
    # PPO hyperparameters
    p.add_argument("--epochs", type=int, default=4, help="K epochs per update.")
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None, help="Optional early-stop KL threshold.")
    # Arch (only used on random init; checkpoint config wins when warm-starting)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-feedforward", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # Infra
    p.add_argument("--device", default="cpu", help="cpu or cuda.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", type=Path, default=Path("runs/ppo_checkpoints"))
    p.add_argument("--save-every", type=int, default=10, help="Checkpoint every N updates.")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    # ----------------------------------------------------------------- setup
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    scenarios = json.loads(args.scenarios.read_text())
    if not scenarios:
        raise SystemExit(f"No scenarios loaded from {args.scenarios}")

    # --bc-checkpoint '' (empty) or a missing file => random init.
    bc_ckpt = args.bc_checkpoint
    if bc_ckpt is not None and (str(bc_ckpt) == "" or not bc_ckpt.exists()):
        if str(bc_ckpt) != "":
            print(f"[train_ppo] checkpoint {bc_ckpt} not found — random init.")
        bc_ckpt = None

    policy = load_policy(
        bc_ckpt, device,
        token_dim=DEFAULT_TOKEN_DIM, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    opponent = make_frozen_opponent(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    buffer = RolloutBuffer()
    env = BattleboatsAEC(map_json_path=scenarios[0]["map_path"], max_turns=args.max_steps)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        wandb.init(project="battleboats-ppo", config=vars(args))

    print(
        f"[train_ppo] init={'BC ' + str(bc_ckpt) if bc_ckpt else 'random'}  "
        f"device={device}  scenarios={len(scenarios)}  updates={args.updates}"
    )

    # ----------------------------------------------------------------- main loop
    game_counter = 0
    for update in range(args.updates):
        buffer.clear()  # PPO is on-policy: discard last update's stale data

        policy.eval()  # rollout samples actions; no dropout
        game_counter, roll_stats = collect_rollouts(
            policy, opponent, env, scenarios, buffer, rng,
            target_decisions=args.rollout_decisions, gamma=args.gamma, lam=args.lam,
            max_steps=args.max_steps, game_counter=game_counter,
        )

        policy.train()  # update is a normal training forward (dropout on)
        metrics = ppo_update(
            policy, optimizer, buffer,
            clip_eps=args.clip_eps, vf_coef=args.vf_coef, ent_coef=args.ent_coef,
            epochs=args.epochs, minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm, device=device, target_kl=args.target_kl,
        )

        if (update + 1) % args.opponent_refresh == 0:
            opponent = make_frozen_opponent(policy)

        log = {**roll_stats, **{f"ppo/{k}": v for k, v in metrics.items()}, "update": update}
        print(
            f"[update {update + 1}/{args.updates}] "
            f"win={roll_stats['rollout/win_rate']:.2f} "
            f"draw={roll_stats['rollout/draw_rate']:.2f} "
            f"len={roll_stats['rollout/mean_len']:.0f} | "
            f"pi_loss={metrics['policy_loss']:.4f} v_loss={metrics['value_loss']:.4f} "
            f"ent={metrics['entropy']:.3f} kl={metrics['approx_kl']:.4f} "
            f"clip={metrics['clip_frac']:.2f}"
        )
        if use_wandb:
            wandb.log(log)

        if (update + 1) % args.save_every == 0:
            torch.save(
                {
                    "model_state": policy.state_dict(),
                    "model_config": {
                        "token_dim": policy.proj.in_features,
                        "d_model": policy.d_model,
                        "nhead": args.nhead,
                        "num_layers": args.num_layers,
                        "dim_feedforward": args.dim_feedforward,
                        "dropout": args.dropout,
                    },
                },
                args.save_dir / f"ppo_update_{update + 1}.pt",
            )

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
