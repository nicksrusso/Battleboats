"""Conditional attack rate from a token harvest: P(attack chosen | attack legal).

Tests the "movement is bad, not combat aversion" theory. For every MCTS decision
in a harvest, reconstruct the engine from the logged tokens, call enumerate_legal
to see whether an attack was AVAILABLE, and compare to whether the expert actually
attacked. The discriminating readout:

  - high conditional rate + low availability  => ships fight WHEN they can, but
                                                  rarely get the chance (movement).
  - normal availability + low conditional rate => attacks offered but declined
                                                  (combat undervalued).

CAVEAT: reconstruction uses the ACTING player's fogged view, so "available" means
"an attack on a SIGHTED enemy was legal." God-mode MCTS searched full information,
so this can slightly UNDER-count availability vs the search's true legal set. For
the exact legal set, harvest with --debug-mcts (mcts_root lists the real root
children) and read it directly. This script needs no such flag — only tokens.

    poetry run python scripts/conditional_attack_rate.py runs/harvests/<run>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _resolve(path_str: str) -> str:
    p = Path(path_str)
    return str(p if p.exists() else REPO_ROOT / path_str)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("harvest_dir", type=Path, help="A harvest dir of per-game shards (game_*.jsonl).")
    ap.add_argument("--scenarios-file", type=Path, default=None, help="Override scenarios file (else from run config).")
    args = ap.parse_args()

    from move_scores import engine_from_tokens
    from battleboats.core.actions import AttackAction
    from battleboats.training.policy import VERB_TO_IDX

    ATTACK_VERB = VERB_TO_IDX["attack"]

    cfg_path = args.harvest_dir / "_run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    scenarios_file = args.scenarios_file or Path(cfg.get("scenarios_file", "runs/scenarios/scenarios_1000.json"))
    scenarios = json.loads(Path(_resolve(str(scenarios_file))).read_text())

    n_dec = n_avail = n_chosen = n_mismatch = 0
    shards = sorted(args.harvest_dir.glob("game_*.jsonl"))
    if not shards:
        sys.exit(f"No game_*.jsonl shards in {args.harvest_dir}")

    for shard in shards:
        # Group rows by step -> {perspective: row}; the acting row has action != None.
        by_step: dict = {}
        game_idx = None
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("_type") == "game_footer":
                continue
            game_idx = r["game_idx"]
            by_step.setdefault(r["step"], {})[r["perspective"]] = r
        if game_idx is None:
            continue
        scenario = scenarios[game_idx % len(scenarios)]
        map_path = _resolve(scenario["map_path"])

        for step, persp in by_step.items():
            acting = next((p for p, row in persp.items() if row.get("action") is not None), None)
            if acting is None or 0 not in persp or 1 not in persp:
                continue
            row = persp[acting]
            n_dec += 1
            chose_attack = row["action"][1] == ATTACK_VERB

            eng = engine_from_tokens(
                map_path, scenario.get("seed"),
                persp[0]["tokens"], persp[1]["tokens"], row["cash"], row["turn"], acting,
            )
            attack_legal = any(isinstance(m, AttackAction) for m in eng.enumerate_legal(player_id=acting))

            if attack_legal:
                n_avail += 1
            if chose_attack:
                n_chosen += 1
                if not attack_legal:
                    n_mismatch += 1  # chosen attack not in reconstructed legal set (fog/reconstruction gap)

    avail_rate = n_avail / n_dec if n_dec else 0.0
    uncond = n_chosen / n_dec if n_dec else 0.0
    # Denominator guards against the fog mismatch: count any decision where attack
    # was available OR chosen, so a reconstruction gap can't push the rate > 1.
    denom = n_avail + n_mismatch
    cond = n_chosen / denom if denom else 0.0

    print(f"harvest:            {args.harvest_dir}")
    print(f"  config:           iters={cfg.get('iterations')} games={cfg.get('num_games')} "
          f"max_turns={cfg.get('max_turns')} scn={cfg.get('scenarios_file')}")
    print(f"  MCTS decisions:   {n_dec}")
    print(f"  attack available: {n_avail}  ({avail_rate:.1%} of decisions)")
    print(f"  attack chosen:    {n_chosen}  ({uncond:.2%} of decisions, unconditional)")
    print(f"  --> CONDITIONAL attack rate  P(attack | attack legal) = {cond:.1%}  ({n_chosen}/{denom})")
    if n_mismatch:
        print(f"  note: {n_mismatch} chosen-attack steps had no legal attack in the reconstructed "
              f"(fogged) engine — counted in the denominator. See script CAVEAT.")


if __name__ == "__main__":
    main()
