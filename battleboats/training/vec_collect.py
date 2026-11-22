"""Batched, multi-process game collection for full-game self-play.

``GameCollector`` keeps ``games`` games open at once. Each round it advances
every game to its next policy decision, groups the games by which network
acts (the learner or one opponent snapshot), and runs one ``act_batch``
forward per group. Uniform-random opponent turns are played inline.

The buffer is capped exactly. ``collect(budget)`` stops once it has recorded
``budget`` learner decisions. Games still running are cut there: each open
game is advanced to the learner's next decision, the current policy's V(s)
at that state bootstraps the segment's GAE, and the game carries on in the
next update with the new weights. Finished games get the sparse terminal
reward (+1/-1/0) on their last learner decision and bootstrap with 0; hitting
the per-map step cap still counts as a draw, as before.

With ``shaping`` enabled (``reward_shaping``), each learner decision also gets
the shaped reward its own action earned (``ShapingTracker.score``), added to
that decision's transition. The terminal outcome is added on top of the last
transition's shaping, never replacing it. The kill component's losses happen
on the opponent's turn; they are settled when the learner next decides (or
the game ends, or the segment is cut) and land on the learner's previous
transition. ``landing_progress`` (a potential difference between consecutive
learner decisions) is settled the same way.

``fleet_starts`` gives a fraction of games a random, equal-cost fleet for both
sides at the start (``fleet_starts.py``). Every step (both seats) is checked
for attacks, so ``CollectResult.combat`` counts attacks, sinkings and sunk
hulls for the learner and its opponent.

``ParallelCollector`` runs one ``GameCollector`` per worker process (spawn,
``collect_threads`` torch threads each) and sends the learner and snapshot
weights to every worker before each update. Episode ``k`` of worker ``w``
gets index ``episode_offset + w + num_workers * k``; the index picks the map,
the learner seat (index % 2), the engine seed, and the opponent draw. Games
open when a run stops are dropped on resume.
"""

from __future__ import annotations

import multiprocessing as mp
import random
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from battleboats.agents.random_agent import random_action
from battleboats.core.actions import AttackAction, BuildShipAction, CapturePortAction, EndTurnAction
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.battleboats_aec import BattleboatsAEC
from battleboats.training.policy import PolicyNetwork
from battleboats.training.fleet_starts import FleetStarts, apply_fleet_start
from battleboats.training.reward_shaping import RewardShaping, ShapingTracker, empty_totals
from battleboats.training.selfplay_purchases import hull_name
from battleboats.training.rollout import Transition, compute_gae
from battleboats.training.selfplay_purchases import empty_purchase_counts, record_purchase

LEARNER = "__learner__"
_TENSOR_FIELDS = ("tokens", "asset_mask", "verb_mask", "target_mask", "globals")


def empty_combat() -> Dict[str, Any]:
    return {
        "learner_attacks": 0,
        "learner_sinks": 0,
        "opp_attacks": 0,
        "opp_sinks": 0,
        "sunk_by_learner": {},
        "lost_by_learner": {},
    }


def empty_landing() -> Dict[str, Any]:
    """Learner landing-ship stats. ``alive_sum``/``samples`` average the
    learner's landing ships alive over its decisions; ``min_dist`` is the
    smallest water distance (1 = adjacent) of any learner landing ship to the
    enemy home seen at a learner decision. Captures count both seats."""
    return {
        "alive_sum": 0,
        "samples": 0,
        "min_dist": None,
        "learner_capture_attempts": 0,
        "learner_captures": 0,
        "learner_home_captures": 0,
        "opp_capture_attempts": 0,
        "opp_captures": 0,
        "opp_home_captures": 0,
    }


def merge_landing(into: Dict[str, Any], other: Dict[str, Any]) -> None:
    for k, v in other.items():
        if k == "min_dist":
            if v is not None and (into[k] is None or v < into[k]):
                into[k] = v
        else:
            into[k] += v


def merge_combat(into: Dict[str, Any], other: Dict[str, Any]) -> None:
    for k, v in other.items():
        if isinstance(v, dict):
            for h, n in v.items():
                into[k][h] = into[k].get(h, 0) + n
        else:
            into[k] += v


@dataclass
class CollectResult:
    segments: List[List[Transition]] = field(default_factory=list)
    finished: List[Dict[str, Any]] = field(default_factory=list)
    purchases: Dict[str, int] = field(default_factory=empty_purchase_counts)
    # Learner shaped reward by component, summed over this collect.
    shaping: Dict[str, float] = field(default_factory=empty_totals)
    combat: Dict[str, Any] = field(default_factory=empty_combat)
    landing: Dict[str, Any] = field(default_factory=empty_landing)
    decisions: int = 0
    env_steps: int = 0
    open_games: int = 0


@dataclass
class _Slot:
    env: Any
    ep: int = 0
    agent_pid: int = 0
    opp_tag: Optional[str] = None
    rng: Any = None
    max_steps: int = 0
    size: str = ""
    steps: int = 0
    decisions: int = 0
    segment: List[Transition] = field(default_factory=list)
    purchases: Dict[str, int] = field(default_factory=empty_purchase_counts)
    tracker: Any = None
    fleet_start: bool = False
    combat: Dict[str, Any] = field(default_factory=empty_combat)
    min_dist: Optional[int] = None
    landing: Dict[str, Any] = field(default_factory=empty_landing)


def _batch_inputs(masks_list: Sequence[ActionMasks]):
    toks = [m.tokens[0] for m in masks_list]
    lengths = torch.tensor([t.shape[0] for t in toks])
    tokens = pad_sequence(toks, batch_first=True)
    pad = torch.arange(tokens.shape[1])[None, :] < lengths[:, None]
    glob = torch.cat([m.globals for m in masks_list], dim=0)
    return tokens, pad, glob


class GameCollector:
    def __init__(
        self,
        maps: Sequence[str],
        map_info: Sequence[Dict[str, Any]],
        *,
        model_config: Dict[str, Any],
        seed: int = 0,
        games: int = 4,
        worker_id: int = 0,
        num_workers: int = 1,
        episode_offset: int = 0,
        random_opponent_frac: float = 0.15,
        gamma: float = 0.99,
        lam: float = 0.95,
        shaping: Optional[RewardShaping] = None,
        fleet_starts: Optional[FleetStarts] = None,
    ) -> None:
        self.maps = [str(m) for m in maps]
        self.map_info = list(map_info)
        self.model_config = dict(model_config)
        self.seed = int(seed)
        self.games = max(1, int(games))
        self.worker_id = int(worker_id)
        self.num_workers = max(1, int(num_workers))
        self.episode_offset = int(episode_offset)
        self.random_opponent_frac = float(random_opponent_frac)
        self.gamma, self.lam = float(gamma), float(lam)
        self.shaping = shaping if shaping is not None else RewardShaping()
        self.fleet_starts = fleet_starts if fleet_starts is not None else FleetStarts()
        self.policy = self._new_net()
        self.opponents: Dict[str, PolicyNetwork] = {}
        self.ring: List[str] = []
        self.episodes_started = 0
        self.slots: Optional[List[_Slot]] = None

    def _new_net(self) -> PolicyNetwork:
        net = PolicyNetwork(**self.model_config)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net

    # ------------------------------------------------------------- weights
    def set_weights(self, learner_state, opponent_states: Dict[str, Any], ring: Sequence[str]) -> None:
        self.policy.load_state_dict(learner_state)
        self.ring = sorted(ring)
        for tag in self.ring:
            if tag not in self.opponents:
                net = self._new_net()
                net.load_state_dict(opponent_states[tag])
                self.opponents[tag] = net
        used = {s.opp_tag for s in self.slots or []}
        for tag in list(self.opponents):
            if tag not in self.ring and tag not in used:
                del self.opponents[tag]

    # ------------------------------------------------------------- episodes
    def _start(self, slot: _Slot) -> None:
        k = self.episodes_started
        self.episodes_started += 1
        ep = self.episode_offset + self.worker_id + self.num_workers * k
        i = random.Random(self.seed * 1_000_003 + ep).randrange(len(self.maps))
        info = self.map_info[i]
        slot.env.map_json_path = self.maps[i]
        slot.env.max_turns = info["max_turns"]
        slot.env.reset(seed=self.seed + ep)
        draw = random.Random(self.seed * 7_777_777 + ep + 1)
        if not self.ring or draw.random() < self.random_opponent_frac:
            slot.opp_tag = None
        else:
            slot.opp_tag = draw.choice(self.ring)
        slot.ep = ep
        slot.agent_pid = ep % 2
        slot.rng = random.Random(self.seed * 31 + ep)
        slot.max_steps = int(info["max_steps"])
        slot.size = str(info.get("size", ""))
        slot.steps = 0
        slot.decisions = 0
        slot.segment = []
        slot.purchases = empty_purchase_counts()
        slot.tracker = ShapingTracker(self.shaping)
        fleet = apply_fleet_start(slot.env.engine, self.fleet_starts, random.Random(self.seed * 4_000_037 + ep + 7))
        slot.fleet_start = fleet is not None
        slot.tracker.start(slot.env.engine, slot.agent_pid)
        slot.combat = empty_combat()
        slot.min_dist = None
        slot.landing = empty_landing()

    def _make_slots(self) -> None:
        self.slots = []
        for _ in range(self.games):
            env = BattleboatsAEC(self.maps[0], max_turns=self.map_info[0]["max_turns"])
            env.prime_legal_actions = False
            slot = _Slot(env=env)
            self._start(slot)
            self.slots.append(slot)

    @staticmethod
    def _done(slot: _Slot) -> bool:
        env = slot.env
        return (
            any(env.terminations.values())
            or any(env.truncations.values())
            or slot.steps > slot.max_steps
        )

    def _finish(self, slot: _Slot, result: CollectResult) -> None:
        engine = slot.env.engine
        winner = engine.winner
        outcome = 1.0 if winner == slot.agent_pid else (-1.0 if winner is not None else 0.0)
        if slot.segment:
            self._settle(slot, result)
            slot.segment[-1].reward += outcome
            compute_gae(slot.segment, gamma=self.gamma, lam=self.lam, last_value=0.0)
            result.segments.append(slot.segment)
        result.finished.append(
            {
                "winner": winner,
                "outcome": outcome,
                "steps": slot.steps,
                "turn": int(engine.turn),
                "n_decisions": slot.decisions,
                "truncated": winner is None,
                "purchases": slot.purchases,
                "agent_pid": slot.agent_pid,
                "map_size": f"{engine.map.width}x{engine.map.height}",
                "episode": slot.ep,
                "opponent": slot.opp_tag or "random",
                "shaping": dict(slot.tracker.totals),
                "fleet_start": slot.fleet_start,
                "combat": slot.combat,
                "min_fleet_dist": slot.min_dist,
                "landing": slot.landing,
            }
        )
        self._start(slot)

    def _advance(self, slot: _Slot, result: CollectResult, stop_on_finish: bool = False) -> Optional[int]:
        """Play random-opponent turns and finished-game restarts until a
        network must act; return that seat. None if ``stop_on_finish`` and the
        game ended."""
        while True:
            if self._done(slot):
                self._finish(slot, result)
                if stop_on_finish:
                    return None
                continue
            env = slot.env
            pid = env._player_id(env.agent_selection)
            if pid != slot.agent_pid and slot.opp_tag is None:
                self._step(slot, pid, random_action(env.engine, pid, slot.rng), result)
                continue
            return pid

    def _act(self, key: str, items, result: CollectResult) -> int:
        net = self.policy if key == LEARNER else self.opponents[key]
        live, masks_list, pid_of = [], [], []
        for slot, pid in items:
            masks = ActionMasks(slot.env.engine, pid)
            if masks.n == 0:
                self._step(slot, pid, EndTurnAction(), result)
                continue
            live.append(slot)
            masks_list.append(masks)
            pid_of.append(pid)
        if not live:
            return 0
        tokens, pad, glob = _batch_inputs(masks_list)
        out = net.act_batch(tokens, pad, masks_list, skill_idx=0, globals=glob)
        assets = out["asset_idx"].tolist()
        verbs = out["verb_idx"].tolist()
        targets = out["target_idx"].tolist()
        logps = out["logp"].tolist()
        values = out["value"].tolist()
        recorded = 0
        for i, (slot, masks) in enumerate(zip(live, masks_list)):
            if key == LEARNER:
                # Losses since the last learner step belong to the previous transition.
                if slot.segment:
                    self._settle(slot, result)
                else:
                    slot.tracker.start(slot.env.engine, slot.agent_pid)
                self._track_distance(slot)
                self._track_landing(slot, result)
                slot.segment.append(
                    Transition(
                        tokens=masks.tokens[0],
                        asset_mask=out["asset_masks"][i],
                        verb_mask=out["verb_mask"][i],
                        target_mask=out["target_masks"][i],
                        asset_idx=assets[i],
                        verb_idx=verbs[i],
                        target_idx=targets[i],
                        logp=logps[i],
                        value=values[i],
                        skill_idx=0,
                        globals=masks.globals[0],
                    )
                )
                slot.decisions += 1
                recorded += 1
            action = masks.to_action(asset_idx=assets[i], verb_idx=verbs[i], target_idx=targets[i])
            if key == LEARNER and isinstance(action, BuildShipAction):
                record_purchase(slot.purchases, action.ship_type)
                record_purchase(result.purchases, action.ship_type)
            pre = slot.tracker.snapshot(slot.env.engine, pid_of[i], action) if key == LEARNER else None
            self._step(slot, pid_of[i], action, result)
            if pre is not None:
                earned = slot.tracker.score(slot.env.engine, pid_of[i], action, pre)
                if earned:
                    slot.segment[-1].reward += sum(earned.values())
                    for name, value in earned.items():
                        result.shaping[name] += value
            if key == LEARNER:
                slot.tracker.mark_fleet(slot.env.engine, slot.agent_pid)
        return recorded

    def _settle(self, slot: _Slot, result: CollectResult) -> None:
        """Credit rewards that accrued since the learner's previous decision
        (ship losses, landing progress) to that decision's transition."""
        earned = slot.tracker.settle(slot.env.engine, slot.agent_pid)
        if earned and slot.segment:
            slot.segment[-1].reward += sum(earned.values())
            for name, value in earned.items():
                result.shaping[name] += value

    @staticmethod
    def _track_landing(slot: _Slot, result: CollectResult) -> None:
        d, _norm, count = slot.tracker.landing_distance(slot.env.engine, slot.agent_pid)
        for stats in (slot.landing, result.landing):
            stats["alive_sum"] += count
            stats["samples"] += 1
            if d is not None and (stats["min_dist"] is None or d < stats["min_dist"]):
                stats["min_dist"] = d

    def _step(self, slot: _Slot, pid: int, action, result: CollectResult) -> None:
        """env.step plus attack bookkeeping for either seat."""
        engine = slot.env.engine
        attack = None
        capture = None
        if isinstance(action, CapturePortAction) and engine.map.in_bounds(action.target) and engine.map.is_port(action.target):
            capture = int(engine.map.port_owner[action.target])
        if isinstance(action, AttackAction):
            attacker = engine.ships.get(action.attacker_id)
            target = engine.ships.get(action.target_id)
            if attacker is not None and target is not None and not attacker.has_attacked:
                attack = (attacker, target.type)
        slot.env.step(action)
        slot.steps += 1
        result.env_steps += 1
        if capture is not None:
            side = "learner" if pid == slot.agent_pid else "opp"
            taken = int(engine.map.port_owner[action.target]) == pid and capture != pid
            for stats in (slot.landing, result.landing):
                stats[f"{side}_capture_attempts"] += 1
                stats[f"{side}_captures"] += int(taken)
                stats[f"{side}_home_captures"] += int(taken and engine.winner == pid)
        if attack is not None and attack[0].has_attacked:
            side = "learner" if pid == slot.agent_pid else "opp"
            sunk = action.target_id not in engine.ships
            for stats in (slot.combat, result.combat):
                stats[f"{side}_attacks"] += 1
                if sunk:
                    stats[f"{side}_sinks"] += 1
                    book = stats["sunk_by_learner" if side == "learner" else "lost_by_learner"]
                    name = hull_name(attack[1])
                    book[name] = book.get(name, 0) + 1

    @staticmethod
    def _track_distance(slot: _Slot) -> None:
        """Running per-game min Manhattan distance between the two fleets,
        sampled at learner decisions."""
        engine = slot.env.engine
        mine = [engine.ships[s].position for s in engine.players[slot.agent_pid].owned_ship_ids]
        theirs = [engine.ships[s].position for s in engine.players[1 - slot.agent_pid].owned_ship_ids]
        if not mine or not theirs:
            return
        a = np.asarray(mine)
        b = np.asarray(theirs)
        d = int((np.abs(a[:, None, 0] - b[None, :, 0]) + np.abs(a[:, None, 1] - b[None, :, 1])).min())
        if slot.min_dist is None or d < slot.min_dist:
            slot.min_dist = d

    @torch.no_grad()
    def _values(self, items) -> List[float]:
        masks_list = [ActionMasks(slot.env.engine, pid) for slot, pid in items]
        keep = [i for i, m in enumerate(masks_list) if m.n > 0]
        values = [0.0] * len(items)
        if keep:
            tokens, pad, glob = _batch_inputs([masks_list[i] for i in keep])
            _, context = self.policy.encode(tokens, pad, glob)
            context = self.policy._with_skill(context, 0, len(keep))
            for i, v in zip(keep, self.policy.value(context).tolist()):
                values[i] = v
        return values

    def _bootstrap(self, result: CollectResult) -> None:
        pending = [s for s in self.slots if s.segment]
        while True:
            waiting, groups = [], {}
            for slot in pending:
                if not slot.segment:
                    continue
                pid = self._advance(slot, result, stop_on_finish=True)
                if pid is None:
                    continue
                if pid == slot.agent_pid:
                    waiting.append((slot, pid))
                else:
                    groups.setdefault(slot.opp_tag, []).append((slot, pid))
            if not groups:
                break
            for key, items in groups.items():
                self._act(key, items, result)
        if waiting:
            for slot, _ in waiting:
                self._settle(slot, result)
            for (slot, _), v in zip(waiting, self._values(waiting)):
                compute_gae(slot.segment, gamma=self.gamma, lam=self.lam, last_value=v)
                result.segments.append(slot.segment)
                slot.segment = []

    # ------------------------------------------------------------- collect
    def collect(self, budget: int) -> CollectResult:
        result = CollectResult()
        if budget <= 0:
            return result
        if self.slots is None:
            self._make_slots()
        while result.decisions < budget:
            groups: Dict[str, list] = {}
            for slot in self.slots:
                pid = self._advance(slot, result)
                key = LEARNER if pid == slot.agent_pid else slot.opp_tag
                groups.setdefault(key, []).append((slot, pid))
            remaining = budget - result.decisions
            if len(groups.get(LEARNER, [])) > remaining:
                groups[LEARNER] = groups[LEARNER][:remaining]
            for key, items in groups.items():
                result.decisions += self._act(key, items, result)
        self._bootstrap(result)
        result.open_games = len(self.slots)
        return result


# ---------------------------------------------------------------- transport
def state_to_numpy(state) -> Dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def numpy_to_state(state) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(v)) for k, v in state.items()}


def _pack(result: CollectResult) -> CollectResult:
    for seg in result.segments:
        for tr in seg:
            for name in _TENSOR_FIELDS:
                val = getattr(tr, name)
                if isinstance(val, torch.Tensor):
                    setattr(tr, name, val.clone().numpy())
    return result


def _unpack(result: CollectResult) -> CollectResult:
    for seg in result.segments:
        for tr in seg:
            for name in _TENSOR_FIELDS:
                val = getattr(tr, name)
                if isinstance(val, np.ndarray):
                    setattr(tr, name, torch.from_numpy(val))
    return result


def merge_results(results: Sequence[CollectResult]) -> CollectResult:
    out = CollectResult()
    for r in results:
        out.segments.extend(r.segments)
        out.finished.extend(r.finished)
        for k, v in r.purchases.items():
            out.purchases[k] = out.purchases.get(k, 0) + v
        for k, v in r.shaping.items():
            out.shaping[k] = out.shaping.get(k, 0.0) + v
        merge_combat(out.combat, r.combat)
        merge_landing(out.landing, r.landing)
        out.decisions += r.decisions
        out.env_steps += r.env_steps
        out.open_games += r.open_games
    return out


def split_budget(total: int, parts: int) -> List[int]:
    base, extra = divmod(int(total), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _snapshot_states(opponents: Dict[str, PolicyNetwork]) -> Dict[str, Any]:
    return {tag: net.state_dict() for tag, net in opponents.items()}


class LocalCollector:
    """In-process collector (num_workers=0): one GameCollector, same API."""

    def __init__(self, maps, map_info, *, games_per_worker: int, episode_offset: int, **kwargs) -> None:
        self.episode_offset = int(episode_offset)
        self.inner = GameCollector(
            maps, map_info, games=games_per_worker, worker_id=0, num_workers=1, episode_offset=episode_offset, **kwargs
        )

    def collect(self, policy: PolicyNetwork, opponents: Dict[str, PolicyNetwork], budget: int) -> CollectResult:
        self.inner.set_weights(policy.state_dict(), _snapshot_states(opponents), list(opponents))
        return self.inner.collect(budget)

    def next_episode(self) -> int:
        return self.episode_offset + self.inner.episodes_started

    def close(self) -> None:
        pass


def _worker_main(conn, kwargs: Dict[str, Any], threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))
    torch.manual_seed(int(kwargs.get("seed", 0)) * 1009 + int(kwargs.get("worker_id", 0)) + 1)
    try:
        collector = GameCollector(**kwargs)
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        msg = conn.recv()
        if msg[0] == "close":
            break
        _, learner, opponents, ring, budget = msg
        try:
            collector.set_weights(
                numpy_to_state(learner), {t: numpy_to_state(s) for t, s in opponents.items()}, ring
            )
            result = collector.collect(budget)
            conn.send(("ok", _pack(result), collector.episodes_started))
        except Exception:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class ParallelCollector:
    """``num_workers`` spawned processes, each a GameCollector at
    ``collect_threads`` torch threads. Weights go out before every collect."""

    def __init__(
        self,
        maps,
        map_info,
        *,
        num_workers: int,
        games_per_worker: int,
        collect_threads: int = 1,
        episode_offset: int = 0,
        **kwargs,
    ) -> None:
        self.num_workers = int(num_workers)
        self.episode_offset = int(episode_offset)
        self.started = [0] * self.num_workers
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(self.num_workers):
            parent, child = ctx.Pipe()
            wkw = dict(
                kwargs,
                maps=[str(m) for m in maps],
                map_info=list(map_info),
                games=games_per_worker,
                worker_id=w,
                num_workers=self.num_workers,
                episode_offset=self.episode_offset,
            )
            proc = ctx.Process(target=_worker_main, args=(child, wkw, collect_threads), daemon=True)
            proc.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(proc)

    def collect(self, policy: PolicyNetwork, opponents: Dict[str, PolicyNetwork], budget: int) -> CollectResult:
        learner = state_to_numpy(policy.state_dict())
        opp = {tag: state_to_numpy(net.state_dict()) for tag, net in opponents.items()}
        ring = list(opponents)
        for conn, b in zip(self.conns, split_budget(budget, self.num_workers)):
            conn.send(("collect", learner, opp, ring, b))
        results, errors = [], []
        for w, conn in enumerate(self.conns):
            msg = conn.recv()
            if msg[0] == "ok":
                results.append(_unpack(msg[1]))
                self.started[w] = msg[2]
            else:
                errors.append(f"worker {w}:\n{msg[1]}")
        if errors:
            raise RuntimeError("collection worker failed\n" + "\n".join(errors))
        return merge_results(results)

    def next_episode(self) -> int:
        return self.episode_offset + self.num_workers * max(self.started)

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close",))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def make_collector(maps, map_info, *, num_workers: int, games_per_worker: int, collect_threads: int, **kwargs):
    if num_workers and num_workers > 0:
        return ParallelCollector(
            maps,
            map_info,
            num_workers=num_workers,
            games_per_worker=games_per_worker,
            collect_threads=collect_threads,
            **kwargs,
        )
    return LocalCollector(maps, map_info, games_per_worker=games_per_worker, **kwargs)
