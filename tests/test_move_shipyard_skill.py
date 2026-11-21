"""Move and shipyard sandboxes, and the shared observation width."""

from pathlib import Path

from battleboats.core.actions import (
    AttackAction,
    BuildShipAction,
    CapturePortAction,
    EndTurnAction,
    MoveAction,
)
from battleboats.core.shipyard.ship_data import BASE_STATS
from battleboats.core.shipyard.ship_type import ShipType
from battleboats.envs.action_masks import ActionMasks
from battleboats.envs.factory import make_env
from battleboats.envs.observation import TOKEN_DIM
from battleboats.envs.shipyard_skill_aec import SHIPYARD_TASKS, ShipyardSkillAEC
from battleboats.envs.skill_maps import (
    write_move_map,
    write_shipyard_map,
    write_skill_map,
    write_two_home_map,
)


def _closer(env, goal):
    legal = env.infos["player_0"]["legal_actions"]
    moves = [a for a in legal if isinstance(a, MoveAction)]
    assert moves

    def dist(action: MoveAction) -> int:
        return abs(action.destination[0] - goal[0]) + abs(action.destination[1] - goal[1])

    env.step(min(moves, key=dist))


def test_move_reaches_waypoint_and_pays_once(tmp_path: Path) -> None:
    env = make_env("move", write_move_map(tmp_path / "m.json", 2, 3), max_turns=8, randomize=False)
    env.reset(seed=0)
    waypoint = env._waypoint()
    assert env._dist() > 1
    assert env.reached is False
    legal = env.infos["player_0"]["legal_actions"]
    assert all(isinstance(a, (MoveAction, EndTurnAction)) for a in legal)
    for _ in range(8):
        if env.terminations["player_0"]:
            break
        _closer(env, waypoint)
    assert env.reached
    assert env.rewards["player_0"] == 1.0
    assert env.terminations["player_0"] is True
    env.close()


def test_shipyard_buys_the_ship_the_board_asks_for(tmp_path: Path) -> None:
    path = write_shipyard_map(tmp_path / "s.json", 3, 3)
    wanted = {
        "merchant": ShipType.MERCHANT,
        "combat": ShipType.DESTROYER,
        "landing": ShipType.LANDING,
        "builder": ShipType.BUILDER,
    }
    for task, ship_type in wanted.items():
        env = ShipyardSkillAEC(path, max_turns=6, task=task)
        env.reset(seed=0)
        assert env.active_task == task
        legal = env.infos["player_0"]["legal_actions"]
        assert all(isinstance(a, (BuildShipAction, EndTurnAction)) for a in legal)
        assert not any(isinstance(a, AttackAction) for a in legal)
        builds = [a for a in legal if isinstance(a, BuildShipAction) and a.ship_type is ship_type]
        assert builds, task
        env.step(builds[0])
        assert env.built_correct, task
        assert env.rewards["player_0"] == 1.0
        assert env.terminations["player_0"] is True
        env.close()


def test_shipyard_wrong_type_is_not_success_and_cash_stays_put(tmp_path: Path) -> None:
    path = write_shipyard_map(tmp_path / "s.json", 3, 3)
    env = ShipyardSkillAEC(path, max_turns=6, task="landing")
    env.reset(seed=0)
    cash = BASE_STATS[ShipType.LANDING].cost
    assert env.engine.players[0].cash == cash
    legal = env.infos["player_0"]["legal_actions"]
    wrong = [a for a in legal if isinstance(a, BuildShipAction) and a.ship_type is ShipType.DESTROYER]
    env.step(wrong[0])
    assert env.built_correct is False
    assert env.rewards["player_0"] == 0.0
    assert env.terminations["player_0"] is False
    # A destroyer costs less than a landing, and the spent cash is not refunded.
    assert env.engine.players[0].cash == cash - BASE_STATS[ShipType.DESTROYER].cost
    env.close()

    env = ShipyardSkillAEC(path, max_turns=6, task="merchant")
    env.reset(seed=1)
    opening = BASE_STATS[ShipType.MERCHANT].cost
    env.step(EndTurnAction())
    assert env.engine.players[0].cash == opening
    assert env.built_correct is False
    env.close()


def test_opening_random_layout_has_full_game_cues(tmp_path: Path) -> None:
    env = make_env(
        "opening",
        write_two_home_map(tmp_path / "o.json", 6, 6),
        max_turns=8,
        randomize=True,
    )
    env.reset(seed=2)
    ports = env.engine.players[0].owned_port_positions
    assert len(ports) >= 2
    empty = [
        (x, y)
        for x in range(6)
        for y in range(6)
        if env.engine.map.is_land((x, y)) and not env.engine.map.is_port((x, y))
    ]
    assert empty
    assert env.engine.players[0].port_sightings
    env.close()


def test_opening_buys_a_landing_and_captures(tmp_path: Path) -> None:
    env = make_env(
        "opening",
        write_two_home_map(tmp_path / "o.json", 6, 6),
        max_turns=12,
        randomize=False,
        approach_coef=0.0,
    )
    env.reset(seed=0)
    legal = env.infos["player_0"]["legal_actions"]
    buy = next(a for a in legal if isinstance(a, BuildShipAction) and a.ship_type is ShipType.LANDING)
    env.step(buy)
    assert env.rewards["player_0"] == 0.25
    assert env.captured is False
    goal = env.engine.players[1].home_port
    for _ in range(30):
        if env.terminations["player_0"]:
            break
        legal = env.infos["player_0"]["legal_actions"]
        caps = [a for a in legal if isinstance(a, CapturePortAction)]
        if caps:
            env.step(caps[0])
            break
        moves = [a for a in legal if isinstance(a, MoveAction)]
        if not moves:
            env.step(next(a for a in legal if isinstance(a, EndTurnAction)))
            continue

        def dist(action: MoveAction) -> int:
            return abs(action.destination[0] - goal[0]) + abs(action.destination[1] - goal[1])

        env.step(min(moves, key=dist))
    assert env.captured
    assert env.rewards["player_0"] == 1.0
    assert env.terminations["player_0"] is True
    env.close()


def test_shipyard_cycles_tasks_and_token_width_matches_full_game(tmp_path: Path) -> None:
    seen = set()
    path = write_shipyard_map(tmp_path / "s.json", 3, 3)
    env = ShipyardSkillAEC(path, max_turns=4, randomize_layout=False)
    for seed in range(8):
        env.reset(seed=seed)
        seen.add(env.active_task)
        masks = ActionMasks(env.engine, 0)
        assert int(masks.tokens.shape[-1]) == TOKEN_DIM
    assert seen == set(SHIPYARD_TASKS)
    env.close()

    for skill, width, height in (
        ("move", 3, 3),
        ("merchant", 3, 3),
        ("combat", 3, 3),
        ("landing", 3, 3),
        ("builder", 3, 3),
        ("opening", 4, 4),
        ("full", 4, 4),
    ):
        live = make_env(
            skill,
            write_skill_map(tmp_path / f"{skill}.json", skill, width, height),
            max_turns=4,
        )
        live.reset(seed=0)
        masks = ActionMasks(live.engine, 0)
        assert int(masks.tokens.shape[-1]) == TOKEN_DIM
        live.close()
