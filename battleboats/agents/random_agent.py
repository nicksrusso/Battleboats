"""Random baseline agent. Uniformly samples from engine.enumerate_legal().

Used for the first end-to-end smoke test of the env wrapper, and as a
sparring partner for early debugging before any learning is wired up.
"""
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from battleboats.core.actions import Action
    from battleboats.core.gameEngine import gameEngine


def random_action(engine: "gameEngine", player_id: int, rng: random.Random) -> "Action":
    """Sample a uniformly random legal action for player_id.

    engine.enumerate_legal(player_id) always includes EndTurnAction for the
    current player, so the returned list is non-empty whenever it's
    player_id's turn and the game isn't over.

    Raises:
        IndexError if called when no legal actions exist (e.g., player_id
        is not the current player, or the game is already terminal).
    """
    return rng.choice(engine.enumerate_legal(player_id))
