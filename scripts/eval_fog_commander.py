"""Deprecated entry point. Use scripts/eval_scripted_agent.py.

``fog_commander`` is still accepted as ``--focal`` / ``--opponents``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("eval_scripted_agent.py")
    print(
        "scripts/eval_fog_commander.py is deprecated; use scripts/eval_scripted_agent.py",
        file=sys.stderr,
    )
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
