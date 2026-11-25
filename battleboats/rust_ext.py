"""Optional Rust engine core.

The extension module ``battleboats_core`` is built with maturin. Importing this
wrapper never raises: :func:`available` is false until the extension exists, and
the pure-Python engine does not import it.
"""

from __future__ import annotations

try:
    import battleboats_core as _core
except ImportError:  # pragma: no cover - exercised when the extension is absent
    _core = None


def available() -> bool:
    """True when the PyO3 extension is installed in this interpreter."""
    return _core is not None


def module():
    """Return the extension module, or raise ``ImportError`` with build steps."""
    if _core is None:
        raise ImportError(
            "battleboats_core is not installed. From the repo root: "
            "poetry install -E rust && "
            "poetry run maturin develop --release -m crates/battleboats-py/Cargo.toml"
        )
    return _core
