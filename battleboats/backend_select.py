"""Choose the Python reference engine or the optional Rust core."""

from __future__ import annotations


def resolve_backend(name: str) -> str:
    """``auto`` uses Rust when the extension is installed, else Python.

    ``python`` always uses the reference engine. ``rust`` errors if the
    extension was not built.
    """
    choice = (name or "auto").strip().lower()
    if choice not in {"auto", "python", "rust"}:
        raise ValueError(f"unknown backend {name!r}")
    from battleboats.rust_ext import available

    if choice == "python":
        return "python"
    if choice == "rust" and not available():
        raise RuntimeError(
            "backend rust was requested but battleboats_core is not installed. "
            "Build it with: poetry install -E rust && "
            "poetry run maturin develop --release -m crates/battleboats-py/Cargo.toml"
        )
    if choice == "auto":
        return "rust" if available() else "python"
    return "rust"
