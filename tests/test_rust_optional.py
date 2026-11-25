"""The Rust extension is optional. This module always runs."""

from __future__ import annotations

import pytest

from battleboats.rust_ext import available, module


def test_rust_extension_is_optional() -> None:
    if available():
        assert module().schema_version() == 1
    else:
        with pytest.raises(ImportError, match="battleboats_core is not installed"):
            module()
