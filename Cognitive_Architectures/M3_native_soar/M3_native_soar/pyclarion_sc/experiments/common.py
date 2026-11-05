"""Shared helpers for experiments."""
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any

N_LEVELS = 21  # explicitness codes 0..20 (step 0.05); see REPORT.md, "Modeling choices"


def config_dict(cfg) -> dict:
    return asdict(cfg)


def update_config(cfg, **overrides):
    """Return a copy of a dataclass config with some fields changed."""
    names = {f.name for f in fields(cfg)}
    bad = set(overrides) - names
    if bad:
        raise KeyError(f"Unknown parameter(s): {sorted(bad)}")
    d = asdict(cfg)
    d.update(overrides)
    return type(cfg)(**d)


def param_source(cfg_cls) -> dict[str, str]:
    """Each config field's metadata 'source': 'paper', 'chosen' or 'calibrated'."""
    return {f.name: f.metadata.get("source", "chosen") for f in fields(cfg_cls)}


def P(default: Any, source: str, note: str = ""):
    """Config field with provenance metadata (paper / chosen / calibrated)."""
    from dataclasses import field
    if isinstance(default, (list, dict, tuple)):
        return field(default_factory=lambda: type(default)(default), metadata={"source": source, "note": note})
    return field(default=default, metadata={"source": source, "note": note})
