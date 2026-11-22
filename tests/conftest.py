from pathlib import Path

import pytest

from data.generate import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "configs/data.json")


@pytest.fixture
def tiny_config(config):
    import copy

    cfg = copy.deepcopy(config)
    cfg["periods"] = [
        {"name": "train", "start": "2026-01-01T00:00:00Z",
         "end": "2026-01-05T00:00:00Z", "orders": 35},
        {"name": "valid", "start": "2026-01-05T00:00:00Z",
         "end": "2026-01-08T00:00:00Z", "orders": 15},
    ]
    return cfg
