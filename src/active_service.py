"""FastAPI launcher for the registry-selected release (restart on promotion)."""
from __future__ import annotations
import os
from pathlib import Path
from data.generate import ROOT
from src.retrain import active_bundle
from src.service import create_app


def create_managed_app(registry_dir: Path | None = None, **kwargs):
    registry = Path(registry_dir) if registry_dir is not None else Path(
        os.environ.get("DELIVERY_REGISTRY_DIR", str(ROOT / "retraining"))
    )
    return create_app(artifact_dir=active_bundle(registry), **kwargs)


app = create_managed_app()
