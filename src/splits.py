"""Time-based datasets with an explicit label-availability snapshot."""

from __future__ import annotations

import pandas as pd


def closed_split(frame: pd.DataFrame, *, start: str, end: str, as_of: str) -> pd.DataFrame:
    """Include only decisions in [start, end) with labels delivered by as_of."""
    ordered_at = pd.to_datetime(frame["ordered_at"], utc=True)
    labels_at = pd.to_datetime(frame["label_available_at"], utc=True)
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    cutoff = pd.Timestamp(as_of)
    if any(ts.tzinfo is None for ts in (start_at, end_at, cutoff)):
        raise ValueError("Split boundaries must include UTC offsets")
    if start_at >= end_at or cutoff < end_at:
        raise ValueError("Invalid split boundaries or label snapshot")
    mask = (ordered_at >= start_at) & (ordered_at < end_at) & (labels_at <= cutoff)
    return frame.loc[mask].copy().reset_index(drop=True)


def label_maturity(frame: pd.DataFrame, as_of: str) -> pd.Series:
    """Return whether every row's label is actually available by as_of."""
    return pd.to_datetime(frame["label_available_at"], utc=True) <= pd.Timestamp(as_of)
