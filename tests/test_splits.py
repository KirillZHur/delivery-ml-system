import pandas as pd
import pytest

from src.splits import closed_split, label_maturity


FRAME = pd.DataFrame([
    {"order_id": "old", "ordered_at": "2026-01-01T00:00:00Z", "label_available_at": "2026-01-08T01:00:00Z"},
    {"order_id": "young", "ordered_at": "2026-01-01T01:00:00Z", "label_available_at": "2026-01-08T02:00:00Z"},
    {"order_id": "next", "ordered_at": "2026-02-01T00:00:00Z", "label_available_at": "2026-02-08T01:00:00Z"},
])


def test_label_at_snapshot_included_not_before():
    mask = label_maturity(FRAME, "2026-01-08T01:00:00Z")
    assert mask.to_list() == [True, False, False]


def test_split_includes_start_excludes_end_and_rejects_unready():
    got = closed_split(FRAME, start="2026-01-01T00:00:00Z",
                       end="2026-02-01T00:00:00Z", as_of="2026-02-01T00:00:00Z")
    assert got["order_id"].tolist() == ["old", "young"]


def test_split_rejects_early_snapshot():
    with pytest.raises(ValueError, match="Invalid"):
        closed_split(FRAME, start="2026-01-01T00:00:00Z",
                     end="2026-02-01T00:00:00Z", as_of="2026-01-31T23:00:00Z")
