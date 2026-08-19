from __future__ import annotations

from helmet_safety.data.audit import check_split_integrity


def test_split_intersections_are_reported() -> None:
    """同一 ID 出现在两个正式 split 时，应报告具体交集并判定不互斥。"""

    result = check_split_integrity(
        {"train": ["a", "shared"], "val": ["b", "shared"], "test": ["c"], "trainval": ["a", "b", "shared"]}
    )

    assert result["intersections"] == {"train_val": ["shared"], "train_test": [], "val_test": []}
    assert result["mutually_exclusive"] is False


def test_trainval_must_equal_train_union_val() -> None:
    """trainval 必须恰好等于 train 与 val 的集合并集，不能缺少或额外增加 ID。"""

    good = check_split_integrity(
        {"train": ["a"], "val": ["b"], "test": ["c"], "trainval": ["a", "b"]}
    )
    bad = check_split_integrity(
        {"train": ["a"], "val": ["b"], "test": ["c"], "trainval": ["a", "extra"]}
    )

    assert good["trainval_matches_union"] is True
    assert bad["trainval_matches_union"] is False
    assert bad["trainval_missing"] == ["b"]
    assert bad["trainval_extra"] == ["extra"]
