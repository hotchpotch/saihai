import copy

import pytest

from saihai.training_filter import filter_training_records, validate_filter


def row(key, flag=False, parent=None, group=None, dataset="unrelated"):
    return dict(
        id=key,
        parent_id=parent or key,
        group_id=group or key,
        subset="subset",
        dataset_group="group",
        jdmb_source_family=flag,
        provenance={"dataset": dataset},
        input={"type": "choice", "state": "JCoLA"},
        target={"probabilities": [0.2, 0.8]},
    )


def test_filter_preserves_rows_and_removes_transitively_connected_groups():
    rows = [
        row("a", True, group="g"),
        row("b", parent="p", group="g"),
        row("c", parent="p"),
        row("d"),
    ]
    original = copy.deepcopy(rows)
    kept, receipt = filter_training_records(rows, {"exclude_s1mb_japanese_source_family": True})
    assert kept == [rows[3]]
    assert rows == original
    assert receipt["excluded_ids"] == ["a", "b", "c"]
    assert receipt["direct_matches"] == 1
    assert receipt["remaining"] == 1


def test_provenance_only_not_input_or_teacher():
    rows = [row("a", dataset="osekilab/JCoLA"), row("b")]
    rows[1]["provenance"]["teacher"] = {"dataset": "osekilab/JCoLA"}
    kept, receipt = filter_training_records(rows, {"exclude_source_patterns": ["jcola"]})
    assert kept == [rows[1]]
    assert receipt["excluded_ids"] == ["a"]


@pytest.mark.parametrize(
    "policy",
    [
        False,
        {"unknown": True},
        {"exclude_s1mb_japanese_source_family": "yes"},
        {"exclude_subsets": "subset"},
        {"exclude_source_patterns": ["["]},
        {"exclude_dataset_groups": [1]},
    ],
)
def test_invalid_filter(policy):
    with pytest.raises(ValueError):
        validate_filter(policy)


def test_missing_marker_fails_closed_only_when_requested():
    rows = [row("a")]
    del rows[0]["jdmb_source_family"]
    assert filter_training_records(rows, {})[0] == rows
    with pytest.raises(ValueError, match="jdmb_source_family"):
        filter_training_records(rows, {"exclude_s1mb_japanese_source_family": True})


def test_explicit_lists_and_empty_result():
    rows = [row("a"), row("b")]
    rows[1]["subset"] = "keep"
    assert filter_training_records(rows, {"exclude_subsets": ["subset"]})[0] == [rows[1]]
    with pytest.raises(ValueError, match="No training"):
        filter_training_records(rows, {"exclude_dataset_groups": ["group"]})


def test_prepare_freezes_revision_and_preserves_validation(tmp_path, monkeypatch):
    import json

    from saihai import prepare_training

    train = [row("a", True), row("b")]
    val = [row("v")]
    calls = []

    def load(config, *, split="train"):
        calls.append((split, config["dataset"]["revision"]))
        return (train if split == "train" else val), {"revision": "r" * 40}

    monkeypatch.setattr(prepare_training, "load_hub_records", load)
    monkeypatch.setattr(prepare_training, "select_validation_records", lambda rows, *_: rows)
    config = {
        "dataset": {"revision": "latest", "train_filter": {"exclude_s1mb_japanese_source_family": True}},
        "training": {"validation_per_subset": 3, "seed": 42},
    }
    summary = prepare_training.prepare(config, tmp_path / "prepared")
    assert calls == [("train", "latest"), ("validation", "r" * 40)]
    assert summary["remaining"] == 1
    assert json.loads((tmp_path / "prepared/prepared-ids.json").read_text()) == {
        "train": ["b"],
        "validation": ["v"],
    }
    assert config["dataset"]["revision"] == "latest"


def test_holdout_recipes_change_only_training_filter():
    from pathlib import Path

    from saihai.training import load_config

    for name in ("llmjp41-8b", "qwen35-4b", "qwen35-0.8b", "qwen35-9b"):
        baseline = load_config(Path("configs") / f"{name}.yaml")
        filtered = load_config(Path("configs") / f"{name}-s1mb-japanese-family-holdout.yaml")
        assert filtered["dataset"].pop("train_filter")["exclude_s1mb_japanese_source_family"] is True
        assert filtered == baseline


def test_legacy_family_filter_alias_matches_current_name():
    rows = [row("marked", True), row("kept", False)]
    current, _ = filter_training_records(rows, {"exclude_s1mb_japanese_source_family": True})
    legacy, _ = filter_training_records(rows, {"exclude_jdmb_source_family": True})
    assert current == legacy
    with pytest.raises(ValueError, match="Conflicting"):
        filter_training_records(
            rows,
            {"exclude_s1mb_japanese_source_family": True, "exclude_jdmb_source_family": False},
        )
