"""Round-trip soft decisions without exposing targets in model inputs."""

import json

from datasets import Dataset, DatasetDict

from saihai.datasets import iter_records, load_subset


def test_saved_soft_targets_and_input_roundtrip(tmp_path):
    decision = {
        "input": {
            "state": {"amount": None},
            "instructions": "Does it meet the limit?",
            "type": "noul",
        },
        "target": {"probabilities": [0.5, 0.5]},
    }
    row = {
        "source_order": 0,
        "record_json": json.dumps(decision),
        "input_json": json.dumps(decision["input"]),
        "probabilities": [0.5, 0.5],
    }
    DatasetDict({"train": Dataset.from_list([row])}).save_to_disk(str(tmp_path / "subset"))
    (tmp_path / "release-manifest.json").write_text(
        json.dumps({"configs": [{"config_name": "example", "disk_path": "subset"}]})
    )
    assert len(load_subset(tmp_path, "example")["train"]) == 1
    assert list(iter_records(tmp_path, "train")) == [decision]
