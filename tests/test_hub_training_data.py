"""Both Hub layouts preserve authored inputs, soft targets and global order."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from saihai.training import load_hub_records


def encoded(key, order):
    record = {
        "id": key,
        "input": {
            "state": "日本語",
            "type": "choice",
            "instructions": "選択",
            "options": [{"label": "a", "description": "A"}, {"label": "b", "description": "B"}],
        },
        "target": {"probabilities": [0.25, 0.75]},
    }
    return {
        "id": key,
        "source_order": order,
        "input_json": json.dumps(record["input"]),
        "record_json": json.dumps(record),
        "probabilities": record["target"]["probabilities"],
    }


def release(tmp_path, monkeypatch, *, legacy=False):
    files = (
        {"data/train.parquet": [encoded("a", 0), encoded("b", 1)]}
        if legacy
        else {
            "data/first/train.parquet": [encoded("b", 9)],
            "data/second/train.parquet": [encoded("a", 2)],
        }
    )
    manifest = {
        "training_view": "merged-no-3000",
        "reinforcement_included": False,
        "splits": {"train": {"rows": 2}},
        "source_file_sha256": {"train": "parent-hash"},
    }
    if not legacy:
        manifest.update(
            configuration_layout="one-config-per-subset",
            configs=[
                {"config_name": "first", "train_rows": 1},
                {"config_name": "second", "train_rows": 1},
                {"config_name": "validation-only", "train_rows": 0},
            ],
        )
    for name, rows in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), p)
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest))
    checks = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in [*files, "dataset-manifest.json"]
    }
    (tmp_path / "checksums.json").write_text(json.dumps(checks))
    calls = []

    def download(repo, filename, **kwargs):
        calls.append(filename)
        assert kwargs["revision"] == "r" * 40
        return str(tmp_path / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return (
        {"dataset": {"repo_id": "owner/data", "revision": "r" * 40, "split": "train"}},
        calls,
        files,
    )


def test_subset_layout_merges_all_trains_and_accepts_order_gaps(tmp_path, monkeypatch):
    config, calls, _ = release(tmp_path, monkeypatch)
    rows, receipt = load_hub_records(config)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert all(r["target"]["probabilities"] == [0.25, 0.75] for r in rows)
    assert set(receipt["train_artifact_sha256"]) == {
        "data/first/train.parquet",
        "data/second/train.parquet",
    }
    assert not any("validation" in name for name in calls)


def test_legacy_layout_still_loads(tmp_path, monkeypatch):
    config, _, _ = release(tmp_path, monkeypatch, legacy=True)
    rows, receipt = load_hub_records(config)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert receipt["rows"] == 2


@pytest.mark.parametrize(
    "mutation", ["checksum", "duplicate_order", "duplicate_id", "target", "counts"]
)
def test_corrupt_subset_release_fails_closed(tmp_path, monkeypatch, mutation):
    config, _, files = release(tmp_path, monkeypatch)
    p = tmp_path / "data/first/train.parquet"
    if mutation == "counts":
        m = json.loads((tmp_path / "dataset-manifest.json").read_text())
        m["configs"][0]["train_rows"] = 2
        (tmp_path / "dataset-manifest.json").write_text(json.dumps(m))
    elif mutation == "checksum":
        p.write_bytes(p.read_bytes() + b"changed")
    else:
        row = files["data/first/train.parquet"][0]
        if mutation == "duplicate_order":
            row["source_order"] = 2
        elif mutation == "duplicate_id":
            row["id"] = "a"
            r = json.loads(row["record_json"])
            r["id"] = "a"
            row["record_json"] = json.dumps(r)
        elif mutation == "target":
            row["probabilities"] = [0.5, 0.5]
        pq.write_table(pa.Table.from_pylist([row]), p)
    if mutation != "checksum":
        c = json.loads((tmp_path / "checksums.json").read_text())
        for name in [*files, "dataset-manifest.json"]:
            c[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        (tmp_path / "checksums.json").write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_hub_records(config)


def test_latest_resolves_main_once_and_pins_every_download(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config, calls, _ = release(tmp_path, monkeypatch)
    config["dataset"]["revision"] = "latest"
    lookups = []

    def dataset_info(self, repo_id, *, revision):
        lookups.append((repo_id, revision))
        return SimpleNamespace(sha="r" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", dataset_info)
    rows, receipt = load_hub_records(config)
    assert len(rows) == 2
    assert len(calls) == 4
    assert lookups == [("owner/data", "main")]
    assert receipt["requested_revision"] == "latest"
    assert receipt["revision"] == "r" * 40


def test_current_manifest_without_historical_parent_fields(tmp_path, monkeypatch):
    config, _, _ = release(tmp_path, monkeypatch)
    path = tmp_path / "dataset-manifest.json"
    manifest = json.loads(path.read_text())
    for key in ("training_view", "reinforcement_included", "source_file_sha256"):
        manifest.pop(key)
    manifest.update(
        dataset="owner/data",
        schema_version=1,
        file_integrity={"algorithm": "SHA-256", "manifest": "checksums.json"},
    )
    path.write_text(json.dumps(manifest))
    checks_path = tmp_path / "checksums.json"
    checks = json.loads(checks_path.read_text())
    checks["dataset-manifest.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks_path.write_text(json.dumps(checks))
    rows, receipt = load_hub_records(config)
    assert len(rows) == 2
    assert receipt["source_file_sha256"] is None


def test_validation_uses_only_validation_shards_at_same_revision(tmp_path, monkeypatch):
    config, calls, _ = release(tmp_path, monkeypatch)
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["validation"] = {"rows": 1}
    manifest["source_file_sha256"]["validation"] = "validation-parent"
    for item in manifest["configs"]:
        item["validation_rows"] = int(item["config_name"] == "validation-only")
    manifest_path.write_text(json.dumps(manifest))
    validation_path = tmp_path / "data/validation-only/validation.parquet"
    validation_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([encoded("v", 7)]), validation_path)
    checks_path = tmp_path / "checksums.json"
    checks = json.loads(checks_path.read_text())
    for path in (manifest_path, validation_path):
        checks[str(path.relative_to(tmp_path))] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks_path.write_text(json.dumps(checks))
    rows, receipt = load_hub_records(config, split="validation")
    assert [r["id"] for r in rows] == ["v"]
    assert receipt["split"] == "validation"
    assert receipt["validation_shards"] == 1
    assert set(receipt["validation_artifact_sha256"]) == {"data/validation-only/validation.parquet"}
    assert not any(name.endswith("/train.parquet") for name in calls)


@pytest.mark.parametrize("conflict", [False, True])
def test_filter_metadata_is_restored_and_conflicts_rejected(tmp_path, monkeypatch, conflict):
    config, _, _ = release(tmp_path, monkeypatch)
    path = tmp_path / "data/first/train.parquet"
    row = encoded("b", 9)
    row["jdmb_source_family"] = True
    if conflict:
        record = json.loads(row["record_json"])
        record["jdmb_source_family"] = False
        row["record_json"] = json.dumps(record)
    pq.write_table(pa.Table.from_pylist([row]), path)
    checksum_path = tmp_path / "checksums.json"
    checksums = json.loads(checksum_path.read_text())
    checksums["data/first/train.parquet"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path.write_text(json.dumps(checksums))
    if conflict:
        with pytest.raises(ValueError, match="filter metadata"):
            load_hub_records(config)
    else:
        rows, _ = load_hub_records(config)
        assert rows[1]["jdmb_source_family"] is True
        assert "jdmb_source_family" not in rows[1]["input"]
        assert rows[1]["target"]["probabilities"] == [0.25, 0.75]
