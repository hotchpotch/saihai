"""Read portable, subset-separated typed decision DatasetDict releases."""

import hashlib
import json
from pathlib import Path

from datasets import load_from_disk


def verify_release(root):
    root = Path(root)
    manifest = json.loads((root / "release-manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Artifact path escapes release root")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def load_subset(root, config_name):
    root = Path(root)
    manifest = json.loads((root / "release-manifest.json").read_text())
    config = next(c for c in manifest["configs"] if c["config_name"] == config_name)
    return load_from_disk(root / config["disk_path"])


def iter_records(root, split):
    root = Path(root)
    manifest = json.loads((root / "release-manifest.json").read_text())
    rows = []
    for config in manifest["configs"]:
        subset = load_subset(root, config["config_name"])
        if split not in subset:
            continue
        dataset = subset[split]
        for row in dataset:
            record = json.loads(row["record_json"])
            if record["target"]["probabilities"] != row["probabilities"]:
                raise ValueError("Soft targets changed during serialization")
            if record["input"] != json.loads(row["input_json"]):
                raise ValueError("Input changed during serialization")
            rows.append((row["source_order"], record))
    rows.sort(key=lambda item: item[0])
    if [i for i, _ in rows] != list(range(len(rows))):
        raise ValueError("Missing or duplicated source ordering")
    for _, record in rows:
        yield record
