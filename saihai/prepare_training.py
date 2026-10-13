"""Resolve and audit training data on CPU, without loading a model or starting training."""

import argparse
import copy
from pathlib import Path

import yaml

from .training import dump, load_config, load_hub_records, select_validation_records
from .training_filter import filter_training_records


def prepare(config, output):
    rows, source = load_hub_records(config)
    frozen = copy.deepcopy(config)
    frozen["dataset"]["revision"] = source["revision"]
    validation, validation_source = load_hub_records(frozen, split="validation")
    for key in ("id", "parent_id", "group_id"):
        identities = {r[key] for r in rows if r.get(key) is not None}
        if identities & {r[key] for r in validation if r.get(key) is not None}:
            raise ValueError("Training and validation identities overlap: " + key)
    kept, receipt = filter_training_records(rows, config["dataset"].get("train_filter", {}))
    selected = select_validation_records(
        validation, config["training"]["validation_per_subset"], config["training"]["seed"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False))
    dump(output / "train-filter-receipt.json", receipt)
    dump(output / "data-receipt.json", source)
    dump(output / "validation-receipt.json", dict(validation_source, selected=len(selected)))
    dump(
        output / "prepared-ids.json",
        {
            "train": [r["id"] for r in kept],
            "validation": [r["id"] for r in selected],
        },
    )
    return {
        "revision": source["revision"],
        "before": len(rows),
        "remaining": len(kept),
        "excluded": receipt["excluded"],
        "validation": len(selected),
        "tokenization_checked": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(load_config(args.config), args.output))


if __name__ == "__main__":
    main()
