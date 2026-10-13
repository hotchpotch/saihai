"""Evaluate a trained export using an external S1MB checkout and its environment."""

import argparse
import gc
import json
from pathlib import Path


def main():
    from s1mb.data import (
        DATA_DIR,
        ModelInfo,
        Prediction,
        load_benchmark,
        load_category,
        validate_result,
    )
    from s1mb.dataset_source import dataset_session
    from s1mb.runner import evaluate

    from .model import Saihai
    from .training import dump, sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-revision")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    model = Saihai.from_pretrained(args.export)
    revision = (
        sha256(args.export.parent / "export-hashes.json")
        if (args.export.parent / "export-hashes.json").exists()
        else sha256(args.export / "model.safetensors")
    )

    filter_path = args.export.parent / "train-filter-receipt.json"
    filter_receipt = json.loads(filter_path.read_text()) if filter_path.exists() else None

    class Adapter:
        def metadata(self):
            return ModelInfo(
                id=args.run_id,
                adapter="saihai-pplx-transformers",
                revision=revision,
                settings={
                    "max_input_tokens": model.backend.contract["max_input_tokens"],
                    "codebook": "pplx-decider",
                    "candidate_order": "authored; noul false,true",
                    "training_source_family_exposure": (
                        True
                        if filter_receipt and filter_receipt["remaining_marked_source_family_rows"]
                        else None
                    ),
                    "training_source_filter": filter_receipt["policy"] if filter_receipt else None,
                    "source_filter_scope": "Recorded identities; semantic overlap and pretraining unknown",
                },
            )

        def predict(self, case):
            predictions = []
            for q in case.questions:
                options = list(q.options)
                if q.task == "noul":
                    options = sorted(options, key=lambda o: o.id == "true")
                descriptions = [
                    json.loads(o.description_json) if o.description_json else o.description
                    for o in options
                ]
                inp = {
                    "state": case.state,
                    "instructions": json.loads(q.instructions_json)
                    if q.instructions_json
                    else q.instructions,
                    "type": q.task,
                    "options": [
                        {
                            "label": o.id if q.task == "noul" else f"option_{i}",
                            "description": description,
                            **({"value": o.value} if q.task == "score" else {}),
                        }
                        for i, (o, description) in enumerate(
                            zip(options, descriptions, strict=True)
                        )
                    ],
                }
                if q.system_prompt:
                    inp["instructions"] = {
                        "system": q.system_prompt,
                        "instruction": inp["instructions"],
                    }
                ps = model.backend.predict(inp)["probabilities"]
                predictions.append(
                    Prediction(
                        case_id=case.case_id,
                        question_id=q.id,
                        probabilities={o.id: p for o, p in zip(options, ps, strict=True)},
                    )
                )
            return predictions

        def close(self):
            pass

    adapter = Adapter()
    results = []
    with dataset_session(DATA_DIR, offline=True, category="japanese-v1"):
        source = json.loads((DATA_DIR / "datasets/hub-sources/japanese-v1.json").read_text())
        if source.get("repo_id") != "hotchpotch/S1MB-japanese-v1" or source.get("category") != "japanese-v1":
            raise ValueError("Installed dataset is not S1MB japanese-v1")
        if args.data_revision and source["revision"] != args.data_revision:
            raise ValueError("Installed S1MB dataset differs from the pinned revision")
        category = load_category(DATA_DIR, "japanese-v1")
        for name in category.benchmarks:
            benchmark = load_benchmark(DATA_DIR, name)
            result = evaluate(
                adapter, DATA_DIR, benchmark, DATA_DIR / "results", args.run_id, limit=args.limit
            )
            validate_result(DATA_DIR, result)
            if result.counts.failed:
                raise RuntimeError("S1MB inference failed: " + name)
            results.append(result.model_dump(mode="json"))
            print("S1MB", name, result.counts, flush=True)
        dump(
            args.output,
            {
                "category": "japanese-v1",
                "limit_per_benchmark": args.limit,
                "data_source": source,
                "run_id": args.run_id,
                "results": results,
            },
        )
    gc.collect()
    print("S1MB validated", len(results), "benchmarks", flush=True)


if __name__ == "__main__":
    main()
