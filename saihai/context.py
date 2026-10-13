"""Create an inference-context upgrade without changing trained weights or prompts."""

import argparse
import copy
import json
import os
import shutil
from pathlib import Path


def extend_contract(contract, model_config, limit):
    capacity = model_config.get("text_config", model_config).get("max_position_embeddings")
    if type(limit) is not int or type(capacity) is not int or not 0 < limit <= capacity:
        raise ValueError("Inference limit must fit the base model positional capacity")
    result = copy.deepcopy(contract)
    result.setdefault("training_max_input_tokens", contract["max_input_tokens"])
    result["max_input_tokens"] = limit
    return result


def upgrade_export(source, output, limit):
    source, output = Path(source), Path(output)
    contract = json.loads((source / "decision-inference.json").read_text())
    config = json.loads((source / "config.json").read_text())
    upgraded = extend_contract(contract, config, limit)
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.name != "decision-inference.json":
            if path.suffix == ".safetensors":
                os.link(path, output / path.name)
            else:
                shutil.copy2(path, output / path.name)
    (output / "decision-inference.json").write_text(
        json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n"
    )
    return upgraded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    args = parser.parse_args()
    upgrade_export(args.export, args.output, args.max_input_tokens)


if __name__ == "__main__":
    main()
