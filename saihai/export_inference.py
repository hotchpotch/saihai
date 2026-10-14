"""Attach the standalone runtime to an existing standard Transformers export."""

import argparse
import json
import shutil
from pathlib import Path


def export_inference(directory):
    """Copy the standalone file verbatim and register only the AutoModel entry point."""
    directory = Path(directory)
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    if config["model_type"] not in {"llama", "qwen3_5"}:
        raise ValueError("Unsupported backbone for standalone inference")
    if not (directory / "decision-inference.json").is_file():
        raise FileNotFoundError(directory / "decision-inference.json")
    shutil.copyfile(
        Path(__file__).with_name("inference_saihai.py"), directory / "inference_saihai.py"
    )
    name = "SaihaiModel" if config["model_type"] == "llama" else "SaihaiQwenModel"
    config.setdefault("auto_map", {})["AutoModel"] = f"inference_saihai.{name}"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export_inference(parser.parse_args().directory)
