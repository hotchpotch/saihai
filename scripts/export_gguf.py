"""Export a trained model as native System One BF16 and quantized GGUF files."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--vocab-gguf", type=Path)
    parser.add_argument("--quantize-binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="saihai")
    parser.add_argument("--quantizations", nargs="*", default=["Q8_0", "Q6_K", "Q4_K_M"],
                        choices=["Q8_0", "Q6_K", "Q4_K_M"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be a filename stem")
    if args.threads < 1 or len(set(args.quantizations)) != len(args.quantizations):
        parser.error("Use positive threads and unique quantizations")
    export, upstream = args.export.resolve(), args.llama_cpp.resolve()
    binary = (args.quantize_binary or upstream / "build/bin/llama-quantize").resolve()
    if args.quantizations and not binary.is_file():
        parser.error("Build llama-quantize or supply --quantize-binary")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bf16 = output / f"{args.name}-BF16.gguf"
    command = [sys.executable, "-m", "saihai.convert_gguf", "--export", str(export),
               "--llama-cpp", str(upstream), "--output", str(bf16)]
    if args.vocab_gguf:
        command += ["--vocab-gguf", str(args.vocab_gguf.resolve())]
    subprocess.run(command, check=True)
    sys.path.insert(0, str(upstream / "gguf-py"))
    import gguf

    source = gguf.GGUFReader(bf16)
    native = {k: f.contents() for k, f in source.fields.items()
              if ".decision." in k or k.startswith("tokenizer.")}
    artifacts = {bf16.name: {"sha256": digest(bf16), "bytes": bf16.stat().st_size}}
    for kind in args.quantizations:
        path = output / f"{args.name}-{kind}.gguf"
        subprocess.run([str(binary), str(bf16), str(path), kind, str(args.threads)], check=True)
        reader = gguf.GGUFReader(path)
        if any(reader.get_field(k) is None or reader.get_field(k).contents() != v
               for k, v in native.items()):
            raise ValueError(f"Native decision/tokenizer metadata changed: {kind}")
        if len(reader.tensors) != len(source.tensors):
            raise ValueError(f"Tensor count changed: {kind}")
        artifacts[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    receipt = {
        "format": "native-pplx-decider", "quantizations": args.quantizations,
        "imatrix": False, "output_head": "llama.cpp default mixed precision",
        "export_hashes": {p.name: digest(p) for p in sorted(export.iterdir()) if p.is_file()},
        "converter_sha256": digest(upstream / "convert_hf_to_gguf.py"),
        "quantize_binary_sha256": digest(binary) if args.quantizations else None,
        "artifacts": artifacts,
    }
    (output / "gguf-export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Exported:", ", ".join(artifacts), flush=True)


if __name__ == "__main__":
    main()
