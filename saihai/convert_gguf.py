"""Convert a trained HF export with upstream llama.cpp and embed native metadata."""

import argparse
import json
import runpy
import sys
from pathlib import Path


def official_unigram_metadata(export, vocab_gguf, gguf):
    tokenizer = json.loads((export / "tokenizer.json").read_text())
    if tokenizer["model"]["type"] != "Unigram":
        raise ValueError("Reference vocabulary is supported only for Unigram tokenizers")
    reader = gguf.GGUFReader(vocab_gguf)
    metadata = {
        field.name: field.contents()
        for field in reader.fields.values()
        if field.name.startswith("tokenizer.ggml.")
    }
    vocab = tokenizer["model"]["vocab"]
    if metadata["tokenizer.ggml.tokens"] != [token for token, _ in vocab]:
        raise ValueError("Reference vocabulary token order differs from export")
    if any(
        abs(score - original) > 1e-5
        for score, (_, original), kind in zip(
            metadata["tokenizer.ggml.scores"],
            vocab,
            metadata["tokenizer.ggml.token_type"],
            strict=True,
        )
        if kind == 1
    ):
        raise ValueError("Reference vocabulary scores differ from export")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--vocab-gguf", type=Path, help="Official base-model GGUF vocabulary reference for LLM-jp"
    )
    args = parser.parse_args()
    upstream = args.llama_cpp.resolve()
    export = args.export.resolve()
    output = args.output.resolve()
    raw = output.with_suffix(".raw.gguf")
    if output.exists() or raw.exists():
        raise FileExistsError("Choose a fresh GGUF output")
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(upstream / "gguf-py"), str(upstream)]
    import gguf

    if args.vocab_gguf:
        from conversion.llama import LlamaModel

        metadata = official_unigram_metadata(export, args.vocab_gguf, gguf)
        reference = gguf.GGUFReader(args.vocab_gguf)
        field_types = {key: reference.get_field(key).types for key in metadata}

        def set_vocab(model):
            for key, value in metadata.items():
                types = field_types[key]
                model.gguf_writer.add_key_value(
                    key,
                    value,
                    types[0],
                    types[-1] if types[0] == gguf.GGUFValueType.ARRAY else None,
                )
            model.gguf_writer.add_chat_template((export / "chat_template.jinja").read_text())

        LlamaModel.set_vocab = set_vocab
    sys.argv = [
        str(upstream / "convert_hf_to_gguf.py"),
        str(export),
        "--outfile",
        str(raw),
        "--outtype",
        "bf16",
    ]
    runpy.run_path(str(upstream / "convert_hf_to_gguf.py"), run_name="__main__")
    from .gguf import embed_contract

    contract = json.loads((export / "decision-inference.json").read_text())
    receipt = embed_contract(raw, output, contract, gguf, systemone_format="pplx")
    output.with_suffix(".native.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print("Converted native pplx-decider GGUF:", output, flush=True)


if __name__ == "__main__":
    main()
