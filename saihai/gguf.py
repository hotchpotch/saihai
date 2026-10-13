"""Embed a saihai inference contract into GGUF for native /v1/systemone."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def native_metadata(contract, architecture, *, systemone_format="saihai"):
    """Build model-bound metadata without changing the learned prompt."""
    codes, ids = contract["codes"], contract["code_token_ids"]
    if not 2 <= len(codes) <= 256 or len(codes) != len(ids):
        raise ValueError("Expected 2..256 aligned labels")
    if len(set(ids)) != len(ids) or len(set(codes)) != len(codes):
        raise ValueError("Duplicate label or token ID")
    if any(not isinstance(c, str) or not c or len(c.encode()) > 32 for c in codes):
        raise ValueError("Invalid label text")
    if any(type(i) is not int or not 0 <= i < contract["vocab_size"] for i in ids):
        raise ValueError("Invalid label token ID")
    limit = contract["max_input_tokens"]
    if type(limit) is not int or not 0 < limit <= 1048576:
        raise ValueError("Invalid input token limit")
    temperature = contract["temperature"]
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    if contract["prompt_version"] not in (
        "saihai-json-instruction-first-v2",
        "saihai-pplx-json-v1",
    ):
        raise ValueError("Unsupported prompt protocol")
    if contract["prompt_version"] == "saihai-pplx-json-v1" and systemone_format != "pplx":
        raise ValueError("The native training protocol requires --systemone-format pplx")
    prefix = architecture + ".decision."
    if systemone_format in ("pplx-26", "pplx"):
        if systemone_format == "pplx" and (
            contract["prompt_version"] != "saihai-pplx-json-v1" or len(codes) > 255
        ):
            raise ValueError(
                "Full pplx format requires the native training protocol and at most 255 labels"
            )
        if codes[:26] != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            raise ValueError("The compatible profile requires the A..Z code prefix")
        capacity = 26 if systemone_format == "pplx-26" else len(codes)
        filename = (
            "systemone_pplx26.jinja" if systemone_format == "pplx-26" else "systemone_pplx.jinja"
        )
        template = (
            (Path(__file__).parent / "templates" / filename)
            .read_text()
            .replace("__CAPACITY__", str(capacity))
        )
        template = template.replace(
            "__PREFIX_EXPR__", json.dumps(contract["prompt_prefix"], ensure_ascii=False)
        )
        template = template.replace(
            "__SUFFIX_EXPR__", json.dumps(contract["prompt_suffix"], ensure_ascii=False)
        )
        return {
            prefix + "type": "pplx-decider",
            prefix + "max_input_tokens": limit,
            prefix + "compatibility.max_candidates": capacity,
            **{
                prefix + "temperature." + kind: float(temperature)
                for kind in ("choice", "noul", "score")
            },
            "tokenizer.chat_template.systemone": template,
        }
    if systemone_format != "saihai":
        raise ValueError("Unknown System One format")
    # A Jinja string literal keeps learned delimiters separate from template code.
    template = (
        "{{ " + json.dumps(contract["prompt_prefix"], ensure_ascii=False) + " }}"
        "{{ payload }}"
        "{{ " + json.dumps(contract["prompt_suffix"], ensure_ascii=False) + " }}"
    )
    return {
        prefix + "type": "saihai",
        prefix + "labels": json.dumps(
            [{"text": c, "token_id": i} for c, i in zip(codes, ids, strict=True)],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        prefix + "max_input_tokens": limit,
        **{
            prefix + "temperature." + kind: float(temperature)
            for kind in ("choice", "noul", "score")
        },
        "tokenizer.chat_template.systemone": template,
    }


def tensor_digest(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast("B")).hexdigest()


def embed_contract(source, destination, contract, gguf, *, systemone_format="saihai"):
    """Copy tensors unchanged, validate their hashes, and add native metadata."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or source.resolve() == destination.resolve():
        raise FileExistsError(destination)
    reader = gguf.GGUFReader(source)
    arch = reader.get_field("general.architecture").contents()
    metadata = native_metadata(contract, arch, systemone_format=systemone_format)
    tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    if len(tokens) != contract["vocab_size"]:
        raise ValueError("GGUF vocabulary size differs from contract")
    if any(
        tokens[i] not in (code, "\u2581" + code)
        for code, i in zip(contract["codes"], contract["code_token_ids"], strict=True)
    ):
        raise ValueError("GGUF label tokens differ from contract")
    writer = gguf.GGUFWriter(destination, arch, endianess=reader.endianess)
    writer.data_alignment = reader.alignment
    template_names = reader.get_field("tokenizer.chat_templates")
    names = template_names.contents() if template_names else []
    metadata["tokenizer.chat_templates"] = list(dict.fromkeys([*names, "systemone"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        for field in reader.fields.values():
            if (
                field.name.startswith("GGUF.")
                or field.name == "general.architecture"
                or field.name in metadata
            ):
                continue
            writer.add_key_value(
                field.name,
                field.contents(),
                field.types[0],
                field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None,
            )
        for key, value in metadata.items():
            if isinstance(value, str):
                writer.add_string(key, value)
            elif isinstance(value, list):
                writer.add_array(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            else:
                writer.add_uint32(key, value)
        for tensor in reader.tensors:
            writer.add_tensor_info(
                tensor.name,
                tensor.data.shape,
                tensor.data.dtype,
                tensor.data.nbytes,
                tensor.tensor_type,
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for tensor in reader.tensors:
            writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        writer.close()
        output = gguf.GGUFReader(destination)
        if len(reader.tensors) != len(output.tensors):
            raise ValueError("Tensor count changed")
        hashes = {}
        for original, copied in zip(reader.tensors, output.tensors, strict=True):
            original_hash, copied_hash = tensor_digest(original), tensor_digest(copied)
            if (
                original.name != copied.name
                or original.tensor_type != copied.tensor_type
                or original.shape.tolist() != copied.shape.tolist()
                or original_hash != copied_hash
            ):
                raise ValueError("Tensor changed: " + original.name)
            hashes[original.name] = copied_hash
        return {"tensor_count": len(hashes), "tensor_hashes": hashes, "metadata": metadata}
    except BaseException:
        writer.close()
        destination.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--systemone-format", choices=["pplx-26", "pplx", "saihai"], default="pplx-26"
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_cpp.resolve() / "gguf-py"))
    import gguf

    contract = json.loads(args.contract.read_text())
    receipt = embed_contract(
        args.input, args.output, contract, gguf, systemone_format=args.systemone_format
    )
    args.output.with_suffix(".native.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Wrote {args.output}; verified {receipt['tensor_count']} unchanged tensors")


if __name__ == "__main__":
    main()
