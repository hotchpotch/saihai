import json
from copy import deepcopy

import pytest

from saihai.gguf import embed_contract, native_metadata


@pytest.fixture
def contract():
    return {
        "codes": ["A", "B", "C"], "code_token_ids": [0, 1, 2], "vocab_size": 3,
        "max_input_tokens": 8192, "temperature": 1.0,
        "prompt_version": "saihai-json-instruction-first-v2",
        "prompt_prefix": '日本語\n"prefix"', "prompt_suffix": "<|message|>",
    }


def test_native_metadata_is_model_bound(contract):
    metadata = native_metadata(contract, "llama")
    assert metadata["llama.decision.type"] == "saihai"
    assert metadata["llama.decision.max_input_tokens"] == 8192
    assert json.loads(metadata["llama.decision.labels"])[2] == {"text": "C", "token_id": 2}
    assert "{{ payload }}" in metadata["tokenizer.chat_template.systemone"]
    assert metadata["llama.decision.temperature.score"] == 1.0


def test_compatible_profile_uses_existing_type_and_bounds(contract):
    contract["codes"] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    contract["code_token_ids"] = list(range(26))
    contract["vocab_size"] = 26
    metadata = native_metadata(contract, "llama", systemone_format="pplx-26")
    assert metadata["llama.decision.type"] == "pplx-decider"
    assert metadata["llama.decision.compatibility.max_candidates"] == 26
    assert "llama.decision.labels" not in metadata
    assert "at most 26 candidates" in metadata["tokenizer.chat_template.systemone"]
    assert "__PREFIX_EXPR__" not in metadata["tokenizer.chat_template.systemone"]
    contract["codes"][0] = "!"
    with pytest.raises(ValueError, match="A..Z"):
        native_metadata(contract, "llama", systemone_format="pplx-26")


@pytest.mark.parametrize("field,value", [
    ("code_token_ids", [0, 0, 2]), ("code_token_ids", [0, 1, 3]),
    ("code_token_ids", [False, 1, 2]), ("codes", ["A", "A", "C"]),
    ("max_input_tokens", 0), ("max_input_tokens", True),
    ("temperature", float("nan")), ("temperature", 0),
    ("prompt_version", "unknown"),
])
def test_invalid_contract_rejected(contract, field, value):
    contract[field] = value
    with pytest.raises(ValueError):
        native_metadata(contract, "llama")


def test_embedding_preserves_quantized_tensors_and_templates(tmp_path, contract):
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    source, output = tmp_path / "source.gguf", tmp_path / "native.gguf"
    writer = gguf.GGUFWriter(source, "llama")
    writer.data_alignment = 64
    writer.add_uint32("general.alignment", 64)
    writer.add_token_list(["\u2581A", "\u2581B", "\u2581C"])
    writer.add_chat_template([
        {"name": "default", "template": "default template"},
        {"name": "other", "template": "other template"},
    ])
    writer.add_tensor("dense.weight", np.arange(32, dtype=np.float32).reshape(2, 16))
    quantized = gguf.quants.quantize(np.arange(32, dtype=np.float32).reshape(1, 32), gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor("quantized.weight", quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    receipt = embed_contract(source, output, contract, gguf)
    assert receipt["tensor_count"] == 2
    reader = gguf.GGUFReader(output)
    assert reader.alignment == 64
    assert reader.get_field("tokenizer.chat_template").contents() == "default template"
    assert reader.get_field("tokenizer.chat_template.other").contents() == "other template"
    assert set(reader.get_field("tokenizer.chat_templates").contents()) == {"other", "systemone"}
    assert reader.tensors[1].tensor_type == gguf.GGMLQuantizationType.Q8_0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        embed_contract(source, output, contract, gguf)
    assert output.read_bytes() == before
    incompatible = deepcopy(contract)
    incompatible["codes"][0] = "X"
    with pytest.raises(ValueError, match="label tokens"):
        embed_contract(source, tmp_path / "wrong.gguf", incompatible, gguf)
    assert not (tmp_path / "wrong.gguf").exists()
