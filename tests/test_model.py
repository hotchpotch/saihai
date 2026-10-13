"""Verify portable loading, candidate masking, and backend-independent answers."""

import json

import pytest

from saihai import Saihai
from saihai.inference import LlamaDecisionClient


@pytest.fixture
def tiny_export(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
    )).eval()
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    contract = {"model_id": "test-model", "codes": ["t11", "t23", "t41"],
                "code_token_ids": [11, 23, 41], "prompt_prefix": "t1 ", "prompt_suffix": " t2",
                "vocab_size": 64, "max_input_tokens": 512, "tokenizer_file": "tokenizer.json"}
    (tmp_path / "decision-inference.json").write_text(json.dumps(contract))
    return tmp_path, model, tokenizer


def test_torch_readout_matches_export_and_normalizes_only_candidates(tiny_export):
    import torch

    path, reference, tokenizer = tiny_export
    loaded = Saihai.from_pretrained(path, device="cpu")
    inp = {"state": "t3", "instructions": "t4", "type": "choice",
           "options": [{"label": "a", "description": "t5"}, {"label": "b", "description": "t6"}]}
    text = loaded.backend.render(inp)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    with torch.inference_mode():
        logits = reference(torch.tensor([ids])).logits[0, -1, [11, 23]]
    expected = logits.softmax(-1).tolist()
    actual = loaded.backend.predict(inp)
    assert actual["probabilities"] == pytest.approx(expected, abs=1e-7)
    assert sum(actual["probabilities"]) == pytest.approx(1)
    assert actual["backend_output_tokens"] == 0
    assert actual["input_tokens"] == len(ids)


def test_systemone_contract_matches_between_backends(tiny_export):
    path, _, _ = tiny_export
    torch_model = Saihai.from_pretrained(path, device="cpu")
    llama = LlamaDecisionClient("http://example.invalid", path / "decision-inference.json")
    llama.predict = torch_model.backend.predict
    request = {"model": "test-model", "state": {"x": 1}, "questions": {
        "choice": {"type": "choice", "criteria": {"z": "Z", "a": "A"}},
        "true": {"type": "noul", "instructions": "True?"},
        "score": {"type": "score", "criteria": [
            {"value": -2, "description": "low"}, {"value": 10, "description": "high"}]},
    }}
    assert torch_model.systemone(request) == llama.systemone(request)
    assert torch_model.systemone(request)["backend_usage"]["output_tokens"] == 0
    assert torch_model.predict("x", {"type": "noul", "instructions": "True?"})["type"] == "noul"


def test_native_tokenizer_mode_does_not_require_external_file(tmp_path):
    path = tmp_path / "decision-inference.json"
    path.write_text(json.dumps({"model_id": "x", "codes": ["A", "B"], "code_token_ids": [11, 23],
                               "tokenizer_file": "absent.json", "prompt_prefix": "", "prompt_suffix": "",
                               "vocab_size": 64, "max_input_tokens": 10}))
    client = LlamaDecisionClient("http://example.invalid", path, native_tokenizer=True)
    assert client.tokenizer is None
    with pytest.raises(FileNotFoundError):
        LlamaDecisionClient("http://example.invalid", path)


def test_model_selector_and_overflow_are_rejected(tiny_export):
    path, _, _ = tiny_export
    loaded = Saihai.from_pretrained(path, device="cpu")
    with pytest.raises(ValueError, match="does not match"):
        loaded.systemone({"model": "wrong", "state": "x", "questions": {"a": {"type": "noul"}}})
    loaded.backend.contract["max_input_tokens"] = 1
    with pytest.raises(ValueError, match="refusing truncation"):
        loaded.predict("x", {"type": "noul", "instructions": "True?"})


def test_invalid_backend_fails_before_download():
    with pytest.raises(ValueError, match="backend must"):
        Saihai.from_pretrained("unavailable/model", backend="invalid")
    with pytest.raises(ValueError, match="backend_url"):
        Saihai.from_pretrained("unavailable/model", backend="llama.cpp")
