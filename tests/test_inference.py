"""Check candidate masking, typed mapping and preservation of numeric levels."""

import json

import pytest

from saihai.inference import LlamaDecisionClient


@pytest.fixture
def client():
    return LlamaDecisionClient("http://backend", {"codes": ["A", "B", "C"], "code_token_ids": [11, 23, 41],
        "prompt_prefix": "prefix", "prompt_suffix": "suffix", "max_input_tokens": 100,
        "model_id": "example", "vocab_size": 64})


def test_readout_masks_exact_token_ids_and_disables_top_k(client):
    calls = []
    def post(endpoint, body):
        calls.append((endpoint, body))
        if endpoint == "/tokenize":
            return {"tokens": [1, 2, 3]}
        return {"completion_probabilities": [{"top_probs": [{"id": 23, "prob": 0.3}, {"id": 11, "prob": 0.7}]}]}
    client._post = post
    result = client.predict({"state": {"a": None}, "instructions": "choose", "type": "choice",
                             "options": [{"label": "first", "description": {"x": 1}}, {"label": "second", "description": "y"}]})
    assert result["probabilities"] == [0.7, 0.3]
    assert calls[1][1]["grammar"] == "root ::= <[11]> | <[23]>"
    assert calls[1][1]["samplers"] == ["temperature"]
    assert calls[1][1]["logit_bias"] == [[i, False] for i in range(64) if i not in {11, 23}]
    assert calls[0][1]["add_special"] is False
    payload = json.loads(calls[0][1]["content"][6:-6])
    assert payload["state"] == {"a": None}
    assert payload["options"][0]["definition"] == {"x": 1}


def test_rejects_overflow_before_completion(client):
    client._post = lambda endpoint, body: {"tokens": list(range(101))}
    with pytest.raises(ValueError, match="refusing truncation"):
        client.predict({"state": "x", "instructions": "q", "type": "noul",
                        "options": [{"label": "false", "description": "no"}, {"label": "true", "description": "yes"}]})


def test_typed_mapping_preserves_exact_order_and_score_values(client):
    inputs = []
    def predict(inp):
        inputs.append(inp)
        return {"probabilities": [0.25, 0.75], "input_tokens": 10}
    client.predict = predict
    request = {"state": {"a": [1]}, "questions": {
        "c": {"type": "choice", "instructions": "pick", "criteria": {"z": "Z", "a": "A"}},
        "n": {"type": "noul", "instructions": "true?", "options": [
            {"id": "true", "label": "true", "description": "authored yes"},
            {"id": "false", "label": "false", "description": "authored no"}]},
        "s": {"type": "score", "instructions": "rate", "criteria": [
            {"value": -2, "description": "low"}, {"value": 10, "description": "high"}]}}}
    result = client.systemone(request)
    assert result["answers"]["c"]["choice"] == "a"
    assert result["answers"]["c"]["confidence"] == 0.5
    assert result["answers"]["n"]["noul"] == 0.25
    assert result["answers"]["s"]["score"] == 0.75
    assert result["answers"]["s"]["expected_value"] == 7
    assert result["answers"]["s"]["confidence"] == 0.5
    assert inputs[1]["options"][0]["description"] == "authored yes"
    assert result["usage"] == {"input_tokens": 30, "output_tokens": 0}
    assert result["backend_usage"]["output_tokens"] == 3


def test_missing_candidate_fails_without_uniform_fallback(client):
    client._post = lambda endpoint, body: ({"tokens": [1]} if endpoint == "/tokenize" else
        {"completion_probabilities": [{"top_probs": [{"id": 11, "prob": 1.0}]}]})
    with pytest.raises(ValueError, match="omitted"):
        client.predict({"state": "x", "instructions": "q", "type": "choice",
                        "options": [{"label": "a", "description": "a"}, {"label": "b", "description": "b"}]})


def test_invalid_values_and_codebook_fail(client):
    with pytest.raises(ValueError, match="numeric values"):
        client.render({"state": "x", "instructions": "q", "type": "score",
                       "options": [{"label": "a", "description": "a"}, {"label": "b", "description": "b"}]})
    with pytest.raises(ValueError, match="codebook"):
        LlamaDecisionClient("http://backend", {"codes": ["A", "B"], "code_token_ids": [1, 1]})
