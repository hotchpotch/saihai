import pytest

from saihai.context import extend_contract


def test_extend_inference_only_and_preserve_original_training_limit():
    source = {"max_input_tokens": 8192, "prompt_prefix": "same", "codes": ["A", "B"]}
    result = extend_contract(source, {"max_position_embeddings": 65536}, 65536)
    assert source["max_input_tokens"] == 8192
    assert result["max_input_tokens"] == 65536
    assert result["training_max_input_tokens"] == 8192
    assert result["prompt_prefix"] == source["prompt_prefix"]
    assert (
        extend_contract(result, {"max_position_embeddings": 65536}, 32768)[
            "training_max_input_tokens"
        ]
        == 8192
    )


@pytest.mark.parametrize("limit", [True, 0, 65537])
def test_reject_context_beyond_model_capacity(limit):
    with pytest.raises(ValueError):
        extend_contract({"max_input_tokens": 8192}, {"max_position_embeddings": 65536}, limit)


def test_llmjp_export_contract_separates_train_and_inference():
    from saihai.training import load_config, prompt_contract

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "prefix" + messages[-1]["content"] + "suffix"

    config = load_config("configs/llmjp41-8b.yaml")
    contract = prompt_contract(Tokenizer(), config, ["A", "B"], [0, 1])
    assert contract["max_input_tokens"] == 65536
    assert contract["training_max_input_tokens"] == 8192
    assert config["training"]["max_length"] == 8192
