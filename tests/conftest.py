"""Small local model exports shared by package and isolated runtime tests."""

import json

import pytest


@pytest.fixture
def tiny_export(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
        )
    ).eval()
    model.save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    contract = {
        "model_id": "test-model",
        "codes": ["t11", "t23", "t41"],
        "code_token_ids": [11, 23, 41],
        "prompt_prefix": "t1 ",
        "prompt_suffix": " t2",
        "vocab_size": 64,
        "max_input_tokens": 512,
        "tokenizer_file": "tokenizer.json",
    }
    (tmp_path / "decision-inference.json").write_text(json.dumps(contract))
    return tmp_path, model, tokenizer
