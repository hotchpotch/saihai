"""Training contract tests: soft labels, canonical inputs and native readout."""

import math

import pytest
import torch

from saihai.training import (
    canonical_json,
    canonicalize_record,
    collate,
    learning_rate_factor,
    load_config,
    pplx_codebook,
    select_records,
    soft_ce,
)


class FakeTokenizer:
    def encode(self, code, add_special_tokens=False):
        if code == "AH":
            return [1, 2]
        return [sum((i + 1) * ord(c) * 100**i for i, c in enumerate(code))]


def record(key, kind="choice", parent=None):
    options = [{"label": "x", "description": "X"}, {"label": "y", "description": "Y"}]
    if kind == "noul":
        options = [{"label": "true", "description": "yes"}, {"label": "false", "description": "no"}]
    if kind == "score":
        options = [
            {"label": "low", "description": "L", "value": 0.0},
            {"label": "high", "description": "H", "value": 5.0},
        ]
    return {
        "id": key,
        "parent_id": parent or key,
        "provenance": {"dataset": "source"},
        "input": {"state": "状態", "instructions": "判定", "type": kind, "options": options},
        "target": {"probabilities": [0.2, 0.8]},
    }


def test_codebook_matches_native_skip_and_capacity():
    codes, ids = pplx_codebook(FakeTokenizer())
    assert len(codes) == len(ids) == 255
    assert codes[:28] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB"]
    assert "AH" not in codes
    assert len(set(ids)) == 255


def test_noul_reorders_soft_targets_and_score_labels_without_mutating_source():
    old = record("n", "noul")
    new = canonicalize_record(old)
    assert [o["label"] for o in new["input"]["options"]] == ["false", "true"]
    assert new["target"]["probabilities"] == [0.8, 0.2]
    assert old["target"]["probabilities"] == [0.2, 0.8]
    assert [o["label"] for o in canonicalize_record(record("s", "score"))["input"]["options"]] == [
        "option_0",
        "option_1",
    ]


@pytest.mark.parametrize("ps", [[0.1, 0.2], [-0.1, 1.1], [math.nan, 1], [1]])
def test_invalid_targets_rejected(ps):
    row = record("bad")
    row["target"]["probabilities"] = ps
    with pytest.raises(ValueError):
        canonicalize_record(row)


def test_parent_sampling_deterministic_and_no_validation_selection():
    rows = [record(str(i), parent=str(i // 2)) for i in range(40)]
    chosen = select_records(rows, fraction=0.1, seed=42)
    assert chosen == select_records(rows, fraction=0.1, seed=42)
    assert len(chosen) == 4
    assert {r["parent_id"] for r in chosen} == {r["parent_id"] for r in chosen[::2]}
    assert all(
        r["input"]["type"] in {"noul", "choice", "score"}
        for r in select_records(
            [record("n", "noul"), record("c"), record("s", "score")], smoke=True
        )
    )


def test_masked_soft_ce_gradient_ignores_unused_head_rows():
    logits = torch.tensor([[1.0, 2.0, 100.0], [2.0, 1.0, 3.0]], requires_grad=True)
    target = torch.tensor([[0.25, 0.75, 0.0], [0.1, 0.2, 0.7]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss, _ = soft_ce(logits, target, mask)
    expected = (
        -(target[0, :2] * logits[0, :2].log_softmax(-1)).sum() / 2
        - (target[1] * logits[1].log_softmax(-1)).sum() / 2
    )
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad[0, 2] == 0
    batch = collate(
        [
            {"input_ids": [1, 2], "probabilities": [0.2, 0.8]},
            {"input_ids": [3], "probabilities": [1.0]},
        ],
        0,
        device="cpu",
    )
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_canonical_json_matches_native_number_policy():
    assert (
        canonical_json({"日本語": [0.0, 0.25, 1.23456789, True, None]})
        == '{"日本語": [0, 0.25, 1.23457, true, null]}'
    )
    with pytest.raises(ValueError):
        canonical_json(float("inf"))


def test_yaml_recipes_and_overrides():
    from pathlib import Path

    for name in ("qwen35-0.8b", "qwen35-4b", "qwen35-9b", "llmjp41-8b"):
        c = load_config(Path("configs") / f"{name}.yaml")
        assert c["dataset"]["repo_id"] == "hotchpotch/saihai-v1-dataset"
        assert c["dataset"]["split"] == "train"
        assert c["training"]["batch_size"] * c["training"]["gradient_accumulation"] == 16
        assert c["training"]["optimizer"] == "adamw"
        assert c["evaluation"]["category"] == "japanese-v1"
    assert learning_rate_factor(0, 100, 0.05) == pytest.approx(0.2)
    assert learning_rate_factor(99, 100, 0.05) >= 0.2


def test_tied_base_export_preserves_embeddings_and_reloads_decision_rows(tmp_path):
    from types import SimpleNamespace

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from saihai import Saihai
    from saihai.training import export_model

    base = tmp_path / "base"
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    original = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    original.save_pretrained(base)
    embedding = original.model.embed_tokens.weight.detach().clone()
    head = torch.nn.Linear(16, 26, bias=False)
    with torch.no_grad():
        head.weight.fill_(0.125)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({**{chr(65 + i): i for i in range(26)}, "unk": 26}, unk_token="unk")
        )
    )
    contract = {
        "model_id": "tiny",
        "codes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "code_token_ids": list(range(26)),
        "prompt_prefix": "",
        "prompt_suffix": "",
        "prompt_version": "saihai-pplx-json-v1",
        "max_input_tokens": 4096,
        "tokenizer_file": "tokenizer.json",
    }
    training = SimpleNamespace(backbone=original.model, head=head, eval=lambda: None)
    export = tmp_path / "export"
    export_model(
        training,
        tokenizer,
        {"model": {"model_type": "llama", "model_id": str(base), "revision": "0" * 40}},
        contract,
        export,
    )
    restored = LlamaForCausalLM.from_pretrained(export)
    assert torch.equal(
        restored.model.embed_tokens.weight.float(), embedding.to(torch.bfloat16).float()
    )
    assert torch.equal(restored.lm_head.weight[:26].float(), head.weight.float())
    assert restored.lm_head.weight.data_ptr() != restored.model.embed_tokens.weight.data_ptr()
    loaded = Saihai.from_pretrained(export, device="cpu")
    result = loaded.systemone(
        {
            "state": "A",
            "questions": {
                "q": {"type": "choice", "instructions": "B", "criteria": {"x": "C", "y": "D"}}
            },
        }
    )
    assert result["answers"]["q"]["probabilities"] == pytest.approx({"x": 0.5, "y": 0.5})


def test_native_template_and_python_renderer_share_number_and_option_format():
    from types import SimpleNamespace

    from jinja2 import Environment

    from saihai.gguf import native_metadata
    from saihai.training import render_input

    contract = {
        "codes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB"],
        "code_token_ids": list(range(28)),
        "vocab_size": 28,
        "temperature": 1.0,
        "max_input_tokens": 8192,
        "prompt_prefix": "<P>",
        "prompt_suffix": "<S>",
        "prompt_version": "saihai-pplx-json-v1",
    }
    template = native_metadata(contract, "llama", systemone_format="pplx")[
        "tokenizer.chat_template.systemone"
    ]
    env = Environment()
    env.filters["tojson"] = lambda v, **kw: canonical_json(v)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    for kind in ("noul", "choice", "score"):
        row = canonicalize_record(record(kind, kind))
        inp = row["input"]
        inp["state"] = {"日本語": '改行\n引用"', "x": [1.23456789, 0.0, True, None]}
        options = []
        for code, o in zip(contract["codes"], inp["options"]):
            desc = (
                {"description": o["description"], "value": o["value"]}
                if kind == "score"
                else o["description"]
            )
            options.append(SimpleNamespace(label=code, key=o["label"], description=desc))
        native = env.from_string(template).render(
            instructions=inp["instructions"], state=inp["state"], type=kind, options=options
        )
        assert native == render_input(inp, contract)


def test_source_probability_rounding_is_preserved():
    row = record("rounded")
    row["target"]["probabilities"] = [0.333333, 0.666666]
    assert canonicalize_record(row)["target"]["probabilities"] == row["target"]["probabilities"]


def test_cli_passes_fraction_and_batch_overrides_without_gpu_work(monkeypatch, tmp_path):
    import subprocess
    import sys

    from saihai import training

    seen = {}
    monkeypatch.setattr(
        training,
        "train",
        lambda config, output, **kw: seen.update(config=config, output=output, options=kw),
    )
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kw: seen.update(command=command, environment=kw)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/llmjp41-8b.yaml",
            "--output",
            str(tmp_path / "run"),
            "--fraction",
            "0.1",
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "16",
            "--s1mb-root",
            str(tmp_path / "s1mb"),
            "--eval-limit",
            "2",
        ],
    )
    training.main()
    assert seen["options"]["fraction"] == 0.1
    assert seen["config"]["training"]["batch_size"] == 1
    assert seen["config"]["training"]["gradient_accumulation"] == 16
    assert "saihai.s1mb_eval" in seen["command"]
    assert str((tmp_path / "run" / "s1mb-summary.json").resolve()) in seen["command"]
    assert seen["environment"]["cwd"] == tmp_path / "s1mb" / "evaluator"
    assert "--data-revision" in seen["command"]
    assert seen["command"][-1] == seen["config"]["evaluation"]["revision"]


def test_new_protocol_rejects_non_native_candidate_shapes():
    from saihai.protocol import render_input

    contract = {
        "codes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "prompt_prefix": "",
        "prompt_suffix": "",
    }
    row = record("n", "noul")
    with pytest.raises(ValueError, match="false then true"):
        render_input(row["input"], contract)
    row = record("s", "score")
    row["input"]["options"] *= 6
    with pytest.raises(ValueError, match="candidate count"):
        render_input(row["input"], contract)


def test_native_protocol_cannot_be_exported_with_legacy_template():
    from saihai.gguf import native_metadata

    contract = {
        "codes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "code_token_ids": list(range(26)),
        "vocab_size": 26,
        "temperature": 1.0,
        "max_input_tokens": 8192,
        "prompt_prefix": "",
        "prompt_suffix": "",
        "prompt_version": "saihai-pplx-json-v1",
    }
    with pytest.raises(ValueError, match="requires --systemone-format pplx"):
        native_metadata(contract, "llama", systemone_format="pplx-26")


def test_native_noul_null_descriptions_use_default_labels():
    from saihai.inference import DecisionInterface

    class Capture(DecisionInterface):
        def predict(self, inp):
            self.inp = inp
            return {"probabilities": [0.5, 0.5], "input_tokens": 1}

    model = Capture(
        {
            "model_id": "test",
            "codes": ["A", "B"],
            "code_token_ids": [0, 1],
            "prompt_version": "saihai-pplx-json-v1",
        }
    )
    model.systemone(
        {
            "state": "x",
            "questions": {"q": {"type": "noul", "criteria": {"false": None, "true": None}}},
        }
    )
    assert [o["description"] for o in model.inp["options"]] == ["false", "true"]


def test_evaluation_run_ids_distinguish_campaigns(tmp_path):
    from saihai.training import evaluation_run_id

    first = tmp_path / "baseline" / "smoke" / "llmjp41-8b"
    second = tmp_path / "holdout" / "smoke" / "llmjp41-8b"
    assert evaluation_run_id(first) != evaluation_run_id(second)
    assert evaluation_run_id(first) == evaluation_run_id(first)
