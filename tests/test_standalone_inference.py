"""The distributed file MUST run without saihai, including via AutoModel."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from saihai import Saihai
from saihai.export_inference import export_inference


@pytest.mark.parametrize("protocol", [None, "saihai-pplx-json-v1"])
@pytest.mark.parametrize("backbone", ["llama", "qwen3_5"])
def test_isolated_automodel_and_direct_import_match_package(
    tiny_export, tmp_path, protocol, backbone
):
    directory, _, _ = tiny_export
    if backbone == "qwen3_5":
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        cfg = Qwen3_5Config(
            text_config=dict(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                layer_types=["full_attention"],
                max_position_embeddings=512,
            ),
            vision_config=dict(
                depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16
            ),
        )
        Qwen3_5ForConditionalGeneration(cfg).eval().save_pretrained(directory)
    contract_path = directory / "decision-inference.json"
    contract = json.loads(contract_path.read_text())
    if protocol:
        contract["prompt_version"] = protocol
        contract_path.write_text(json.dumps(contract))
    requests = [
        {
            "state": {"x": [0.123456789, "日本語"]},
            "questions": {
                "c": {"type": "choice", "criteria": {"z": "last", "a": "first"}},
                "n": {"type": "noul", "instructions": "True?"},
                "s": {
                    "type": "score",
                    "criteria": [
                        {"value": -2, "description": "low"},
                        {"value": 10, "description": "high"},
                    ],
                },
            },
        }
    ]
    reference = Saihai.from_pretrained(directory, device="cpu")
    expected = [reference.systemone(r) for r in requests]
    inp = {
        "state": {"number": 0.123456789},
        "type": "choice",
        "instructions": "pick",
        "options": [{"label": "a", "description": "日本語"}],
    }
    rendered = reference.backend.render(inp)
    export_inference(directory)
    assert (directory / "inference_saihai.py").read_bytes() == (
        Path(__file__).parents[1] / "saihai/inference_saihai.py"
    ).read_bytes()
    # A separate cwd, -I and an explicit import blocker ensure neither the
    # checkout nor an installed saihai can accidentally satisfy dependencies.
    script = r"""
import importlib.abc
import json
import sys
from pathlib import Path
class BlockSaihai(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "saihai" or fullname.startswith("saihai."):
            raise ImportError("saihai is deliberately unavailable")
sys.meta_path.insert(0, BlockSaihai())
from transformers import AutoModel, AutoModelForCausalLM
import torch
path = sys.argv[1]
requests = json.loads(sys.argv[2])
remote = AutoModel.from_pretrained(path, trust_remote_code=True, device="cpu", dtype=torch.float32)
sys.path.insert(0, path)
from inference_saihai import SaihaiModel
local = SaihaiModel.from_pretrained(path, device="cpu", dtype=torch.float32)
results = [remote.systemone(r) for r in requests]
assert results == [local.systemone(r) for r in requests]
assert remote.predict("test", {"type": "noul"})["type"] == "noul"
# The same export remains a normal CausalLM for converters and existing users.
if remote.config.model_type == "llama":
    assert AutoModelForCausalLM.from_pretrained(path).__class__.__name__ == "LlamaForCausalLM"
remote.interface.contract["max_input_tokens"] = 1
try:
    remote.predict("long input", {"type": "noul"})
except ValueError as exc:
    assert "refusing truncation" in str(exc)
else:
    raise AssertionError("overflow was silently accepted")
print(json.dumps({"results": results, "rendered": remote.interface.render(json.loads(sys.argv[3]))}))
"""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(directory), json.dumps(requests), json.dumps(inp)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.splitlines()[-1])
    assert output["rendered"] == rendered
    actual = output["results"]
    for ref, got in zip(expected, actual, strict=True):
        assert ref["usage"] == got["usage"]
        assert ref["backend_usage"] == got["backend_usage"]
        for key, answer in ref["answers"].items():
            for field, value in answer.items():
                if isinstance(value, float):
                    assert got["answers"][key][field] == pytest.approx(value, abs=1e-7)
                elif field == "probabilities":
                    assert got["answers"][key][field] == pytest.approx(value, abs=1e-7)
                else:
                    assert got["answers"][key][field] == value
