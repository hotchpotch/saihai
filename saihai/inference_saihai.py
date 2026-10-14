"""Standalone saihai inference for Transformers AutoModel remote-code loading.

This file MUST run independently, copied unchanged into a model export. Never
import saihai, its training code, or sibling files. Required dependencies are
PyTorch, Transformers, tokenizers, huggingface-hub and (for device_map) accelerate.
Keep rendering and response mapping aligned with protocol.py / inference.py;
isolated-process tests verify independence and numerical parity.
"""

import json
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaConfig, PreTrainedModel, Qwen3_5Config
from transformers.utils.hub import cached_file


def canonical_json(value):
    """Match minja's six-significant-digit float JSON representation explicitly."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite input number")
        return format(value, ".6g")
    if isinstance(value, list):
        return "[" + ", ".join(canonical_json(x) for x in value) + "]"
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return (
            "{"
            + ", ".join(canonical_json(k) + ": " + canonical_json(v) for k, v in value.items())
            + "}"
        )
    raise ValueError("Input must contain JSON values with string object keys")


def render_input(inp, contract):
    kind = inp["type"]
    count = len(inp["options"])
    if kind not in ("choice", "noul", "score"):
        raise ValueError("Unsupported decision type")
    minimum = 1 if kind == "choice" else 2
    if count < minimum or (kind == "noul" and count != 2) or (kind == "score" and count > 10):
        raise ValueError("Invalid native candidate count")
    if kind == "noul" and [o["label"] for o in inp["options"]] != ["false", "true"]:
        raise ValueError("Native Noul requires false then true")
    options = []
    if len(inp["options"]) > len(contract["codes"]):
        raise ValueError("Native candidate capacity exceeded")
    for code, o in zip(contract["codes"], inp["options"]):
        item = {"code": code, "label": o["label"], "definition": o["description"]}
        if inp["type"] == "score":
            value = o["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Score values must be finite numbers")
            item["numeric_value"] = value
        options.append(item)
    return (
        contract["prompt_prefix"]
        + canonical_json(
            {
                "instruction": inp["instructions"],
                "state": inp["state"],
                "type": inp["type"],
                "options": options,
            }
        )
        + contract["prompt_suffix"]
    )


class DecisionInterface:
    """Shared renderer and llama.cpp-style System One response mapping."""

    def __init__(self, contract):
        self.contract_path = Path(contract) if isinstance(contract, (str, Path)) else None
        self.contract = (
            json.loads(self.contract_path.read_text()) if self.contract_path else contract
        )
        ids = self.contract["code_token_ids"]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(self.contract["codes"]):
            raise ValueError("Invalid candidate codebook")

    def render(self, inp):
        if self.contract.get("prompt_version") == "saihai-pplx-json-v1":
            return render_input(inp, self.contract)
        options = inp["options"]
        minimum = 1 if inp["type"] == "choice" else 2
        if not minimum <= len(options) <= len(self.contract["codes"]):
            raise ValueError("Candidate count exceeds contract bounds")
        if inp["type"] not in {"choice", "noul", "score"}:
            raise ValueError("Unsupported decision type")
        mapped = []
        for code, option in zip(self.contract["codes"], options):
            item = {"code": code, "label": option["label"], "definition": option["description"]}
            if "value" in option:
                value = option["value"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Score values must be finite numbers")
                item["numeric_value"] = value
            if inp["type"] == "score" and "value" not in option:
                raise ValueError("Score options require numeric values")
            mapped.append(item)
        payload = {
            "instruction": inp["instructions"],
            "state": inp["state"],
            "type": inp["type"],
            "options": mapped,
        }
        return (
            self.contract["prompt_prefix"]
            + json.dumps(payload, ensure_ascii=False, allow_nan=False)
            + self.contract["prompt_suffix"]
        )

    def systemone(self, request):
        """Accept TypeSafe-style criteria plus an exact-input options extension."""
        if request.get("images") or request.get("videos"):
            raise ValueError("This model is text-only")
        state, questions = request["state"], request["questions"]
        if "model" in request and request["model"] != self.contract["model_id"]:
            raise ValueError("Requested model does not match the loaded model")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 64:
            raise ValueError("Expected 1..64 named questions")
        answers, input_tokens, backend_output_tokens = {}, 0, 0
        for qid, q in questions.items():
            if not isinstance(qid, str) or not qid:
                raise ValueError("Question IDs must be nonempty strings")
            kind = q["type"]
            if "options" in q:
                options = q["options"]
                keys = [option["id"] for option in options]
            elif kind == "choice":
                criteria = q["criteria"]
                if not isinstance(criteria, dict):
                    raise ValueError("Choice criteria must be an ordered object")
                keys = list(criteria)
                options = [{"label": key, "description": criteria[key]} for key in keys]
            elif kind == "noul":
                criteria = q.get("criteria") or {}
                keys = ["false", "true"]
                options = [
                    {
                        "label": key,
                        "description": key if criteria.get(key) is None else criteria[key],
                    }
                    for key in keys
                ]
            elif kind == "score":
                criteria = q["criteria"]
                if not isinstance(criteria, list):
                    raise ValueError("Score criteria must be an array")
                keys = [str(i) for i in range(len(criteria))]
                options = [
                    {
                        "label": f"option_{i}",
                        "description": value["description"]
                        if isinstance(value, dict) and "value" in value
                        else value,
                        "value": value["value"]
                        if isinstance(value, dict) and "value" in value
                        else i,
                    }
                    for i, value in enumerate(criteria)
                ]
            else:
                raise ValueError("Unsupported decision type")
            if len(keys) != len(set(keys)) or any(not isinstance(key, str) for key in keys):
                raise ValueError("Option IDs must be unique strings")
            if kind == "noul" and set(keys) != {"false", "true"}:
                raise ValueError("Noul requires false and true options")
            result = self.predict(
                {
                    "state": state,
                    "type": kind,
                    "instructions": q.get("instructions", qid),
                    "options": options,
                }
            )
            ps = result["probabilities"]
            input_tokens += result["input_tokens"]
            backend_output_tokens += result.get("backend_output_tokens", 1)
            answer = {"type": kind}
            if kind == "noul":
                answer["noul"] = ps[keys.index("true")]
            else:
                answer["probabilities"] = dict(zip(keys, ps, strict=True))
                if kind == "choice":
                    mode = max(range(len(ps)), key=ps.__getitem__)
                    answer["choice"] = keys[mode]
                    answer["confidence"] = (
                        1.0
                        if len(ps) == 1
                        else max(0.0, (ps[mode] - 1 / len(ps)) / (1 - 1 / len(ps)))
                    )
                else:
                    mode = max(range(len(ps)), key=ps.__getitem__)
                    distance = sum(p * abs(i - mode) for i, p in enumerate(ps))
                    uniform_distance = sum(
                        abs(i - (len(ps) - 1) / 2) for i in range(len(ps))
                    ) / len(ps)
                    answer["confidence"] = max(0.0, 1 - distance / uniform_distance)
                    answer["score"] = sum(i * p for i, p in enumerate(ps))
                    answer["legend"] = {
                        key: option["description"]
                        for key, option in zip(keys, options, strict=True)
                    }
                    answer["expected_value"] = sum(
                        option["value"] * p for option, p in zip(options, ps, strict=True)
                    )
                    answer["numeric_values"] = dict(
                        zip(keys, [o["value"] for o in options], strict=True)
                    )
            answers[qid] = answer
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "backend_usage": {"output_tokens": backend_output_tokens},
            "model": self.contract["model_id"],
        }


class SaihaiModel(PreTrainedModel):
    """Typed decision wrapper; standard backbone weights retain their original names."""

    config_class = LlamaConfig
    _supports_sdpa = True

    def __init__(self, model, tokenizer, contract):
        super().__init__(model.config)
        self.model = model
        self.tokenizer = tokenizer
        self.interface = DecisionInterface(contract)
        self.interface.predict = self._predict_input
        ids = contract["code_token_ids"]
        if any(not isinstance(i, int) or not 0 <= i < model.lm_head.out_features for i in ids):
            raise ValueError("Candidate token IDs exceed model vocabulary")
        if contract["max_input_tokens"] > model.config.get_text_config().max_position_embeddings:
            raise ValueError("Contract context limit exceeds model positional capacity")
        self.token_ids = ids

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        revision="main",
        token=None,
        cache_dir=None,
        local_files_only=False,
        force_download=False,
        subfolder="",
        device=None,
        dtype=None,
        device_map=None,
        attn_implementation="sdpa",
        **kwargs,
    ):
        """Load a local export or Hub revision without installing the saihai package."""
        from transformers import AutoConfig, LlamaForCausalLM, Qwen3_5ForConditionalGeneration

        hub = dict(
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            force_download=force_download,
            subfolder=subfolder,
        )
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **hub)
        # AutoModel resolves config first. Reuse that commit for contract, tokenizer
        # and weights so a moving branch cannot mix artifacts from different releases.
        if getattr(config, "_commit_hash", None):
            hub["revision"] = config._commit_hash
        contract_path = cached_file(pretrained_model_name_or_path, "decision-inference.json", **hub)
        contract = json.loads(Path(contract_path).read_text())
        tokenizer_file = contract["tokenizer_file"]
        if Path(tokenizer_file).is_absolute() or ".." in Path(tokenizer_file).parts:
            raise ValueError("Tokenizer must be a file within the model export")
        tokenizer = Tokenizer.from_file(
            cached_file(pretrained_model_name_or_path, tokenizer_file, **hub)
        )
        if config.model_type == "llama":
            backbone_cls = LlamaForCausalLM
        elif config.model_type == "qwen3_5":
            backbone_cls = Qwen3_5ForConditionalGeneration
        else:
            raise ValueError(f"Unsupported backbone: {config.model_type}")
        if device is not None and device_map is not None:
            raise ValueError("Specify device or device_map, not both")
        if device is not None:
            device_map = {"": str(device)}
        if dtype is None:
            dtype = kwargs.pop("torch_dtype", "auto")
        for name in (
            "trust_remote_code",
            "_from_auto",
            "adapter_kwargs",
            "_commit_hash",
            "code_revision",
        ):
            kwargs.pop(name, None)
        model = backbone_cls.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **hub,
            **kwargs,
        ).eval()
        return cls(model, tokenizer, contract).eval()

    def forward(self, *args, **kwargs):
        """Delegate ordinary tensor forward calls to the exported backbone."""
        return self.model(*args, **kwargs)

    def _predict_input(self, inp):
        text = self.interface.render(inp)
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or len(ids) > self.interface.contract["max_input_tokens"]:
            raise ValueError("Input exceeds contract context limit; refusing truncation")
        inputs = torch.tensor([ids], device=self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            backbone = self.model.model
            if hasattr(backbone, "language_model"):
                backbone = backbone.language_model
            hidden = backbone(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False
            ).last_hidden_state[:, -1:, :]
            logits = self.model.lm_head(hidden)[0, 0, self.token_ids].float()
            probabilities = logits[: len(inp["options"])].softmax(-1).cpu().tolist()
        return {
            "probabilities": probabilities,
            "input_tokens": len(ids),
            "backend_output_tokens": 0,
        }

    def systemone(self, request):
        return self.interface.systemone(request)

    def predict(self, state, question):
        return self.systemone({"state": state, "questions": {"decision": question}})["answers"][
            "decision"
        ]

    def save_pretrained(self, *args, **kwargs):
        raise NotImplementedError("Export the original backbone with the saihai export helper")


class SaihaiQwenModel(SaihaiModel):
    """The same standalone decision interface for Qwen3.5 exports."""

    config_class = Qwen3_5Config
