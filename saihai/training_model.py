"""Text-only decision training with a FP32 head and BF16 LoRA."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch import nn

PROJECTIONS = {
    "attention": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ],
    "all": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "qv": ["q_proj", "v_proj"],
}


def enable_kernels(backbone, training):
    import kernels
    from transformers.integrations import hub_kernels

    raw = backbone.get_base_model() if hasattr(backbone, "get_base_model") else backbone
    # Torch 2.10 supports the FLA build; current convolution/rotary builds need 2.11.
    mapping = {
        **hub_kernels.get_kernel_mapping_transformers(),
        "causal_conv1d_fn": {},
        "causal_conv1d_update": {},
        "rotary_pos_emb": {},
    }
    with kernels.use_kernel_mapping(mapping, inherit_mapping=False):
        kernels.kernelize(
            raw, device="cuda", mode=kernels.Mode.TRAINING if training else kernels.Mode.INFERENCE
        )


class DecisionModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        last = attention_mask.sum(-1) - 1  # right padding, never truncate
        state = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return self.head(state.float())


def build_model(config, cache, training=True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, LlamaForCausalLM, Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU through CUDA_VISIBLE_DEVICES")
    mode = config["mode"]
    if mode not in {"lora", "qlora", "full", "baseline"}:
        raise ValueError("Unknown training mode")
    kwargs = {
        "revision": cache["base"]["revision"],
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
    }
    if mode == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    is_llama = cache["base"].get("model_type") == "llama"
    cls = LlamaForCausalLM if is_llama else Qwen3_5ForConditionalGeneration
    original = cls.from_pretrained(cache["base"]["model_id"], **kwargs)
    head = nn.Linear(
        original.config.hidden_size if is_llama else original.config.text_config.hidden_size,
        len(cache["codes"]),
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    with torch.no_grad():
        head.weight.copy_(original.lm_head.weight[cache["code_token_ids"]].float())
    backbone = original.model if is_llama else original.model.language_model
    del original
    gc.collect()
    torch.cuda.empty_cache()
    backbone.config.use_cache = False
    if mode == "qlora":
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
    if mode in {"lora", "qlora"}:
        requested = PROJECTIONS[config["projections"]]
        if is_llama:
            requested = [
                name for name in requested if not name.startswith("in_proj_") and name != "out_proj"
            ]
        actual = [
            (name, module)
            for name, module in backbone.named_modules()
            if name.rsplit(".", 1)[-1] in requested and isinstance(module, nn.Linear)
        ]
        found = {name.rsplit(".", 1)[-1] for name, _ in actual}
        if found != set(requested):
            raise ValueError(f"Missing projections: {set(requested) - found}")
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=config["rank"],
                lora_alpha=config["alpha"],
                lora_dropout=config.get("dropout", 0),
                target_modules=requested,
                bias="none",
            ),
        )
        projection_names = [name for name, _ in actual]
    else:
        projection_names = []
        backbone.requires_grad_(mode == "full")
        if config.get("freeze_embeddings", True):
            backbone.embed_tokens.requires_grad_(False)
    if training and mode != "baseline" and config.get("gradient_checkpointing", True):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if not is_llama:
        enable_kernels(backbone, training)
    model = DecisionModel(backbone, head)
    model.train(training)
    info = {
        "mode": mode,
        "actual_projection_modules": projection_names,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_trainable_parameters": sum(
            p.numel() for p in backbone.parameters() if p.requires_grad
        ),
        "head_parameters": head.weight.numel(),
        "text_only": True,
        "head_initialization": "original lm-head single-token code rows",
        "attention": "sdpa",
        "delta_rule": None if is_llama else "kernels-community/fla",
        "convolution": None if is_llama else "torch GPU",
        "model_type": "llama" if is_llama else "qwen3_5",
        "compute_dtype": "bfloat16",
        "head_dtype": "float32",
        "quantization": "nf4-double-quant" if mode == "qlora" else None,
    }
    return model, info


def save_checkpoint(model, config, cache, output):
    output = Path(output)
    output.mkdir(parents=True)
    if config["mode"] in {"lora", "qlora"}:
        model.backbone.save_pretrained(output / "adapter")
    elif config["mode"] == "full":
        model.backbone.save_pretrained(output / "backbone", safe_serialization=True)
    torch.save(model.head.state_dict(), output / "head.pt")
    (output / "runtime.json").write_text(json.dumps({"config": config, "cache": cache}, indent=2))


def load_checkpoint(path, training=False, qlora_bf16=False):
    from peft import PeftModel
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    path = Path(path)
    runtime = json.loads((path / "runtime.json").read_text())
    config, cache = runtime["config"], runtime["cache"]
    if qlora_bf16 and config["mode"] != "qlora":
        raise ValueError("BF16 QLoRA inference requires a QLoRA-trained checkpoint")
    # Instantiate the base without random new LoRA weights; retain matching quantization.
    build_config = dict(config)
    if config["mode"] == "lora" or (config["mode"] == "qlora" and qlora_bf16):
        build_config["mode"] = "baseline"
    model, info = build_model(build_config, cache, training)
    if config["mode"] == "qlora" and not qlora_bf16:
        model.backbone = model.backbone.unload()
    if config["mode"] in {"lora", "qlora"}:
        model.backbone = PeftModel.from_pretrained(
            model.backbone, path / "adapter", is_trainable=training
        )
    elif config["mode"] == "full":
        del model.backbone
        gc.collect()
        torch.cuda.empty_cache()
        from transformers import LlamaModel

        cls = LlamaModel if cache["base"].get("model_type") == "llama" else Qwen3_5TextModel
        model.backbone = cls.from_pretrained(
            path / "backbone", dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
        )
        if cache["base"].get("model_type") != "llama":
            enable_kernels(model.backbone, training)
    model.head.load_state_dict(torch.load(path / "head.pt", map_location="cuda", weights_only=True))
    model.eval()
    model.requires_grad_(False)
    info["mode"] = config["mode"]
    info["checkpoint_loaded"] = True
    if config["mode"] in {"lora", "qlora"}:
        trained_backbone = sum(
            p.numel() for name, p in model.backbone.named_parameters() if "lora_" in name
        )
    else:
        trained_backbone = sum(
            p.numel()
            for name, p in model.backbone.named_parameters()
            if not (config.get("freeze_embeddings", True) and name == "embed_tokens.weight")
        )
    info["trained_parameters"] = trained_backbone + model.head.weight.numel()
    info["trainable_parameters"] = 0
    info["backbone_trainable_parameters"] = 0
    info["qlora_adapter_on_bf16_backbone"] = config["mode"] == "qlora" and qlora_bf16
    info["trained_projection_group"] = config.get("projections")
    info["trained_lora_rank"] = config.get("rank") if config["mode"] in {"lora", "qlora"} else None
    return model, runtime, info


def expanded_head(head, initial_rows):
    if initial_rows.shape[0] < head.out_features or initial_rows.shape[1] != head.in_features:
        raise ValueError("Invalid head extension dimensions")
    extended = nn.Linear(
        head.in_features,
        initial_rows.shape[0],
        bias=False,
        device=head.weight.device,
        dtype=head.weight.dtype,
    )
    with torch.no_grad():
        extended.weight.copy_(initial_rows)
        extended.weight[: head.out_features].copy_(head.weight)
    extended.requires_grad_(False)
    return extended


def collate(rows, pad_token_id, device="cuda"):
    width = max(len(r["input_ids"]) for r in rows)
    count = max(len(r["probabilities"]) for r in rows)
    ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    target = torch.zeros((len(rows), count), dtype=torch.float32)
    options = torch.zeros_like(target, dtype=torch.bool)
    for i, r in enumerate(rows):
        n, k = len(r["input_ids"]), len(r["probabilities"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        attention[i, :n] = 1
        target[i, :k] = torch.tensor(r["probabilities"])
        options[i, :k] = True
    return {
        k: v.to(device)
        for k, v in {
            "input_ids": ids,
            "attention_mask": attention,
            "target": target,
            "options": options,
        }.items()
    }


def soft_ce(logits, target, options):
    masked = logits[:, : target.shape[1]].float().masked_fill(~options, -1e9)
    return -(target * masked.log_softmax(-1)).sum(-1).mean(), masked
