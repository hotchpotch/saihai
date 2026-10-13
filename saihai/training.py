"""YAML-driven, reproducible soft-target decision training and portable export."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import yaml

from .protocol import canonical_json, render_input
from .training_filter import filter_training_records, validate_filter
from .training_model import build_model, collate, save_checkpoint, soft_ce

SYSTEM = "Decide from the supplied state, instructions and option definitions. Return only the option code. Do not reason aloud."


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {
        "model",
        "dataset",
        "training",
        "evaluation",
        "prompt",
    }:
        raise ValueError("Expected model, dataset, training, evaluation and prompt sections")
    model, data, train = config["model"], config["dataset"], config["training"]
    validate_filter(data.get("train_filter", {}))
    if train.get("compute_dtype") != "bfloat16" or train.get("head_dtype") != "float32":
        raise ValueError("Training requires a BF16 backbone and FP32 decision head")
    if config["prompt"].get("serialization") != "minja-json-6-significant-digits":
        raise ValueError("Unsupported prompt serialization")
    if not model.get("model_id") or len(model.get("revision", "")) != 40:
        raise ValueError("Pin the model to a full Hub revision")
    if data.get("split") != "train" or (
        data.get("revision") != "latest" and len(data.get("revision", "")) != 40
    ):
        raise ValueError("Use the train split and a full dataset revision or latest")
    for key in (
        "batch_size",
        "gradient_accumulation",
        "epochs",
        "max_length",
        "smoke_steps",
        "smoke_examples",
        "validation_per_subset",
        "validation_runs",
    ):
        if isinstance(train[key], bool) or not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError("Invalid training option: " + key)
    for key in ("learning_rate", "head_learning_rate", "max_grad_norm"):
        if not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError("Invalid training option: " + key)
    if train["optimizer"] not in ("adamw", "adafactor") or train["mode"] != "lora":
        raise ValueError("Use AdamW or Adafactor with BF16 LoRA")
    if model.get("model_type") not in ("llama", "qwen3_5"):
        raise ValueError("Unsupported base-model architecture")
    for key in ("rank", "alpha", "cpu_threads", "log_every"):
        if isinstance(train[key], bool) or not isinstance(train[key], int) or train[key] <= 0:
            raise ValueError("Invalid training option: " + key)
    if (
        not 0 <= train["warmup_ratio"] < 1
        or not 0 <= train["dropout"] < 1
        or train["weight_decay"] < 0
    ):
        raise ValueError("Invalid warmup, dropout or weight decay")
    if config["evaluation"]["category"] != "japanese-v1":
        raise ValueError("The configured post-training evaluation must be japanese-v1")
    limit = config["evaluation"].get("max_input_tokens", train["max_length"])
    if type(limit) is not int or limit < train["max_length"] or limit > 1048576:
        raise ValueError("Invalid inference max_input_tokens")
    return config


def pplx_codebook(tokenizer):
    """Exactly mirror native A..Z, AA..ZZ single-token selection, capped at 255."""
    codes, ids = [], []
    candidates = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [
        a + b for a in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]
    for code in candidates:
        encoded = tokenizer.encode(code, add_special_tokens=False)
        if len(encoded) == 1 and len(codes) < 255:
            codes.append(code)
            ids.append(encoded[0])
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError("Insufficient or aliased native candidate tokens")
    return codes, ids


def canonicalize_record(record):
    result = copy.deepcopy(record)
    inp, target = result["input"], result["target"]["probabilities"]
    options = inp["options"]
    n = len(options)
    if inp["type"] not in ("choice", "noul", "score") or not 1 <= n <= 255:
        raise ValueError("Unsupported type or candidate capacity")
    if (
        len(target) != n
        or any(
            not isinstance(p, (int, float))
            or isinstance(p, bool)
            or not math.isfinite(p)
            or not 0 <= p <= 1
            for p in target
        )
        or abs(sum(target) - 1) > 1e-5
    ):
        raise ValueError("Invalid aligned soft targets")
    if inp["type"] == "noul":
        labels = [o["label"] for o in options]
        if n != 2 or set(labels) != {"false", "true"}:
            raise ValueError("Noul requires false and true labels")
        order = [labels.index("false"), labels.index("true")]
        inp["options"] = [options[i] for i in order]
        result["target"]["probabilities"] = [target[i] for i in order]
    elif inp["type"] == "score":
        if not 2 <= n <= 10:
            raise ValueError("Native Score requires 2..10 levels")
        for i, option in enumerate(options):
            option["label"] = f"option_{i}"
            value = option["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Score values must be finite numbers")
    elif len({o["label"] for o in options}) != n:
        raise ValueError("Choice labels must be unique")
    return result


def select_records(rows, fraction=1.0, seed=42, smoke=False, smoke_examples=24):
    if not 0 < fraction <= 1 or (smoke and fraction != 1):
        raise ValueError("Use either smoke or a fraction in (0,1]")
    if smoke:
        selected = []
        for kind in ("noul", "choice", "score"):
            selected.extend(
                sorted(
                    [r for r in rows if r["input"]["type"] == kind],
                    key=lambda r: (len(canonical_json(r["input"])), r["id"]),
                )[: max(1, smoke_examples // 3)]
            )
        return selected
    if fraction == 1:
        return rows
    groups = {}
    for r in rows:
        key = (r.get("provenance", {}).get("dataset", r.get("subset", "")), r["input"]["type"])
        groups.setdefault(key, set()).add(r.get("parent_id", r["id"]))
    selected = set()
    for parents in groups.values():
        ordered = sorted(parents, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest())
        selected.update(ordered[: max(1, math.ceil(len(ordered) * fraction))])
    return [r for r in rows if r.get("parent_id", r["id"]) in selected]


def learning_rate_factor(step, total, warmup_ratio):
    warmup = max(1, int(total * warmup_ratio))
    return min(1.0, (step + 1) / warmup) * (
        0.2 + 0.8 * max(0.0, 1 - max(0, step - warmup) / max(1, total - warmup))
    )


def prompt_contract(tokenizer, config, codes, ids):
    marker = "SAIHAI_PAYLOAD_9dc8a3f2"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if config["model"]["model_type"] == "llama":
        kwargs.update(
            reasoning_effort=config["prompt"]["reasoning_effort"],
            conversation_start_date=config["prompt"]["conversation_start_date"],
        )
    else:
        kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **kwargs)
    if text.count(marker) != 1:
        raise ValueError("Base chat template did not preserve payload marker")
    prefix, suffix = text.split(marker)
    if config["model"]["model_type"] == "llama":
        suffix += "<|channel|>final<|message|>"
    return {
        "model_id": config["model"]["model_id"] + "-saihai",
        "base": config["model"],
        "codes": codes,
        "code_token_ids": ids,
        "prompt_prefix": prefix,
        "prompt_suffix": suffix,
        "prompt_version": "saihai-pplx-json-v1",
        "temperature": 1.0,
        "max_input_tokens": config["evaluation"].get(
            "max_input_tokens", config["training"]["max_length"]
        ),
        "training_max_input_tokens": config["training"]["max_length"],
        "tokenizer_file": "tokenizer.json",
        "serialization": "minja-json-6-significant-digits",
        "max_score_levels": 10,
    }


def load_hub_records(config, *, split="train"):
    """Read split shards, verify artifacts and restore global source order."""
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    data = config["dataset"]
    if data.get("split") != "train" or split not in {"train", "validation"}:
        raise ValueError("Training must use train with a separate validation split")

    requested_revision = data["revision"]
    revision = (
        HfApi().dataset_info(data["repo_id"], revision="main").sha
        if requested_revision == "latest"
        else requested_revision
    )
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("Could not resolve the dataset to a full Hub revision")

    def download(name):
        return Path(hf_hub_download(data["repo_id"], name, repo_type="dataset", revision=revision))

    manifest_path = download("dataset-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    checksums = json.loads(download("checksums.json").read_text())
    if sha256(manifest_path) != checksums.get("dataset-manifest.json"):
        raise ValueError("Hub dataset manifest checksum mismatch")
    if "training_view" in manifest:
        if (
            manifest["training_view"] != "merged-no-3000"
            or manifest.get("reinforcement_included") is not False
        ):
            raise ValueError("Expected the merged-no-3000 training view")
    elif not (
        manifest.get("dataset") == data["repo_id"]
        and manifest.get("schema_version") == 1
        and manifest.get("file_integrity", {}).get("algorithm") == "SHA-256"
        and manifest.get("file_integrity", {}).get("manifest") == "checksums.json"
        and manifest.get("configuration_layout") == "one-config-per-subset"
    ):
        raise ValueError("Unsupported dataset release manifest")
    layout = manifest.get("configuration_layout", "merged")
    if layout == "one-config-per-subset":
        configs = manifest["configs"]
        names = [c["config_name"] for c in configs]
        if len(set(names)) != len(names) or any(
            not isinstance(n, str) or not n or n in (".", "..") or "/" in n or "\\" in n
            for n in names
        ):
            raise ValueError("Invalid or duplicate config names")
        files = []
        for c in configs:
            count = c[f"{split}_rows"]
            if type(count) is not int or count < 0:
                raise ValueError("Invalid config train count")
            if count:
                files.append((f"data/{c['config_name']}/{split}.parquet", count))
    elif layout == "merged":
        files = [(f"data/{split}.parquet", manifest["splits"][split]["rows"])]
    else:
        raise ValueError("Unsupported dataset configuration layout")
    if not files:
        raise ValueError("No training shards")

    def read_shard(item):
        name, count = item
        path = download(name)
        if sha256(path) != checksums.get(name):
            raise ValueError("Hub dataset artifact checksum mismatch: " + name)
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != count:
            raise ValueError("Training shard count differs from manifest: " + name)
        result = []
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                record = json.loads(row["record_json"])
                if (
                    record["id"] != row["id"]
                    or record["input"] != json.loads(row["input_json"])
                    or record["target"]["probabilities"] != row["probabilities"]
                ):
                    raise ValueError("Serialized identity, inputs or soft targets changed")
                for key in ("jdmb_source_family", "subset", "dataset_group"):
                    if key in row:
                        if key in record and record[key] != row[key]:
                            raise ValueError("Serialized filter metadata changed: " + key)
                        record[key] = row[key]
                order = row["source_order"]
                if type(order) is not int or order < 0:
                    raise ValueError("Invalid source ordering")
                result.append((order, record))
        return result

    with ThreadPoolExecutor(max_workers=min(8, len(files))) as pool:
        shards = list(pool.map(read_shard, files))
    rows = sorted((row for shard in shards for row in shard), key=lambda x: x[0])
    if len(rows) != manifest["splits"][split]["rows"]:
        raise ValueError("Training split size differs from release manifest")
    orders = [order for order, _ in rows]
    if len(set(orders)) != len(rows) or len({record["id"] for _, record in rows}) != len(rows):
        raise ValueError("Duplicate source ordering or record identity")
    if layout == "merged" and orders != list(range(len(rows))):
        raise ValueError("Missing legacy source ordering")
    # Filtered subset releases retain original global indices, including gaps.
    artifact_hashes = {name: checksums[name] for name, _ in files}
    receipt = {
        "repo_id": data["repo_id"],
        "requested_revision": requested_revision,
        "revision": revision,
        "split": split,
        "configuration_layout": layout,
        f"{split}_shards": len(files),
        "manifest_sha256": sha256(manifest_path),
        f"{split}_artifact_sha256": artifact_hashes,
        "source_file_sha256": manifest.get("source_file_sha256", {}).get(split),
        "source_order_min": orders[0],
        "source_order_max": orders[-1],
        "rows": len(rows),
    }
    if layout == "merged":
        receipt[f"{split}_parquet_sha256"] = artifact_hashes[f"data/{split}.parquet"]
    return [r for _, r in rows], receipt


def validation_steps(total, count):
    """Spread evaluations over optimizer steps and always include the final step."""
    if total <= 0 or count <= 0:
        raise ValueError("Validation steps and count must be positive")
    return {math.ceil(total * i / min(total, count)) for i in range(1, min(total, count) + 1)}


def select_validation_records(rows, per_subset, seed):
    """Keep a deterministic, bounded held-out sample from every available subset."""
    if per_subset <= 0:
        raise ValueError("Validation per-subset count must be positive")
    grouped = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        grouped.setdefault(row["subset"], []).append(row)
    rng = random.Random(seed)
    selected = []
    for subset in sorted(grouped):
        candidates = grouped[subset]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_subset])
    return selected


def validation_loss(model, rows, pad, *, batch_size=1, device="cuda"):
    """Measure example-weighted soft-target CE, including per-subset losses."""
    was_training = model.training
    totals = {}
    try:
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                group = rows[offset : offset + batch_size]
                batch = collate(group, pad, device=device)
                logits = model(batch["input_ids"], batch["attention_mask"])
                masked = (
                    logits[:, : batch["target"].shape[1]]
                    .float()
                    .masked_fill(~batch["options"], -1e9)
                )
                losses = -(batch["target"] * masked.log_softmax(-1)).sum(-1)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("Nonfinite validation loss")
                for row, loss in zip(group, losses.cpu().tolist(), strict=True):
                    bucket = totals.setdefault(row["subset"], {"sum": 0.0, "examples": 0})
                    bucket["sum"] += loss
                    bucket["examples"] += 1
    finally:
        model.train(was_training)
    count = sum(v["examples"] for v in totals.values())
    if not count:
        raise ValueError("No eligible validation records")
    return {
        "loss": sum(v["sum"] for v in totals.values()) / count,
        "examples": count,
        "subsets": {
            k: {"loss": v["sum"] / v["examples"], "examples": v["examples"]}
            for k, v in totals.items()
        },
    }


def batches(rows, size, seed):
    order = list(rows)
    rng = random.Random(seed)
    rng.shuffle(order)
    groups = []
    for i in range(0, len(order), 128):
        bucket = sorted(order[i : i + 128], key=lambda r: len(r["input_ids"]))
        groups.extend(bucket[j : j + size] for j in range(0, len(bucket), size))
    rng.shuffle(groups)
    return groups


def export_model(model, tokenizer, config, contract, output):
    from transformers import LlamaForCausalLM, Qwen3_5ForConditionalGeneration

    model.eval()
    backbone = (
        model.backbone.merge_and_unload()
        if hasattr(model.backbone, "merge_and_unload")
        else model.backbone
    )
    cls = (
        LlamaForCausalLM
        if config["model"]["model_type"] == "llama"
        else Qwen3_5ForConditionalGeneration
    )
    # CPU export avoids holding both 8B backbones on the training GPU.
    backbone = backbone.to("cpu")
    original = cls.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    if config["model"]["model_type"] == "llama":
        original.model = backbone
        original.config.tie_word_embeddings = False
    else:
        original.model.language_model = backbone
        original.config.text_config.tie_word_embeddings = False
        original.config.tie_word_embeddings = False
    # Clone even for tied bases: writing decision rows must not alter input embeddings.
    original.lm_head.weight = torch.nn.Parameter(
        original.lm_head.weight.detach().clone(), requires_grad=False
    )
    with torch.no_grad():
        original.lm_head.weight[contract["code_token_ids"]] = (
            model.head.weight.detach().cpu().to(torch.bfloat16)
        )
    output.mkdir()
    original.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    saved = dict(contract, vocab_size=original.lm_head.out_features)
    dump(output / "decision-inference.json", saved)
    return saved


def train(config, output, *, smoke=False, fraction=1.0, max_steps=None):
    from transformers import Adafactor, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one free GPU with CUDA_VISIBLE_DEVICES")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = config["training"]
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg.get("cpu_threads", 8))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dump(
        output / "config.json",
        dict(config, run={"smoke": smoke, "fraction": fraction, "max_steps": max_steps}),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["model_id"],
        revision=config["model"]["revision"],
        trust_remote_code=config["model"].get("trust_remote_code", False),
    )
    codes, ids = pplx_codebook(tokenizer)
    contract = prompt_contract(tokenizer, config, codes, ids)
    rows, source = load_hub_records(config)
    validation_config = copy.deepcopy(config)
    validation_config["dataset"]["revision"] = source["revision"]
    held_out, validation_source = load_hub_records(validation_config, split="validation")
    for key in ("id", "parent_id", "group_id"):
        training_keys = {r[key] for r in rows if r.get(key) is not None}
        if training_keys & {r[key] for r in held_out if r.get(key) is not None}:
            raise ValueError("Training and validation identities overlap: " + key)
    rows, filter_receipt = filter_training_records(rows, config["dataset"].get("train_filter", {}))
    dump(output / "train-filter-receipt.json", filter_receipt)
    source["train_filter"] = filter_receipt
    probe_candidates = []
    if smoke and cfg.get("memory_probe", False):
        for row in sorted(rows, key=lambda r: len(canonical_json(r["input"])), reverse=True)[:64]:
            canonical = canonicalize_record(row)
            tokens = tokenizer.encode(
                render_input(canonical["input"], contract), add_special_tokens=False
            )
            if len(tokens) <= cfg["max_length"]:
                probe_candidates.append(
                    {
                        "id": row["id"],
                        "input_ids": tokens,
                        "probabilities": canonical["target"]["probabilities"],
                    }
                )
    held_out = select_validation_records(held_out, cfg.get("validation_per_subset", 3), seed)
    rows = select_records(rows, fraction, seed, smoke, cfg["smoke_examples"])
    tokenized = []
    excluded = []
    validation_tokenized = []
    validation_excluded = []
    for row in held_out:
        canonical = canonicalize_record(row)
        tokens = tokenizer.encode(
            render_input(canonical["input"], contract), add_special_tokens=False
        )
        if len(tokens) > cfg["max_length"]:
            validation_excluded.append(row["id"])
            continue
        validation_tokenized.append(
            {
                "id": row["id"],
                "subset": row["subset"],
                "input_ids": tokens,
                "probabilities": canonical["target"]["probabilities"],
            }
        )
    if not validation_tokenized:
        raise ValueError("No eligible validation records")
    dump(
        output / "validation-receipt.json",
        dict(
            validation_source,
            selected=len(validation_tokenized),
            per_subset=cfg.get("validation_per_subset", 3),
            excluded_overflow=validation_excluded,
        ),
    )
    dump(output / "validation-ids.json", [r["id"] for r in validation_tokenized])
    for row in rows:
        canonical = canonicalize_record(row)
        text = render_input(canonical["input"], contract)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > cfg["max_length"]:
            excluded.append(row["id"])
            continue
        tokenized.append(
            {
                "id": row["id"],
                "task": canonical["input"]["type"],
                "input": canonical["input"],
                "input_ids": tokens,
                "probabilities": canonical["target"]["probabilities"],
            }
        )
    if not tokenized:
        raise ValueError("No eligible training records")
    dump(
        output / "selected-ids.json",
        {"train": [r["id"] for r in tokenized], "excluded_overflow": excluded},
    )
    dump(
        output / "data-receipt.json",
        dict(
            source,
            selected=len(tokenized),
            excluded_overflow=len(excluded),
            native_capacity=len(codes),
        ),
    )
    dump(
        output / "implementation-receipt.json",
        {
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "allocator": os.environ.get("PYTORCH_ALLOC_CONF"),
        },
    )
    dump(output / "decision-inference.json", contract)
    # These are training-only probes, never S1MB targets or model-selection scores.
    dump(output / "training-probes.json", tokenized[: min(24, len(tokenized))])
    cache = {"base": config["model"], "codes": codes, "code_token_ids": ids}
    model, info = build_model(cfg, cache)
    groups = [
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": cfg["learning_rate"],
        },
        {"params": list(model.head.parameters()), "lr": cfg["head_learning_rate"]},
    ]
    opt = (
        Adafactor(
            groups,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=cfg["weight_decay"],
        )
        if cfg["optimizer"] == "adafactor"
        else torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    )
    initial = [g["lr"] for g in opt.param_groups]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if cfg.get("memory_probe", False):
        probe = max(probe_candidates or tokenized, key=lambda r: len(r["input_ids"]))
        model.train()
        torch.cuda.reset_peak_memory_stats()
        batch = collate([probe], pad)
        loss, _ = soft_ce(
            model(batch["input_ids"], batch["attention_mask"]), batch["target"], batch["options"]
        )
        loss.backward()
        opt.zero_grad(set_to_none=True)
        measured = {
            "id": probe["id"],
            "tokens": len(probe["input_ids"]),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "loss": loss.item(),
        }
        dump(output / "memory-probe.json", measured)
        print("MEMORY_PROBE", json.dumps(measured), flush=True)
        del batch, loss
        torch.cuda.empty_cache()
    all_batches = []
    for epoch in range(cfg["epochs"]):
        all_batches.extend(batches(tokenized, cfg["batch_size"], seed + epoch))
    accumulation = cfg["gradient_accumulation"]
    total = math.ceil(len(all_batches) / accumulation)
    if smoke:
        required = cfg["smoke_steps"] * accumulation
        all_batches = (all_batches * math.ceil(required / len(all_batches)))[:required]
        total = cfg["smoke_steps"]
    if max_steps is not None:
        total = min(total, max_steps)
    evaluation_steps = validation_steps(total, cfg.get("validation_runs", 10))
    validation_history = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    history = []
    examples = 0
    with (output / "train-log.jsonl").open("w") as log:
        for step, offset in enumerate(range(0, len(all_batches), accumulation)):
            if step >= total:
                break
            chunk = all_batches[offset : offset + accumulation]
            count = sum(len(g) for g in chunk)
            running = 0.0
            model.train()
            for group in chunk:
                batch = collate(group, pad)
                loss, _ = soft_ce(
                    model(batch["input_ids"], batch["attention_mask"]),
                    batch["target"],
                    batch["options"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                (loss * len(group) / count).backward()
                running += loss.item() * len(group) / count
                examples += len(group)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            factor = learning_rate_factor(step, total, cfg["warmup_ratio"])
            for group, lr in zip(opt.param_groups, initial, strict=True):
                group["lr"] = lr * factor
            opt.step()
            opt.zero_grad(set_to_none=True)
            row = {
                "step": step + 1,
                "loss": running,
                "gradient_norm": float(norm),
                "examples": examples,
                "learning_rates": [g["lr"] for g in opt.param_groups],
                "elapsed_seconds": time.perf_counter() - start,
            }
            if step + 1 in evaluation_steps:
                validation_start = time.perf_counter()
                measured = validation_loss(model, validation_tokenized, pad, batch_size=1)
                measured.update(
                    step=step + 1, elapsed_seconds=time.perf_counter() - validation_start
                )
                validation_history.append(measured)
                dump(output / "validation-history.json", validation_history)
                row["validation_loss"] = measured["loss"]
                print("VALIDATION", json.dumps(measured), flush=True)
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step == 0 or (step + 1) % cfg.get("log_every", 10) == 0 or step + 1 == total:
                print("STEP", json.dumps(row), flush=True)
    model.eval()
    probe_references = []
    with torch.inference_mode():
        for row in tokenized[:3]:
            batch = collate([row], pad)
            logits = model(batch["input_ids"], batch["attention_mask"])[
                0, : len(row["probabilities"])
            ].float()
            probe_references.append(
                {"id": row["id"], "probabilities": logits.softmax(-1).cpu().tolist()}
            )
    dump(output / "checkpoint-probe-probabilities.json", probe_references)
    checkpoint = output / "checkpoint"
    save_checkpoint(model, cfg, cache, checkpoint)
    dump(
        output / "checkpoint-hashes.json",
        {str(p.relative_to(checkpoint)): sha256(p) for p in checkpoint.rglob("*") if p.is_file()},
    )
    summary = {
        "steps": len(history),
        "examples_seen": examples,
        "selected_train": len(tokenized),
        "loss_first": history[0]["loss"],
        "loss_last": history[-1]["loss"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "training_seconds": time.perf_counter() - start,
        "validation_runs": len(validation_history),
        "validation_examples": len(validation_tokenized),
        "validation_loss_first": validation_history[0]["loss"],
        "validation_loss_last": validation_history[-1]["loss"],
        "model": info,
    }
    del opt
    gc.collect()
    torch.cuda.empty_cache()
    saved = export_model(model, tokenizer, config, contract, output / "export")
    dump(output / "summary.json", summary)
    dump(
        output / "export-hashes.json",
        {
            str(p.relative_to(output / "export")): sha256(p)
            for p in (output / "export").rglob("*")
            if p.is_file()
        },
    )
    print("SUMMARY", json.dumps(summary), flush=True)
    return saved


def evaluation_run_id(output):
    path = Path(output).resolve()
    return (
        path.parent.name
        + "-"
        + path.name
        + "-"
        + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument(
        "--fraction", type=float, default=1.0, help="Parent-group sample; use 0.1 for ten percent"
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument(
        "--s1mb-root", type=Path, help="S1MB checkout, required for post-training evaluation"
    )
    parser.add_argument(
        "--eval-limit", type=int, help="Cases per S1MB benchmark; omitted means full evaluation"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("batch_size", "gradient_accumulation"):
        value = getattr(args, key)
        if value is not None:
            if value <= 0:
                parser.error(key + " must be positive")
            config["training"][key] = value
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not args.s1mb_root:
        parser.error("--s1mb-root is required to evaluate after training")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", config["training"]["allocator"])
    train(config, args.output, smoke=args.smoke, fraction=args.fraction, max_steps=args.max_steps)
    gc.collect()
    torch.cuda.empty_cache()
    import subprocess

    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "saihai.s1mb_eval",
        "--export",
        str((args.output / "export").resolve()),
        "--run-id",
        evaluation_run_id(args.output),
        "--output",
        str((args.output / "s1mb-summary.json").resolve()),
    ]
    if args.eval_limit is not None:
        command += ["--limit", str(args.eval_limit)]
    if config["evaluation"].get("revision"):
        command += ["--data-revision", config["evaluation"]["revision"]]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    subprocess.run(command, cwd=args.s1mb_root / "evaluator", env=env, check=True)


if __name__ == "__main__":
    main()
