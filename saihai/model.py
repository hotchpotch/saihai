"""Load local or Hub exports with the same interface for Torch and llama.cpp."""

from pathlib import Path

from .inference import DecisionInterface, LlamaDecisionClient


class TorchDecisionModel(DecisionInterface):
    """Read the exported LM-head candidate rows without generating text."""

    def __init__(self, directory, device="cuda", dtype=None):
        import torch
        from tokenizers import Tokenizer
        from transformers import AutoConfig, LlamaForCausalLM, Qwen3_5ForConditionalGeneration

        directory = Path(directory)
        super().__init__(directory / "decision-inference.json")
        if dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        cfg = AutoConfig.from_pretrained(directory)
        cls = Qwen3_5ForConditionalGeneration if cfg.model_type == "qwen3_5" else LlamaForCausalLM
        self.model = cls.from_pretrained(
            directory, dtype=dtype, device_map={"": str(device)}, attn_implementation="sdpa",
        ).eval()
        self.tokenizer = Tokenizer.from_file(str(directory / self.contract["tokenizer_file"]))
        ids = self.contract["code_token_ids"]
        if any(not isinstance(i, int) or not 0 <= i < self.model.lm_head.out_features for i in ids):
            raise ValueError("Candidate token IDs exceed model vocabulary")
        self.token_ids = ids

    def predict(self, inp):
        import torch

        text = self.render(inp)
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or len(ids) > self.contract["max_input_tokens"]:
            raise ValueError("Input exceeds contract context limit; refusing truncation")
        device = self.model.device
        inputs = torch.tensor([ids], device=device)
        with torch.inference_mode():
            backbone = self.model.model.language_model if hasattr(self.model.model, "language_model") else self.model.model
            hidden = backbone(
                input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False,
            ).last_hidden_state[:, -1:, :]
            logits = self.model.lm_head(hidden)[0, 0, self.token_ids].float()
            probabilities = logits[:len(inp["options"])].softmax(-1).cpu().tolist()
        return {"probabilities": probabilities, "input_tokens": len(ids), "backend_output_tokens": 0}


class Saihai:
    """A typed decision model with a llama.cpp-style systemone(request) API."""

    def __init__(self, backend):
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls, model_id, *, revision="main", token=None, backend="transformers",
        device="cuda", dtype=None, backend_url=None, native_tokenizer=False,
    ):
        """Load a self-contained export; llama.cpp requires an existing server."""
        if backend not in {"transformers", "llama.cpp"}:
            raise ValueError("backend must be transformers or llama.cpp")
        if backend == "llama.cpp" and not backend_url:
            raise ValueError("llama.cpp requires backend_url for a running llama-server")
        directory = Path(model_id)
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            patterns = ["decision-inference.json"]
            if backend == "transformers":
                patterns += ["config.json", "model*.safetensors", "model*.safetensors.index.json", "tokenizer.json"]
            elif not native_tokenizer:
                patterns += ["tokenizer.json"]
            directory = Path(snapshot_download(
                repo_id=str(model_id), revision=revision, token=token, allow_patterns=patterns,
            ))
        if backend == "transformers":
            return cls(TorchDecisionModel(directory, device=device, dtype=dtype))
        return cls(LlamaDecisionClient(
            backend_url, directory / "decision-inference.json", native_tokenizer=native_tokenizer,
        ))

    def systemone(self, request):
        return self.backend.systemone(request)

    def predict(self, state, question):
        """Convenience wrapper returning a single typed answer."""
        return self.systemone({"state": state, "questions": {"decision": question}})["answers"]["decision"]
