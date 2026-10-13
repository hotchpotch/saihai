"""Typed decisions over an unmodified llama-server and a model-bound contract."""

import json
import math
from pathlib import Path
from urllib.request import Request, urlopen


class DecisionInterface:
    """Shared renderer and llama.cpp-style System One response mapping."""

    def __init__(self, contract):
        self.contract_path = Path(contract) if isinstance(contract, (str, Path)) else None
        self.contract = json.loads(self.contract_path.read_text()) if self.contract_path else contract
        ids = self.contract["code_token_ids"]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(self.contract["codes"]):
            raise ValueError("Invalid candidate codebook")

    def render(self, inp):
        if self.contract.get("prompt_version") == "saihai-pplx-json-v1":
            from .protocol import render_input

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
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Score values must be finite numbers")
                item["numeric_value"] = value
            if inp["type"] == "score" and "value" not in option:
                raise ValueError("Score options require numeric values")
            mapped.append(item)
        payload = {"instruction": inp["instructions"], "state": inp["state"], "type": inp["type"], "options": mapped}
        return self.contract["prompt_prefix"] + json.dumps(payload, ensure_ascii=False, allow_nan=False) + self.contract["prompt_suffix"]

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
                options = [{"label": key, "description": key if criteria.get(key) is None else criteria[key]} for key in keys]
            elif kind == "score":
                criteria = q["criteria"]
                if not isinstance(criteria, list):
                    raise ValueError("Score criteria must be an array")
                keys = [str(i) for i in range(len(criteria))]
                options = [{"label": f"option_{i}", "description": value["description"] if isinstance(value, dict) and "value" in value else value,
                            "value": value["value"] if isinstance(value, dict) and "value" in value else i}
                           for i, value in enumerate(criteria)]
            else:
                raise ValueError("Unsupported decision type")
            if len(keys) != len(set(keys)) or any(not isinstance(key, str) for key in keys):
                raise ValueError("Option IDs must be unique strings")
            if kind == "noul" and set(keys) != {"false", "true"}:
                raise ValueError("Noul requires false and true options")
            result = self.predict({"state": state, "type": kind, "instructions": q.get("instructions", qid), "options": options})
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
                        1.0 if len(ps) == 1 else max(0.0, (ps[mode] - 1 / len(ps)) / (1 - 1 / len(ps)))
                    )
                else:
                    mode = max(range(len(ps)), key=ps.__getitem__)
                    distance = sum(p * abs(i - mode) for i, p in enumerate(ps))
                    uniform_distance = sum(abs(i - (len(ps) - 1) / 2) for i in range(len(ps))) / len(ps)
                    answer["confidence"] = max(0.0, 1 - distance / uniform_distance)
                    answer["score"] = sum(i * p for i, p in enumerate(ps))
                    answer["legend"] = {key: option["description"] for key, option in zip(keys, options, strict=True)}
                    answer["expected_value"] = sum(option["value"] * p for option, p in zip(options, ps, strict=True))
                    answer["numeric_values"] = dict(zip(keys, [o["value"] for o in options], strict=True))
            answers[qid] = answer
        return {"answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                "backend_usage": {"output_tokens": backend_output_tokens}, "model": self.contract["model_id"]}


class LlamaDecisionClient(DecisionInterface):
    """Read candidate probabilities over an independently managed llama-server."""

    def __init__(self, backend_url, contract, timeout=600, native_tokenizer=False):
        super().__init__(contract)
        self.url = backend_url.rstrip("/")
        self.timeout = timeout
        self.vocab_size = self.contract.get("vocab_size")
        if not isinstance(self.vocab_size, int) or self.vocab_size <= max(self.contract["code_token_ids"]):
            raise ValueError("llama.cpp requires a valid vocab_size in the decision contract")
        self.tokenizer = None
        if "tokenizer_file" in self.contract and not native_tokenizer:
            if self.contract_path is None:
                raise ValueError("File-based tokenization requires a contract path")
            from tokenizers import Tokenizer

            path = self.contract_path.parent / self.contract["tokenizer_file"]
            if not path.is_file():
                raise FileNotFoundError(path)
            self.tokenizer = Tokenizer.from_file(str(path))

    def _post(self, endpoint, payload):
        request = Request(self.url + endpoint, data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def predict(self, inp):
        text = self.render(inp)
        tokens = (
            self.tokenizer.encode(text, add_special_tokens=False).ids if self.tokenizer is not None
            else self._post("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]
        )
        if len(tokens) > self.contract["max_input_tokens"]:
            raise ValueError("Input exceeds contract context limit; refusing truncation")
        ids = self.contract["code_token_ids"][:len(inp["options"])]
        candidates = set(ids)
        # Grammar is applied lazily by llama.cpp rejection sampling. Explicit
        # masking also constrains reported probabilities when the first sampled
        # token already satisfies the grammar. False is the native API ban value.
        logit_bias = [[token, False] for token in range(self.vocab_size) if token not in candidates]
        response = self._post("/completion", {
            "prompt": tokens, "n_predict": 1, "n_probs": len(ids),
            "grammar": "root ::= " + " | ".join(f"<[{token}]>" for token in ids),
            "samplers": ["temperature"], "temperature": 1.0,
            "logit_bias": logit_bias, "post_sampling_probs": True, "cache_prompt": False, "seed": 42,
        })
        values = response["completion_probabilities"][0]["top_probs"]
        probabilities = {row["id"]: row["prob"] for row in values}
        if len(values) != len(ids) or set(probabilities) != set(ids):
            raise ValueError("Backend omitted or duplicated candidate probabilities")
        ps = [probabilities[token] for token in ids]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in ps) or abs(sum(ps) - 1) > 1e-5:
            raise ValueError("Backend returned an invalid probability distribution")
        return {"probabilities": ps, "input_tokens": len(tokens), "backend_output_tokens": 1}
