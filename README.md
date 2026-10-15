# saihai

**saihai (采配)** is a Japanese **Typed Decision Model**. Given a state and named
questions, it returns probabilities for three decision types in one forward
pass, without generating text:

- **Noul**: yes / no (probability of true)
- **Choice**: one of 1–255 candidates (selected key and every candidate's probability)
- **Score**: a 2–10 level rubric (expected level and per-level probabilities)

This repository contains the training, evaluation, inference and GGUF conversion
code used to build the released models. The request format follows llama.cpp's
`/v1/systemone` endpoint.

> 日本語: saihai (采配) は、はい/いいえ・選択・段階評価の判断を、文章を生成せずに確率として返す
> 日本語の Decision モデルです。このリポジトリには、学習・評価・推論・GGUF 変換のコードがあります。
> 学習と評価の手順は [日本語ドキュメント](docs/README.md) を参照してください。

## Released models and data

| Repository | Description |
| --- | --- |
| [hotchpotch/saihai-v1-8b](https://huggingface.co/hotchpotch/saihai-v1-8b) | 8B model based on [llm-jp-4.1-8b-thinking](https://huggingface.co/llm-jp/llm-jp-4.1-8b-thinking) (Transformers) |
| [hotchpotch/saihai-v1-8b-gguf](https://huggingface.co/hotchpotch/saihai-v1-8b-gguf) | GGUF for llama.cpp's `/v1/systemone` (BF16 to Q2_K) |
| [hotchpotch/saihai-v1-310m](https://huggingface.co/hotchpotch/saihai-v1-310m) | Small encoder model trained with [Bekko System One](https://github.com/hotchpotch/bekko-system-one) (not built from this repository; see its [training guide](https://github.com/hotchpotch/bekko-system-one/blob/main/docs/saihai-v1-310m.md)) |
| [hotchpotch/saihai-v1-dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset) | Training data for saihai-v1-8b |
| [hotchpotch/saihai-v1-310m-dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-310m-dataset) | Training data for saihai-v1-310m |

Benchmark results are on the
[S1MB leaderboard](https://huggingface.co/spaces/hotchpotch/S1MB-leaderboard?category=japanese-v1).
Each model card has a quick start and evaluation results.

## Which interface to use

| Goal | Recommended path |
| --- | --- |
| Serve the model on a GPU server | [saihai-v1-8b-gguf](https://huggingface.co/hotchpotch/saihai-v1-8b-gguf) with llama.cpp's native `/v1/systemone` ([details](#llamacpp-native-v1systemone)) |
| Try it from Python with Transformers | `AutoModel` with `trust_remote_code=True`; no saihai install needed ([details](#transformers-standalone-remote-code)) |
| Use one Python API for both PyTorch and llama.cpp | The `Saihai` class in this package ([details](#python-package-saihai-class)) |
| Train, evaluate or convert your own model | This repository ([Training](#training), [Export and conversion](#export-and-conversion)) |
| Train saihai-v1-310m | [Training Saihai v1 310M](https://github.com/hotchpotch/bekko-system-one/blob/main/docs/saihai-v1-310m.md) in the Bekko System One repository |

All inference paths take the same `state` / `questions` request and return the
same `answers` response ([Request and response](#request-and-response)).

> [!IMPORTANT]
> - **llama.cpp with LLM-jp models (including saihai-v1-8b):** use a llama.cpp
>   build that includes the tokenizer fix from
>   [ggml-org/llama.cpp#28864](https://github.com/ggml-org/llama.cpp/pull/28864)
>   (not yet merged upstream). Without it, inputs containing newlines are
>   tokenized differently from training and probabilities change.
> - **Context length:** saihai-v1-8b accepts up to 65,536 input tokens, but it was
>   trained with an 8,192-token limit. Quality on longer inputs has not been
>   evaluated. Overlong inputs are rejected, never truncated.
> - **Probabilities** are model outputs and are not calibrated; they are not
>   guaranteed correctness rates.

## Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Request and response](#request-and-response)
  - [llama.cpp native `/v1/systemone`](#llamacpp-native-v1systemone)
  - [Transformers standalone remote code](#transformers-standalone-remote-code)
  - [Python package (`Saihai` class)](#python-package-saihai-class)
  - [Long-context serving](#long-context-serving)
  - [Advanced: candidate readout client and HTTP frontend](#advanced-candidate-readout-client-and-http-frontend)
- [Training](#training)
- [Export and conversion](#export-and-conversion)
- [Dataset loading utilities](#dataset-loading-utilities)
- [Repository layout](#repository-layout)
- [License](#license)

## Installation

Python 3.12+ is required. Choose the extras you need:

| Extra | Use |
| --- | --- |
| *(none)* | llama.cpp client, HTTP frontend and GGUF tools |
| `inference` | PyTorch / Transformers inference |
| `datasets` | Dataset loading utilities |
| `training` | LoRA training (CUDA) |

```sh
git clone https://github.com/hotchpotch/saihai.git
cd saihai
uv sync --python 3.12 --locked --extra inference
# or: pip install '.[inference]'
```

For training, use `uv sync --python 3.12 --locked --extra training --extra inference --extra datasets`.

Using a released model through llama.cpp or Transformers remote code does not
require installing this package.

## Usage

### Request and response

A request has a `state` (text or JSON) and named `questions`. Each question has
a `type`, `instructions` and `criteria`. Questions are evaluated independently.

| Type | `criteria` | Answer fields |
| --- | --- | --- |
| `noul` | Optional `false` / `true` definitions | `noul`: probability of true |
| `choice` | Ordered object mapping candidate keys to descriptions (1–255) | `choice`: selected key; `probabilities`: every candidate |
| `score` | Array of 2–10 levels (strings, or objects with `value` and `description`) | `score`: expected zero-based level index; `probabilities` by level |

- **Noul** candidates are always false then true, with those strings as default
  descriptions.
- **Choice** preserves criterion order and uses the keys as input labels.
- **Score** uses `option_i` labels. When levels carry numeric `value`s, the
  Python backends also return `expected_value` and `numeric_values`; with the
  native llama.cpp API, compute the expected value client-side from the
  probabilities and those values.
- **Confidence** (Choice and Score) uses the llama.cpp formulas: normalized
  maximum probability for Choice, and normalized distance from the mode for
  Score. Temperature is fixed at 1; no calibration is claimed.
- **Numbers** in input JSON are rendered with **six significant digits**, as in
  training. Represent precise quantities in `state` as strings.

The response also contains `model` and `usage`. The optional request `model`
field must match the loaded model.

### llama.cpp native `/v1/systemone`

The released GGUF files embed the learned input format, the decision readout
profile (llama.cpp's existing `pplx-decider`) and a model-specific `systemone`
Jinja template. llama-server needs only the GGUF file: no saihai-specific C++
code, external tokenizer JSON, contract or Python frontend. Use a llama.cpp build
with the [LLM-jp tokenizer fix](#which-interface-to-use).

```sh
hf download hotchpotch/saihai-v1-8b-gguf saihai-v1-8b-Q8_0.gguf \
  --local-dir ./saihai-v1-8b-gguf

llama-server -m ./saihai-v1-8b-gguf/saihai-v1-8b-Q8_0.gguf \
  --alias saihai-v1-8b-gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --parallel 1 --flash-attn on --no-context-shift
```

Send requests to `/v1/systemone`. This example uses only the Python standard
library:

```python
import json
from urllib.request import Request, urlopen

body = {
    "state": "The customer requests a refund.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is a refund requested?"},
        "team": {
            "type": "choice", "instructions": "Choose the team.",
            "criteria": {"billing": "Payments", "technical": "Technical issues"},
        },
        "urgency": {
            "type": "score", "instructions": "Rate the urgency.",
            "criteria": ["Low", "Medium", "High"],
        },
    },
}
request = Request("http://127.0.0.1:8080/v1/systemone",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    print(json.load(response)["answers"])
```

Limits and differences from the Python backends:

- The template rejects questions with more candidates than the model's
  codebook (255 for saihai-v1-8b). llama.cpp reports Jinja template errors as
  HTTP 500, so check limits in your client (for benchmark adapters, set
  `max_candidates`).
- llama.cpp's Jinja float serializer has limited precision. Common levels such
  as 0.0, 0.25, 0.5 and 1.0 work, but floats that cannot reproduce Python's
  training JSON representation are rejected rather than rounded.
- The native API does not implement the `options`, `expected_value` or
  `cache_prompt: false` extensions of the Python frontend.
- Cache and batching conditions can slightly change quantized results.
- Keep the per-slot context at or below the model's input limit, and keep
  context shifting disabled.

To build native GGUF files from your own model, see
[Export and conversion](#export-and-conversion).

### Transformers standalone remote code

Each Transformers export includes `inference_saihai.py`, which runs on its own
without the saihai package or any training code. Install PyTorch,
`transformers==5.17.0`, `tokenizers>=0.23,<0.24`, `huggingface-hub>=1.0,<2` and
`accelerate>=1.0,<2`. Use Python 3.12+ and pin the same model revision for code
and weights.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "hotchpotch/saihai-v1-8b", revision="<commit hash>",
    trust_remote_code=True, device_map="auto", dtype="auto",
)
result = model.systemone({
    "state": "The checkout is failing.",
    "questions": {"urgent": {"type": "noul", "instructions": "Is action urgent?"}},
})
print(result["answers"])
```

`model.predict(state, question)` returns one typed answer. `AutoModel` loads the
decision wrapper; `AutoModelForCausalLM` still loads the standard backbone.
For a downloaded export, `from inference_saihai import SaihaiModel` and
`SaihaiModel.from_pretrained("./model", device="cpu")` also work.

### Python package (`Saihai` class)

The `Saihai` class gives the same interface for the PyTorch and llama.cpp
backends.

```python
from saihai import Saihai

model = Saihai.from_pretrained("hotchpotch/saihai-v1-8b", device="cuda")
result = model.systemone({
    "state": "The checkout is failing.",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is urgent action needed?"},
        "team": {
            "type": "choice", "instructions": "Choose the team.",
            "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages"},
        },
    },
})
print(result["answers"])
```

`from_pretrained` accepts a Hub ID or local export directory, and optional
`revision` and `token` arguments. The PyTorch backend loads a standard merged
Llama export and reads candidate logits without generating text. A single
question can use `model.predict(state, question)`. The input limit is stored in
the model's decision contract; overflow is rejected rather than truncated.

To use a GGUF model, start it in llama-server, then:

```python
model = Saihai.from_pretrained(
    "hotchpotch/saihai-v1-8b-gguf", backend="llama.cpp",
    backend_url="http://127.0.0.1:8080",
)
```

This downloads the decision contract and tokenizer, not the GGUF weights. If
the server's native tokenizer reproduces the training token IDs, pass
`native_tokenizer=True` to use GGUF tokenization; then only the decision
contract is downloaded.

### Long-context serving

saihai-v1-8b was trained with an 8,192-token input limit
(`training.max_length: 8192`) and exported with a 65,536-token inference limit
(`evaluation.max_input_tokens: 65536`), matching the base model's positional
capacity. Inputs up to 65,536 tokens are accepted, but performance on long
inputs, especially near 64K tokens, may be poor and has not been established by
the short-context benchmark results.

To serve 64K inputs with llama.cpp:

```sh
llama-server -m ./saihai-v1-8b-gguf/saihai-v1-8b-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 65536 --parallel 1 -b 8192 -ub 512 \
  --flash-attn on --no-context-shift
```

- The tested llama-server reserves one context position: `-c 65536` accepts up
  to 65,535 input tokens per slot. Python inference accepts 65,536 tokens.
- Context is divided between slots: four slots with 64K each require
  `-c 262144 --parallel 4`. KV-cache memory grows with total context, so
  allocate slots according to GPU memory.
- Quantization reduces model-weight memory; KV-cache precision is a separate
  setting. The batch size (`-b`) controls prompt processing, not the request
  context limit.
- Requests above the model or per-slot limit must be rejected rather than
  silently truncated.

### Advanced: candidate readout client and HTTP frontend

This path runs a Python frontend in front of an unmodified `llama-server` and
reads candidate probabilities through a one-token completion. It supports the
`options` and `expected_value` extensions, and passes exact token IDs from the
training tokenizer. For normal serving, use the
[native endpoint](#llamacpp-native-v1systemone) instead.

`LlamaDecisionClient` reads a model-bound `decision-inference.json` contract.
To prepare a model, export the merged backbone as a standard HF
`LlamaForCausalLM`, write the trained decision rows into their original LM-head
candidate-token rows, then convert to GGUF. Keep the exact training prompt,
codebook, tokenizer and context limit in the contract. Validate the conversion
against the original checkpoint; merge and precision changes can change
probabilities.

The client needs only the Python standard library and `tokenizers` when the
contract supplies `tokenizer_file`; it does not load Torch or model weights.
For an LLM-jp contract, keep its original `tokenizer.json` beside the contract:
the tested llama.cpp Unigram tokenizer without the fix did not reproduce
byte-fallback newlines. A plain GGUF string prompt is not equivalent until its
native tokenization passes parity checks.

```python
from saihai.inference import LlamaDecisionClient

client = LlamaDecisionClient("http://127.0.0.1:8080", "decision-inference.json")
result = client.predict({
    "state": "The customer asks for a refund.",
    "type": "choice",
    "instructions": "Choose the team.",
    "options": [
        {"label": "billing", "description": "Payments and refunds"},
        {"label": "technical", "description": "Technical problems"},
    ],
})
```

The typed HTTP frontend runs as a separate process. Bind both processes to
localhost or a trusted private network address. Select the GPU explicitly when
starting llama-server and keep its context slot large enough for the contract.

```sh
PYTHONPATH=. python -m saihai.server \
  --backend-url http://127.0.0.1:8080 \
  --contract /path/to/decision-inference.json \
  --host 127.0.0.1 --port 8098
```

`POST /v1/systemone` accepts the [standard request](#request-and-response). For
exact benchmark inputs, a question can instead supply an ordered `options`
array with `id`, `label`, `description`, and a numeric `value` for Score. This
preserves labels, definitions, order and numeric levels independently;
transport IDs are not included in model text. The `expected_value` and
`numeric_values` extensions preserve authored numeric scales, including
irregular levels.

The backend performs a one-token completion restricted by a token-ID grammar,
with all sampling filters disabled except unit temperature. All declared
candidate probabilities must be present; missing candidates and overflow fail,
with no uniform fallback. No reasoning or free text is generated. The API
reports zero user-facing output tokens and reports the internal
one-token-per-question readout separately in `backend_usage`.

## Training

This section covers the 8B-class models trained by this repository. A
step-by-step Japanese guide is in
[docs/training-and-evaluation.md](docs/training-and-evaluation.md).

### Setup

Install the training dependencies with Python 3.12:

```sh
uv sync --python 3.12 --locked --extra training --extra inference --extra datasets
```

Prepare an external [S1MB](https://github.com/hotchpotch/S1MB) checkout and its
dataset using that project's setup instructions. Its evaluator environment
needs the `open-models` extra:

```sh
cd /path/to/S1MB/evaluator
uv sync --locked --extra open-models
# Pin category_sources.japanese-v1.revision in dataset-source.json to the YAML revision.
uv run --no-sync python scripts/fetch_dataset.py --category japanese-v1
uv run --no-sync s1mb check-data --category japanese-v1
```

### Recipes

Training is configured with YAML recipes. The four recipes in `configs/` cover
Qwen3.5 0.8B / 4B / 9B and LLM-jp 4.1 8B (`configs/llmjp41-8b.yaml` is the
saihai-v1-8b recipe), plus source-family holdout variants. They pin the model
revisions and use the `train` split of `hotchpotch/saihai-v1-dataset`.

The optimization recipe is BF16 LoRA, an FP32 candidate head initialized from
the original LM-head rows, masked soft-target cross entropy, separate
backbone/head learning rates, AdamW, gradient clipping and a warmup/decay
schedule. Batch size, accumulation, epochs, LoRA settings, input limit and smoke
settings are in YAML.

### Run training

Run from this checkout, exposing one available GPU:

```sh
# Smoke run
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config configs/llmjp41-8b.yaml --output /path/to/new-smoke-run \
  --smoke --s1mb-root /path/to/S1MB --eval-limit 2

# 10% run
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config configs/qwen35-9b.yaml --output /path/to/new-ten-percent-run \
  --fraction 0.1 --batch-size 1 --gradient-accumulation 16 \
  --s1mb-root /path/to/S1MB
```

| Option | Effect |
| --- | --- |
| *(neither `--smoke` nor `--fraction`)* | One full training epoch as configured |
| `--smoke` | 24 short training cases across all three decision types, repeated for ten optimizer steps |
| `--fraction 0.1` | Selects parent groups deterministically within source/task strata; the row count need not equal exactly 10% |
| `--max-steps N` | Bounds either run |
| `--eval-limit 2` | Evaluates two cases per benchmark (smoke); omit it to evaluate all cases |

A run directory must be new. Overlong inputs are excluded with recorded IDs,
without truncation. Smoke checks do not establish full-run memory capacity or
quality.

After training, the CLI saves a checkpoint and a merged safetensors export,
including the [standalone Transformers runtime](#transformers-standalone-remote-code),
then runs S1MB `japanese-v1` and validates every result. Evaluation never
selects training data or tunes the model. The S1MB Japanese data must match the
YAML evaluation revision; using an updated snapshot requires deliberately
updating that setting. Saved artifacts include selected IDs, dataset receipts,
logs and checkpoint / export hashes.

> [!NOTE]
> The training dataset includes train-split data from source families evaluated
> by S1MB `japanese-v1`. Disclose that exposure with results, or exclude those
> families with the `exclude_s1mb_japanese_source_family` filter (see
> [docs/s1mb-japanese-source-family-holdout.md](docs/s1mb-japanese-source-family-holdout.md)).

### Training data

`dataset.revision: latest` resolves to the dataset Hub repository's `main`
commit at the start of each run. All artifacts are downloaded at that same
commit, and `data-receipt.json` records both the requested and resolved
revisions. Specify a full commit ID to reproduce a run.

The published dataset has 118 subset configs with 33,532 training cases. The
loader reads every config's train shard, verifies each checksum and count, and
merges by original `source_order`, preserving gaps from exclusions. Validation
shards are excluded from training. Older dataset revisions that use a single
config can still be loaded by specifying their commit.

### Validation

Validation uses the same resolved dataset commit as training and the separate
validation split.

- `validation_per_subset: 3` selects up to three fixed cases per subset with the
  configured seed; identity, parent and group overlap are rejected.
- `validation_runs: 10` distributes ten evaluations across optimizer steps,
  including the final step. Shorter runs evaluate at every available step;
  smoke runs use ten steps and exercise all ten validations.
- Aggregate example-weighted soft-target cross entropy and per-subset losses are
  saved in `validation-history.json`; selected IDs and overflow exclusions are
  recorded separately.

The 9B recipe also runs a long-input forward/backward memory probe before
training, without updating weights. In smoke mode it probes the longest eligible
input among the 64 largest rendered JSON payloads; full runs probe the longest
tokenized training input.

### Input protocol

Models trained by this repository use the `saihai-pplx-json-v1` input protocol.
It selects candidate codes exactly as llama.cpp's `pplx-decider`: A..Z followed
by single-token AA..ZZ codes, capped at 255. It preserves Choice order, aligns
Noul inputs and soft targets to false/true, and uses `option_i` Score labels
with authored numeric values. Input JSON numbers use the native renderer's six
significant digits in both Python and Jinja; soft-target probabilities are
unchanged.

Checkpoints trained with a different protocol use different codes. Do not swap
a checkpoint's codebook without retraining or a separately verified migration.

### Naming note

S1MB `japanese-v1` was formerly called JDMB, so dataset fields such as
`jdmb_source_family` and subset IDs such as `jp-jdmb-*` keep that name.
`exclude_jdmb_source_family` is accepted as an alias of
`exclude_s1mb_japanese_source_family`.

## Export and conversion

### Native GGUF files

To build a BF16 GGUF plus Q8_0, Q6_K and Q4_K_M variants from a training run,
run the release script from this checkout. Build `llama-quantize` first, or
supply its location with `--quantize-binary`. The output directory must be new.

```sh
uv run --no-sync python scripts/export_gguf.py \
  --llama-cpp /path/to/llama.cpp --export /path/to/run/export \
  --vocab-gguf /path/to/official-base-model.gguf \
  --output-dir /path/to/new-gguf-export --name saihai-8b
```

The vocabulary reference (`--vocab-gguf`) is required for LLM-jp; omit it for
other tokenizers. Pass `--quantizations` with no values for BF16 only. The
script verifies the preserved native decision/tokenizer metadata in every
variant and saves file hashes in `gguf-export.json`. Quantization uses
llama.cpp's default mixed precision, including the output head, without an
importance matrix.

Validate each output with native `/v1/systemone` before distributing it;
conversion alone does not establish inference parity or accuracy. Run the
result as in [llama.cpp native `/v1/systemone`](#llamacpp-native-v1systemone).
Exports support the model's codebook capacity (up to 255 candidates), with
Score limited to 2–10 levels.

### BF16 only

The underlying conversion module can also be run directly:

```sh
python -m saihai.convert_gguf --llama-cpp /path/to/llama.cpp \
  --export /path/to/run/export --output /path/to/model-BF16.gguf
```

For LLM-jp, also pass `--vocab-gguf /path/to/official-base-model.gguf` to reuse
and validate its official Unigram vocabulary; it is needed only for conversion.
The command uses upstream conversion, embeds the trained `systemone` template
and readout profile, and verifies that GGUF tensors are unchanged. The
`.raw.gguf` conversion output is kept beside the final GGUF.

### Embed a decision contract into an existing GGUF

`saihai.gguf` embeds a `decision-inference.json` contract into a GGUF without
changing tensors (every tensor's bytes are verified):

```sh
python -m saihai.gguf --llama-cpp /path/to/llama.cpp \
  --contract decision-inference.json --input model.gguf --output model-native.gguf \
  --systemone-format pplx
```

| `--systemone-format` | Use |
| --- | --- |
| `pplx` | Models trained with `saihai-pplx-json-v1` (current; up to 255 candidates) |
| `pplx-26` (default) | Older exports whose first 26 codes are A..Z; at most 26 candidates |
| `saihai` | Full learned codebook and extended API; requires a custom saihai C++ runtime, not stock llama.cpp |

The native tokenizer must reproduce the training token IDs; metadata alone
cannot fix a tokenizer mismatch.

### Change the inference context of an export

To change the inference limit of an existing export without changing its
weights or learned prompt:

```sh
uv run --no-sync python -m saihai.context --export ./model \
  --output ./model-64k --max-input-tokens 65536
```

The command validates against the base model's `max_position_embeddings`, keeps
the training limit recorded separately and creates a new export. Then re-embed
the updated `decision-inference.json` into each GGUF with `python -m saihai.gguf`
and `--systemone-format pplx`. Changing only a server flag updates neither the
Python contract nor the GGUF decision metadata.

### Add the Transformers runtime to an older export

To add `inference_saihai.py` to an export that lacks it, without changing
weights:

```sh
python -m saihai.export_inference ./model
```

The helper copies `inference_saihai.py` unchanged and adds the `AutoModel` entry
to `config.json`. The test suite checks that the runtime works from export files
alone and returns the same probabilities as the package backend.

## Dataset loading utilities

The dataset utilities require Python 3.12 and Hugging Face Datasets 5.0.1
(`datasets==5.0.1`, or the `datasets` extra).

```python
from saihai.datasets import verify_release, load_subset, iter_records

verify_release("dataset-directory")
dataset = load_subset("dataset-directory", "config-name")
for decision in iter_records("dataset-directory", "train"):
    model_input = decision["input"]
    soft_targets = decision["target"]["probabilities"]
```

A local release consists of a checksummed manifest and independent DatasetDicts
saved with `save_to_disk`; corresponding Parquet files back the Hub configs.
Metadata and targets are not model inputs. Keep dataset provenance and
source-specific license requirements; the code license does not grant rights to
redistribute data.

## Repository layout

| Path | Contents |
| --- | --- |
| `saihai/` | Training, inference, llama.cpp client/server, GGUF conversion and S1MB evaluation |
| `saihai/inference_saihai.py` | Standalone Transformers runtime copied into model exports |
| `configs/` | Training recipes for LLM-jp 4.1 8B and Qwen3.5 0.8B / 4B / 9B, with source-family holdout variants |
| `scripts/export_gguf.py` | Release GGUF export and quantization |
| `docs/` | Training and evaluation guides (Japanese) |
| `tests/` | Unit tests (`uv run pytest`) |

## License

The code in this repository is licensed under Apache-2.0; see [LICENSE](LICENSE).
Released model weights follow their model cards. Training data keeps its
source-specific licenses; see the
[dataset card](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset).
