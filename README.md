# saihai

Portable data and inference for typed decision models.

See the [Japanese training and evaluation guide](docs/training-and-evaluation.md)
for reproducible training, validation and S1MB Japanese evaluation.

## Python System One interface

Install from a checkout with `pip install '.[inference]'` for PyTorch inference,
or `pip install .` for a separately managed llama-server. Released exports use
the same `state` / `questions` request and `answers` response in both backends.

```python
from saihai import Saihai

model = Saihai.from_pretrained("owner/model", token=True, device="cuda")
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
question can use `model.predict(state, question)`. Multiple questions are
evaluated independently. The inference limit is stored in the model contract;
overflow is rejected rather than truncated.

For GGUF, start the matching model in llama-server, then use
`Saihai.from_pretrained("owner/gguf-model", backend="llama.cpp",
backend_url="http://127.0.0.1:8080", token=True)`.
This loads the decision contract and tokenizer, not the GGUF weights again.
With a backend whose native tokenizer matches the training token IDs, pass
`native_tokenizer=True` to use GGUF tokenization without an external JSON.
In that mode the Hub client downloads only the decision contract.

The optional request `model` field must match the loaded contract. The response
reports the loaded model. Confidence and Score follow the mapping described
below; numeric Score levels are preserved through explicit extensions.

## Dataset loading

The dataset utilities require Python 3.12 and Hugging Face Datasets 5.0.1.
Install `datasets==5.0.1` in your environment. No experiment workspace is required.

```python
from saihai.datasets import verify_release, load_subset, iter_records

verify_release("dataset-directory")
dataset = load_subset("dataset-directory", "config-name")
for decision in iter_records("dataset-directory", "train"):
    model_input = decision["input"]
    soft_targets = decision["target"]["probabilities"]
```

A local release consists of a checksummed manifest and independent DatasetDicts
saved with `save_to_disk`. Corresponding Parquet files can support Hub configs.
Keep dataset provenance and source-specific license requirements; code packaging
does not grant rights to redistribute data. Metadata and targets are not model inputs.

## llama.cpp decision inference

### Native `/v1/systemone`

For models whose first 26 decision codes are A through Z, the default export
uses llama.cpp's existing `pplx-decider` readout and a model-specific Jinja
template. This is a compatible readout profile; the model remains saihai.
No saihai-specific C++ changes are needed. For LLM-jp tokenization, use a build
with the tokenizer fix from llama.cpp PR #28864 until that fix is upstream.

```sh
python -m saihai.gguf --llama-cpp /path/to/llama.cpp \
  --contract decision-inference.json --input model.gguf --output model-native.gguf \
  --systemone-format pplx-26
llama-server -m model-native.gguf --alias my-model --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --parallel 1 --no-context-shift
```

The exporter preserves and verifies every tensor's bytes. The GGUF embeds the
learned input format and named `systemone` template. Runtime requires only the
GGUF and server, without an external tokenizer JSON, contract or Python frontend.
The native tokenizer must reproduce training token IDs; metadata alone cannot
fix a tokenizer mismatch. Keep the per-slot context at the contract's limit.

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

Use standard `criteria`, with at most 26 candidates per question. Choice preserves
criterion order and uses keys as input labels. Noul always uses false then true,
with those strings as default descriptions. Score uses `option_i` labels and
returns the expected level index as `score`. Levels may contain `value` and
`description`; compute an expected numeric value client-side from probabilities
and those values when the authored scale differs from level indices.

The template rejects more than 26 candidates. Existing llama.cpp reports
Jinja template errors as HTTP 500; preflight the limit in your client (for
benchmark adapters, set `max_candidates=26`). Its Jinja float serializer has
limited precision: common levels such as 0.0, 0.25, 0.5 and 1.0 are supported,
but floats that cannot reproduce Python's training JSON representation are
rejected rather than rounded. Represent precise quantities in state as strings.
The stock native API does not implement the custom `options`, `expected_value`
or `cache_prompt: false` extensions. Cache and batching conditions can affect
quantized numerical results.

The optional `--systemone-format saihai` export retains the full learned codebook
and requires the corresponding custom saihai C++ runtime. It supports larger
candidate sets and the extended API; it is not the default compatibility path.

### Candidate readout client

`LlamaDecisionClient` reads a model-bound `decision-inference.json` contract and
uses an unmodified `llama-server`. Export the merged backbone as a standard HF
`LlamaForCausalLM`, write the trained decision rows into their original LM-head
candidate-token rows, then convert to GGUF. Keep the exact training prompt,
codebook, tokenizer and context limit in the contract. The conversion must be
validated against the original checkpoint; merge and precision changes can
change probabilities.

The inference client needs only the Python standard library and `tokenizers`
when the contract supplies `tokenizer_file`. It does not load Torch or model
weights. For an LLM-jp contract, keep its original `tokenizer.json` beside the
contract: the tested llama.cpp Unigram tokenizer did not reproduce byte-fallback
newlines. The client uses the saved tokenizer and passes exact token IDs to the
GPU backend. A standalone GGUF string prompt is not equivalent until its
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

Run the typed HTTP frontend separately from llama-server. Bind both processes to
localhost or a machine's Tailscale IPv4 address. Select the GPU explicitly when
starting llama-server and keep its context slot large enough for the contract.

```sh
PYTHONPATH=. python -m saihai.server \
  --backend-url http://127.0.0.1:8080 \
  --contract /path/to/decision-inference.json \
  --host 127.0.0.1 --port 8098
```

`POST /v1/systemone` accepts `state` and named `questions` with `type`,
`instructions` and `criteria`. Choice uses an ordered object; Noul uses
`false`/`true` definitions; Score uses a level array. Noul defaults to false then
true. Score entries can be objects containing `value` and `description`.
For exact benchmark inputs, a question can instead supply an ordered `options`
array with `id`, `label`, `description`, and a numeric `value` for Score. This
preserves labels, definitions, order and numeric levels independently. Transport
IDs are not included in model text.

Choice returns a winner and full probability mapping. Noul returns the true
probability. Score returns probabilities indexed by level and the expected
level index as `score`, matching the common System One interface. The
`expected_value` and `numeric_values` extensions preserve authored numeric
scales, including irregular levels. Confidence uses the current llama.cpp
formulas: normalized maximum probability for Choice, and normalized distance
from the mode for Score. Temperature is fixed at 1; no calibration is claimed.

The backend performs a one-token completion restricted by token-ID grammar,
with all sampling filters disabled except unit temperature. All declared
candidate probabilities must be present. Missing candidates and overflow fail;
there is no uniform fallback. No reasoning or free text is generated. The API
reports zero user-facing output tokens and separately reports its internal
one-token-per-question readout in `backend_usage`. This frontend is not native
llama.cpp `/v1/systemone` support: registering a new decision type in upstream
is a separate integration step. Do not disguise the model as OpenJev or Nimble.

## YAML training

Install the training dependencies with Python 3.12:

```sh
uv sync --python 3.12 --locked --extra training --extra inference --extra datasets
```

The four recipes in `configs/` cover Qwen3.5 0.8B / 4B / 9B and LLM-jp 4.1 8B.
They pin the model revisions and use `dataset.revision: latest` with the `train` split of
`hotchpotch/saihai-v1-dataset` (merged-no-3000), and keep the pre-v4 optimization
recipe: BF16 LoRA, FP32 candidate head initialized from original LM-head rows,
masked soft-target cross entropy, separate backbone/head learning rates, AdamW,
gradient clipping and a warmup/decay schedule. Batch size, accumulation, epochs,
LoRA settings, input limit and smoke settings are in YAML.

Validation uses the same resolved dataset commit as training and the separate
validation split. `validation_per_subset: 3` selects up to three fixed cases per
subset with the configured seed; identity, parent and group overlap are rejected.
`validation_runs: 10` distributes ten evaluations across optimizer steps,
including the final step. Shorter runs evaluate at every available step. Both
aggregate example-weighted soft-target cross entropy and per-subset losses are
saved in `validation-history.json`; selected IDs and overflow exclusions are
recorded separately. Smoke runs use ten steps and exercise all ten validations.
The 9B recipe also runs a long-input forward/backward memory probe before training
without updating weights. In smoke mode it probes the longest eligible input
among the 64 largest rendered JSON payloads; full runs probe the longest tokenized
training input.

At the start of each run, `latest` resolves to the dataset Hub repository’s
`main` commit. All artifacts are downloaded at that same commit, and
`data-receipt.json` records both the requested and resolved revisions. A full
commit ID can also be specified for reproduction.

The current published dataset has 118 subset configs with 33,532 training
cases. The loader reads every config’s train shard, verifies each checksum and
count, and merges by original `source_order`, preserving gaps from exclusions.
Validation shards are excluded from training. The previous single-config layout
remains supported when an older revision is explicitly selected.

The new `saihai-pplx-json-v1` protocol selects codes exactly as the existing
llama.cpp `pplx-decider`: A..Z followed by single-token AA..ZZ codes, capped at
255. It preserves Choice order, aligns Noul inputs and soft targets to false/true,
and uses `option_i` Score labels with authored numeric values. Input JSON numbers
use the native renderer's **six significant digits** in both Python and Jinja;
soft target probabilities remain unchanged. Store precise input quantities as
strings. This protocol differs from existing exports; do not replace an old
checkpoint's codebook without training or a separately verified migration.

Prepare an external S1MB checkout and its dataset using that project's setup
instructions. Its evaluator environment needs the `open-models` extra:

```sh
cd /path/to/S1MB/evaluator
uv sync --locked --extra open-models
# Pin category_sources.japanese-v1.revision in dataset-source.json to the YAML revision.
uv run --no-sync python scripts/fetch_dataset.py --category japanese-v1
uv run --no-sync s1mb check-data --category japanese-v1
```

Run from this checkout, exposing one available GPU:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config configs/llmjp41-8b.yaml --output /path/to/new-smoke-run \
  --smoke --s1mb-root /path/to/S1MB --eval-limit 2

CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config configs/qwen35-9b.yaml --output /path/to/new-ten-percent-run \
  --fraction 0.1 --batch-size 1 --gradient-accumulation 16 \
  --s1mb-root /path/to/S1MB
```

Omit `--smoke` and `--fraction` for one full training epoch as configured.
`--max-steps` can bound either run. Smoke selects 24 short training cases across
all three decision types and repeats them for ten optimizer steps. Fractional
runs select parent groups deterministically within source/task strata; their
exact row count need not equal the fraction times the total. Overlong inputs
are excluded with recorded IDs, without truncation. A run directory must be new.

After training, the CLI saves a checkpoint and merged safetensors export, then
runs S1MB `japanese-v1` and validates every result. Evaluation never selects training data
or tunes the model. S1MB Japanese data must match the YAML evaluation revision; an updated
snapshot requires deliberately updating that setting. `--eval-limit 2` is a
smoke per benchmark, while omitting it evaluates all cases. This training dataset
includes S1MB `japanese-v1` source-family train data; disclose that exposure with results.
JDMB is the former benchmark name. Dataset fields such as `jdmb_source_family`
and subset IDs such as `jp-jdmb-*` retain that name for compatibility. The current
filter option is `exclude_s1mb_japanese_source_family`;
`exclude_jdmb_source_family` remains supported as a legacy alias.
Saved artifacts include selected IDs, dataset receipts, logs and checkpoint /
export hashes. Smoke checks do not establish full-run memory capacity or quality.

### Convert a newly trained model to native GGUF

For a BF16 export plus Q8_0, Q6_K and Q4_K_M variants, run the release script
from this checkout. Build `llama-quantize` first, or supply its location with
`--quantize-binary`. The output directory must be new.

```sh
uv run --no-sync python scripts/export_gguf.py \
  --llama-cpp /path/to/llama.cpp --export /path/to/run/export \
  --vocab-gguf /path/to/official-base-model.gguf \
  --output-dir /path/to/new-gguf-export --name saihai-8b
```

The vocabulary reference is required for LLM-jp; omit it for other tokenizers.
Use `--quantizations` with no values for BF16 only. The script verifies preserved
native decision/tokenizer metadata in every variant and saves file hashes in
`gguf-export.json`. Quantization uses llama.cpp's default mixed precision,
including the output head, without an importance matrix. Validate each output
with native `/v1/systemone` before distributing it; conversion alone does not
establish inference parity or accuracy.

For BF16 only, the underlying module is also available:

```sh
python -m saihai.convert_gguf --llama-cpp /path/to/llama.cpp \
  --export /path/to/run/export --output /path/to/model-BF16.gguf
```

For LLM-jp, also pass `--vocab-gguf /path/to/official-base-model.gguf` to reuse
and validate its official Unigram vocabulary. This reference is needed only
for conversion. The command uses upstream conversion, embeds the trained
`systemone` template and readout profile, and verifies unchanged GGUF tensors.
The `.raw.gguf` conversion output is retained beside the final GGUF.

```sh
llama-server -m /path/to/model-BF16.gguf --alias my-model \
  --host 127.0.0.1 --port 8080 -ngl 99 -c 8192 --parallel 1 --no-context-shift
```

Use the standard `/v1/systemone` request shown above. LLM-jp still requires its
native tokenizer fix; saihai-specific C++ code is unnecessary. New trained
exports support the selected codebook capacity (up to 255), with Score limited
to 2..10 levels. No external JSON or Python frontend is needed at inference.


### LLM-jp 8B inference context

The LLM-jp recipe trains with `training.max_length: 8192` and exports with
`evaluation.max_input_tokens: 65536`, matching the base model's positional capacity.
Training was performed with an 8,192-token input limit. Although inference accepts
up to 65,536 tokens, performance on long inputs, especially near 64K tokens, may
be poor and has not been established by the short-context benchmark results.

To upgrade an existing export without changing its weights or learned prompt:

```sh
uv run --no-sync python -m saihai.context --export ./model \
  --output ./model-64k --max-input-tokens 65536
```

The command validates the base model's `max_position_embeddings`, preserves the
training limit separately and creates a new export. Re-embed the updated
`decision-inference.json` into each GGUF with `python -m saihai.gguf`, using
`--systemone-format pplx` for the current model. Updating only a server flag does
not update the Python contract or GGUF decision metadata.

For native LLM-jp inference, use the tokenizer-corrected llama.cpp build:

```sh
llama-server -m ./model-64k.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 65536 --parallel 1 -b 8192 -ub 512 \
  --flash-attn on --no-context-shift
```

The tested llama-server reserves one context position: `-c 65536` accepts
up to 65,535 input tokens per slot. Python inference accepts 65,536 tokens.
Context size is divided between slots: four slots with 64K capacity each require
`-c 262144 --parallel 4`. KV-cache memory grows with total context; allocate slots
according to available GPU memory. Quantization reduces model-weight memory,
while KV-cache precision is a separate setting. The batch size above controls
prompt processing, not the request context limit. Requests above the model or
per-slot limit must be rejected rather than silently truncated.
