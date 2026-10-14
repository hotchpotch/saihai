# saihai の学習と評価

saihai は、LLM をベースに Choice / Noul / Score の候補確率を返す Typed Decision Model を学習します。この文書では、配布された YAML を使う学習方法と、2026-10-09 に開始した4モデルの測定を追試する手順を説明します。コマンドは、特に指定がなければ saihai リポジトリのルートで実行します。

## 1. 学習するもの

通常の文章生成を教師にするのではなく、各事例に付いた候補の確率分布を教師にします。入力には状態、指示、候補の定義を含め、正答や教師確率、出典 ID を渡しません。

処理は次の順序です。

1. データセットの `train` を読み込み、候補と soft target の対応を検証する。
2. 候補ごとに1 token の出力コードを割り当て、状態・指示・候補を JSON にする。
3. ベースモデルの chat template に JSON を埋め込み、入力 token 列を作る。
4. 入力の最後の有効 token の hidden state を取り出す。
5. 学習用 head で候補の logits を計算し、有効な候補内で softmax を取る。
6. 教師確率との cross entropy を最小化する。
7. LoRA をベースモデルに merge し、head の重みを LM head のコード token 行に書き戻して、標準の safetensors モデルを保存する。

推論も入力を1回 forward して候補確率を取り出します。出力文や reasoning 文の生成・解析は行いません。

### 入力 token と候補コード

`prompt_version` は `saihai-pplx-json-v1` です。llama.cpp の `pplx-decider` と同じ規則で、`A`〜`Z`、続いて `AA`〜`ZZ` を調べ、そのモデルの tokenizer で1 token になるコードだけを採用します。上限は255候補です。コードと token ID は `decision-inference.json` に保存します。

モデルに渡す JSON は、概念的には次の形です。

```json
{
  "instruction": "適切な担当を選んでください。",
  "state": "決済処理でエラーが発生しました。",
  "type": "choice",
  "options": [
    {"code": "A", "label": "billing", "definition": "請求・返金"},
    {"code": "B", "label": "technical", "definition": "障害・技術的な問題"}
  ]
}
```

実際の JSON は [protocol.py](../saihai/protocol.py) の renderer が作成します。数値は llama.cpp の minja に合わせた有効数字6桁の表現を使います。これは入力 JSON の規則で、教師確率を6桁に丸めるものではありません。入力値に6桁を超える精度が必要な場合は、文字列として扱います。

Choice は候補順を保存します。Noul は `false` → `true` の順にそろえ、教師確率も同じ順に並べ替えます。Score は2〜10候補で、ラベルを `option_0` などにそろえ、元の数値 `value` を `numeric_value` として保持します。

LLM-jp は YAML の `reasoning_effort` と `conversation_start_date` を使い、chat template の生成開始位置に `<|channel|>final<|message|>` を加えます。Qwen3.5 は `enable_thinking=False` で template を適用します。prefix / suffix は export とともに保存されるため、推論でも学習と同じ renderer を使います。

### Head と loss

backbone は BF16、候補 head は bias のない FP32 の線形層です。head の初期値には、ベースモデルの LM head の候補コード token 行をコピーします。最後の有効 token の hidden state を FP32 に変換して head に渡します。

事例の候補数を `K`、教師確率を `q`、head の logits を `z` とすると、loss は次の式です。

```text
p_i = softmax(z_1, ..., z_K)_i
loss = -sum_i q_i * log(p_i)
```

padding された候補と、その事例で使わない head 行は softmax の対象から外します。batch と gradient accumulation 内では事例数で重み付けします。教師の soft target を argmax の hard label に置き換えません。

学習対象は LoRA と候補 head です。ベースの embedding を含む backbone の元の重みは固定します。LoRA は attention と MLP の projection に適用します。Qwen3.5 の線形 attention に必要な projection も含みます。

## 2. 依存関係と設定

Python 3.12 と uv、BF16 を扱える CUDA GPU を用意します。Linux 用の lock は PyTorch の CUDA 13.0 wheel を選ぶため、対応する NVIDIA driver が必要です。学習は CPU fallback を行いません。

```sh
uv sync --python 3.12 --locked --extra training --extra inference --extra datasets
```

主な固定依存関係は PyTorch 2.10.0、Transformers 5.17.0、datasets 5.0.1、kernels 0.16.0 です。正確な依存関係は `uv.lock` を使います。Qwen3.5 の kernel 設定は [training_model.py](../saihai/training_model.py) にあります。すべての演算が専用 kernel になるとは限りません。

4つの recipe を同梱しています。

| YAML | ベースモデル | LoRA rank / alpha | backbone LR | head LR | batch × accumulation |
|---|---|---:|---:|---:|---:|
| [qwen35-0.8b.yaml](../configs/qwen35-0.8b.yaml) | Qwen/Qwen3.5-0.8B | 16 / 32 | 0.0002 | 0.0002 | 2 × 8 |
| [qwen35-4b.yaml](../configs/qwen35-4b.yaml) | Qwen/Qwen3.5-4B | 32 / 64 | 0.0001 | 0.0001 | 2 × 8 |
| [qwen35-9b.yaml](../configs/qwen35-9b.yaml) | Qwen/Qwen3.5-9B | 32 / 64 | 0.00005 | 0.0001 | 1 × 16 |
| [llmjp41-8b.yaml](../configs/llmjp41-8b.yaml) | llm-jp/llm-jp-4.1-8b-thinking | 16 / 32 | 0.00005 | 0.0001 | 2 × 8 |

共通設定は1 epoch、実効 batch size 16、最大入力長8,192、seed 42、LoRA dropout 0、AdamW、weight decay 0.01、gradient norm の上限1です。gradient checkpointing を有効にし、入力長が近い事例を128件単位の bucket でまとめます。

learning rate は最初の約5%で warmup し、その後は設定 LR の約20%に向けて減衰します。正確な式は [training.py](../saihai/training.py) の `learning_rate_factor` にあります。LoRA rank、LR、batch、accumulation、epochs、validation 回数などは YAML で管理します。

### データの読み込みと revision

学習データは [hotchpotch/saihai-v1-dataset](https://huggingface.co/datasets/hotchpotch/saihai-v1-dataset) の `train` です。現在の複数 config 形式では、各 subset の train shard を読み込み、manifest・Parquet の checksum、件数、ID、入力と教師確率の対応を検証します。事例は元の `source_order` で統合し、除外に伴う番号の欠けを許容します。旧単一 config 形式も扱えます。

配布 YAML の `dataset.revision: latest` は、実行開始時に Hub の `main` の commit ID へ一度だけ解決します。その実行の全ファイルを同じ commit から読み、validation にも同じ commit を使います。

最新データでの再学習には `latest` を使います。同じ測定を追試する場合や複数モデルを比較する場合は、全 recipe を同じ full commit ID に固定します。`latest` のまま別時刻に起動すると、モデル間でデータ revision が変わる可能性があります。

第8節の測定で使った固定値は次のとおりです。S1MB の準備手順は第3節にあります。

| 対象 | repository / revision |
|---|---|
| 学習データ | `hotchpotch/saihai-v1-dataset` / `906504ac4b72e133a613011318afd8206fb5e8ec` |
| 評価データ | [`hotchpotch/S1MB-japanese-v1`](https://huggingface.co/datasets/hotchpotch/S1MB-japanese-v1) / `59b66dd94c430c1ec6db1c948fa2cc0410400a27` |

モデルの revision は各 YAML に固定されています。saihai のコード revision と `uv.lock`、実行に使った recipe も保存してください。seed を固定しても、演算 backend や依存関係の差によって浮動小数の結果が変わるため、数値の完全一致までは保証しません。

追試用 recipe のコピーを作る例です。配布 YAML の `latest` は変更しません。

```sh
uv run --no-sync python - <<'PY'
from pathlib import Path
import yaml

train_revision = "906504ac4b72e133a613011318afd8206fb5e8ec"
eval_revision = "59b66dd94c430c1ec6db1c948fa2cc0410400a27"
output = Path("recipes/reproduction")
output.mkdir(parents=True, exist_ok=True)
for name in ("llmjp41-8b", "qwen35-0.8b", "qwen35-4b", "qwen35-9b"):
    config = yaml.safe_load((Path("configs") / f"{name}.yaml").read_text())
    config["dataset"]["revision"] = train_revision
    config["evaluation"]["revision"] = eval_revision
    (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

## 3. S1MB Japanese の準備

学習 CLI は終了後に外部 S1MB checkout の evaluator 環境で `japanese-v1` を評価します。
評価データ [hotchpotch/S1MB-japanese-v1](https://huggingface.co/datasets/hotchpotch/S1MB-japanese-v1) は公開されています。recipe の revision と同じ snapshot を準備します。

```sh
SAIHAI_ROOT="$(pwd)"
git clone https://github.com/hotchpotch/S1MB.git evaluation/S1MB
S1MB_ROOT="$SAIHAI_ROOT/evaluation/S1MB"
cd "$S1MB_ROOT"
git checkout 7c394a9d0a2c8c7e89564fb850795f88257185bc
cd evaluator
uv sync --python 3.12 --locked --extra open-models
uv run --no-sync python - <<'PYSETUP'
import json
from pathlib import Path
path = Path("dataset-source.json")
config = json.loads(path.read_text())
config["category_sources"]["japanese-v1"]["revision"] = "59b66dd94c430c1ec6db1c948fa2cc0410400a27"
path.write_text(json.dumps(config, indent=2) + "\n")
PYSETUP
uv run --no-sync python scripts/fetch_dataset.py --category japanese-v1
uv run --no-sync s1mb check-data --category japanese-v1
cd "$SAIHAI_ROOT"
```

別の snapshot を使う場合は S1MB の定義、データ、recipe の `evaluation.revision` をそろえます。
評価は offline で `data/datasets/hub-sources/japanese-v1.json` の repo ID・category・revision を検証します。

## 4. Smoke → 10% → 全件学習

1プロセスから見える GPU は1台にします。`CUDA_VISIBLE_DEVICES=0` は使用可能な GPU を選ぶ例で、GPU 番号は利用環境に合わせます。各実行には、まだ存在しない出力ディレクトリを指定してください。

まず1モデルを smoke で確認します。

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config recipes/reproduction/llmjp41-8b.yaml \
  --output runs/smoke/llmjp41-8b \
  --smoke --s1mb-root "$S1MB_ROOT" --eval-limit 2
```

smoke は Noul / Choice / Score から入力 JSON が短いものを各8件、計24件選びます。配布設定ではこれを繰り返し、10 optimizer steps、実効 batch 16、合計160事例の更新を行います。validation は本学習と同じ固定セットで10回測定し、S1MB Japanese は各 benchmark の2件だけを評価します。ほかの3モデルでも `--config` と `--output` を変えて確認します。

10% の学習を試す場合は `--fraction 0.1` を使います。

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config recipes/reproduction/qwen35-9b.yaml \
  --output runs/ten-percent/qwen35-9b \
  --fraction 0.1 --batch-size 1 --gradient-accumulation 16 \
  --s1mb-root "$S1MB_ROOT"
```

fraction は source / task ごとに parent 単位で決定論的に選びます。行を単純に先頭10%切り出すものではなく、正確な事例数は全体の10%と一致するとは限りません。この例は学習を10%にしますが、S1MB Japanese は全件を評価します。

全件学習は `--smoke` と `--fraction` を指定しません。

```sh
CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m saihai.training \
  --config recipes/reproduction/qwen35-9b.yaml \
  --output runs/full/qwen35-9b \
  --s1mb-root "$S1MB_ROOT"
```

`--max-steps` で optimizer steps の上限を設定できます。`--batch-size` と `--gradient-accumulation` は YAML の値を上書きします。実効 batch size は両者の積です。`--smoke` と `--fraction` は同時に指定できません。

最大入力長を超える学習事例は、切り詰めず除外 ID を保存します。今回の固定 snapshot では、全4モデルが33,532件を除外なしで1 epoch学習し、2,096 optimizer stepsを実行しました。

### メモリと2 GPU の運用

9B recipe は batch size 1、gradient accumulation 16、gradient checkpointing、`expandable_segments:True` を使用します。さらに `memory_probe: true` で、パラメータを更新せず長入力の forward/backward を確認します。

本学習では token 化済み train の最長事例を probe します。smoke では入力 JSON が大きい上位64事例のうち、上限内で token 数が最大のものを使います。probe は optimizer 更新を含まないため、その数値だけで本学習の全メモリ使用量を判断しません。

2 GPU を使う場合は、異なるモデルを別々の1 GPU プロセスで起動します。たとえば9Bを `CUDA_VISIBLE_DEVICES=0`、LLM-jpを `CUDA_VISIBLE_DEVICES=1` に割り当て、それぞれ異なる出力先を指定します。4モデルの smoke を先に確認し、その後の全件学習・評価が終わって空いた GPU に4B、0.8Bを順に投入できます。

短い smoke 入力の速度は、全件の長入力を含む速度と一致しません。実際のステップ時間を確認して割り当てを調整します。同じ GPU で学習と別モデルの評価を同時に動かさず、学習後の S1MB Japanese 評価も含めて1ジョブとして扱います。

## 5. Validation loss

validation は train と同じ dataset revision の `validation` split を使います。train 全体との ID・parent・group の重複を検査し、重複があれば停止します。

選び方は、IDで並べた事例を subset ごとに分け、seed 42 の乱数で固定選択するものです。`validation_per_subset: 3` は各 subset の最大3件を意味します。3件未満なら存在する件数を使い、validation がない subset には事例を補いません。学習と同じ入力長検査を行い、上限を超えた validation 事例は除外 ID を保存します。

今回の snapshot は validation 8,729件で、validation のある93 subsetから3件ずつ、279件を選びました。全4モデルで選択 ID が同じで、入力長による除外はありませんでした。

`validation_runs: 10` は optimizer steps 全体に均等に配置し、最終ステップも含めます。総ステップ数を `T`、評価回数を `V=min(T, 10)` とすると、`ceil(T*i/V)`（`i=1..V`）で測定します。総ステップ数が10未満なら、実行可能な各ステップで測定します。

validation は `eval()` と inference mode で、batch size 1、同じ masked soft-target cross entropy を計算します。主値は事例ごとの loss の平均です。subset ごとの平均 loss と件数も保存します。終了後に元の train mode に戻します。

validation による early stopping や最良 checkpoint の選択は行いません。学習終了時の checkpoint を評価します。smoke の loss は疎通確認として扱い、全件学習の精度とは区別します。

## 6. S1MB Japanese 評価

学習 CLI は LoRA checkpoint と merge 済みの safetensors を保存し、その export を使って `saihai.s1mb_eval` を実行します。学習時の FP32 head は export 時に BF16 の LM head 行へ書き戻すため、validation と export の推論では通常の浮動小数差が生じ得ます。

この評価は Transformers backend で行います。GGUF・量子化モデル・llama.cpp backend の数値を測った結果とは区別します。

S1MB Japanese の入力には状態・指示・候補定義・Score の元の数値を使います。正答、出典 ID、評価 family を入力に加えません。Noul の候補順は `false` → `true` とし、予測確率は S1MB Japanese の元の option ID に対応付けます。文を生成せず、候補 LM-head 行の logits を softmax にします。

`--eval-limit 2` は benchmark ごとに最初の2事例を使う部分評価です。指定しなければ全件を評価します。S1MB Task Avg の集約には27 benchmarkすべての測定が必要です。各結果は S1MB Japanese の `validate_result` で再計算・検証し、推論失敗が1件でもあれば評価処理は停止します。入力長を超えた場合も黙って切り詰めません。

### 学習済み export の再評価

学習を再実行せず、同じ export を新しい run ID で評価できます。

```sh
SAIHAI_EXPORT="$SAIHAI_ROOT/runs/full/qwen35-9b/export"
SAIHAI_EVAL_OUTPUT="$SAIHAI_ROOT/runs/reevaluation/qwen35-9b.json"
cd "$S1MB_ROOT/evaluator"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$SAIHAI_ROOT" \
  uv run --no-sync python -m saihai.s1mb_eval \
  --export "$SAIHAI_EXPORT" \
  --run-id saihai-qwen35-9b-reevaluation-001 \
  --output "$SAIHAI_EVAL_OUTPUT" \
  --data-revision 59b66dd94c430c1ec6db1c948fa2cc0410400a27

uv run --no-sync s1mb validate data/results/saihai-qwen35-9b-reevaluation-001
cd "$SAIHAI_ROOT"
```

`s1mb_eval` は準備済み snapshot を offline で使い、指定した revision と一致するか確認します。評価途中に最新 snapshot へ切り替えることはありません。

### 指標

[S1MB Japanese の採点仕様](https://github.com/hotchpotch/S1MB/blob/main/evaluator/SCORING.md)では、タスクごとに異なる raw metric を使います。

| タスク | 主な raw metric | ordinary の補正スコア |
|---|---|---|
| Choice | 最上位予測候補に割り当てられた教師確率。hard labelならaccuracy。 | uniform / 固定ID / 固定位置の基準に対する改善 |
| Noul | `p(true)` と教師確率の binary Brier | `2 × balanced accuracy − 1`。Brier skillとは異なる。 |
| Score | 元の数値を使った期待値の normalized MAE | benchmark全体の中央値定数予測のMAEに対する改善 |

S1MB Task Avg は各タスク内で benchmark の補正スコアを平均し、Noul / Choice / Score を等重みで平均します。集約値には全27 benchmarkの測定が必要です。27 benchmark の単純平均 (AVG27) とは計算方法が異なります。

raw metric、タスク別の値、予測分布も確認します。学習データには S1MB Japanese source-family の train データを含むため、この評価だけから未知タスクへの汎化や訓練データとの非重複を主張しません。

## 7. 保存されるもの

| ファイル / ディレクトリ | 内容 |
|---|---|
| `config.json` | 実行時の設定と smoke / fraction / max-steps |
| `data-receipt.json` | 学習データの requested / resolved revision、件数、artifact hash、除外件数 |
| `selected-ids.json` | 実際の train ID と入力長で除外した ID |
| `validation-receipt.json` / `validation-ids.json` | validation の revision、選択条件、ID、除外 ID |
| `train-log.jsonl` | step、train loss、gradient norm、LR、累計事例数、実行時間 |
| `validation-history.json` | 評価 step、平均 loss、subset別loss・件数、評価時間 |
| `memory-probe.json` | probe 有効時の事例 ID、token 数、メモリ使用量 |
| `implementation-receipt.json` | 依存バージョンと実装ファイルの hashなど |
| `checkpoint/` / `checkpoint-hashes.json` | LoRA と FP32 head の checkpoint、その hash |
| `export/` / `export-hashes.json` | merge済みモデル、tokenizer、推論 contract、その hash |
| `summary.json` | 学習件数、step数、loss、validation回数、使用メモリなど |
| `s1mb-summary.json` | S1MB Japanese の全 benchmark 結果、評価 revision、run ID |

S1MB Japanese の raw result は evaluator の `data/results/<run-id>/` にも保存します。学習 CLI が作る run ID は、出力ディレクトリの親名と末尾名を `-` でつないだものです。たとえば `runs/full/qwen35-9b` は `full-qwen35-9b` になります。同じ ID を再利用せず、再実行では親名も含めて新しい名前を使います。

追試では recipe、saihai / S1MB Japanese のコード、両方の lock、データ revision、選択 ID、export hash、入力長上限、評価の完全性をそろえて比較します。

## 8. 4モデルの測定結果

上記の固定データと配布 recipe による1 epochの結果です。各モデルは全33,532件を学習し、279件の固定 validation を10回測定しました。S1MB `japanese-v1` は全27 benchmark・8,067判断で、推論失敗は0件でした。

| モデル | 最終 validation loss ↓ | S1MB japanese-v1 Task Avg ↑ |
|---|---:|---:|
| LLM-jp 4.1 8B ([saihai-v1-8b](https://huggingface.co/hotchpotch/saihai-v1-8b)) | 0.4628 | 49.77 |
| Qwen3.5 0.8B | 0.7875 | 25.71 |
| Qwen3.5 4B | 0.4951 | 45.30 |
| Qwen3.5 9B | 0.4480 | 52.15 |

validation loss は checkpoint の FP32 head、S1MB は BF16 の merge済み export で測っています。validation は多言語の複数 subset、S1MB は日本語の decision tasks なので、同じ評価集の別指標ではありません。学習データには S1MB japanese-v1 と同じ出典の train が含まれるため、未知のタスクでの性能を示すものではありません。出典 train を除いた対照実験は [S1MB Japanese 出典ファミリーを除外する対照実験](s1mb-japanese-source-family-holdout.md) を参照してください。最新データや別設定で学習した結果を、この測定の再現と混同しないでください。

## LLM-jp 8B の推論上限と学習上限

LLM-jp 4.1 8B のベースモデルは位置上限65,536 tokensです。現行LLM-jp YAMLでは
`training.max_length: 8192` と `evaluation.max_input_tokens: 65536` を分けています。
前者は学習とvalidationの入力長上限、後者はexportする推論契約の上限です。
既存8K学習の測定値・checkpointを変更するものではありません。

Python推論は更新した`decision-inference.json`から65,536を読み込みます。
既存exportを更新するには、重みやpromptを変更せず別ディレクトリへ作成します。

```bash
uv run --no-sync python -m saihai.context \
  --export ./model --output ./model-64k --max-input-tokens 65536
```

ベースモデルの`max_position_embeddings`を超える設定は拒否します。
GGUFは更新した契約を`python -m saihai.gguf --systemone-format pplx`で
再埋め込みする必要があります。単に起動引数を変更してもPython契約とGGUFの
decision metadataは更新されません。埋め込み処理では全tensorの不変性を検証します。

llama.cppはtokenizer修正版を使い、`-c 65536 --parallel 1 --no-context-shift`
を指定します。検証したserverは1位置を予約するため、このcontextでのnative入力は
最大65,535 tokensです。Pythonでは65,536 tokensの実行を確認しました。
4 slotsすべてを64K contextにするなら`-c 262144 --parallel 4`が必要ですが、
KV cacheのメモリも増えるので、GPU容量に合わせて並列数を減らします。
`-b 8192 -ub 512`のbatch設定は64Kの入力長制限とは別です。

学習は8Kで行っています。64Kまで実行できることは長入力の判断品質の保証ではなく、
64Kのデータでは十分な性能が出ない可能性があります。S1MB のスコアも64K入力の
性能を測ったものではありません。
