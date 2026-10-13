# S1MB Japanese 出典ファミリーを除外する対照実験

今後の評価先は S1MB `japanese-v1`。`jdmb_source_family` と `jp-jdmb-*` は学習データの旧称に由来する識別名として保持します。

LLM-jp 4.1 8B と Qwen3.5 4B について、S1MB `japanese-v1` と同じ出典ファミリーの学習例を除いた条件を比較する。モデル、LoRA、batch size、seed、入力形式、soft target、validation、評価方法は通常設定と同じにする。

ここでの「同じドメイン」は、記録された元データセット・出典ファミリーの一致を指す。知識・数学・推論などの広い分野を丸ごと除外する実験ではない。合成データの意味的な重複やベースモデルの事前学習への混入は、このフィルタでは判定できない。

## YAML の設定

用意した設定は次の4つ。

- [`llmjp41-8b-s1mb-japanese-family-holdout.yaml`](../configs/llmjp41-8b-s1mb-japanese-family-holdout.yaml)
- [`qwen35-4b-s1mb-japanese-family-holdout.yaml`](../configs/qwen35-4b-s1mb-japanese-family-holdout.yaml)
- [`qwen35-0.8b-s1mb-japanese-family-holdout.yaml`](../configs/qwen35-0.8b-s1mb-japanese-family-holdout.yaml)
- [`qwen35-9b-s1mb-japanese-family-holdout.yaml`](../configs/qwen35-9b-s1mb-japanese-family-holdout.yaml)

0.8B と9Bも同じ除外条件を使う。9Bは batch size 1・gradient accumulation 16 とし、smokeでは長い入力の forward/backward によるメモリ確認を行う。

通常設定との差は `dataset.train_filter` だけ。配布設定は `revision: latest` とし、準備時に具体的な revision を固定する。

```yaml
dataset:
  repo_id: hotchpotch/saihai-v1-dataset
  revision: latest
  split: train
  train_filter:
    exclude_s1mb_japanese_source_family: true
    exclude_subsets: []
    exclude_dataset_groups: []
    exclude_source_patterns:
      - 'jglue|jnli|jsts|jcommonsenseqa'
      - 'jcola|jamp|temporalNLI_dataset|jsem|jgpqa|gpqa|sobaco'
```

この例の正規表現は一部のみ。対照実験では配布 YAML の全リストを使う。リストには JGLUE、JSICK、JaNLI、KUCI、Mintaka、commonsense-moral-ja、JCulture、JamC-QA、MMLU 系、JMTEB の採用検索ソース、JCoLA、Jamp、JSeM、GPQA、SOBACO の識別パターンを含む。

各条件は OR で適用する。`exclude_subsets` と `exclude_dataset_groups` は完全一致、`exclude_source_patterns` は大文字・小文字を区別しない正規表現。検索対象は subset と provenance の出典識別フィールドで、入力本文・正答・教師モデル・生成設定・prompt revision は検索しない。`jdmb_source_family` は配布 Parquet の列から読み込み、入力 prompt には入れない。このフラグがない行は、フラグ除外を有効にするとエラーになる。

直接一致した行に加えて、同じ `parent_id` または `group_id` でつながる train 行を推移的に除外する。残る行の順序、候補、soft target は保持し、補充サンプリングはしない。元 release の checksum・件数・ID・train/validation 非重複を検証してからフィルタを適用する。smoke や `--fraction` の選択はその後に行う。

## GPU を使わない準備

環境の用意は [学習と評価の再現手順](training-and-evaluation.md) を参照する。以下はモデルをロードせず、学習も起動しない。出力先は新しいディレクトリにする。

```bash
uv run --no-sync python -m saihai.prepare_training \
  --config configs/llmjp41-8b-s1mb-japanese-family-holdout.yaml \
  --output runs/holdout-llmjp-prepared

uv run --no-sync python -m saihai.prepare_training \
  --config configs/qwen35-4b-s1mb-japanese-family-holdout.yaml \
  --output runs/holdout-qwen4b-prepared
```

出力の `config.yaml` はデータ revision を固定した設定。`train-filter-receipt.json` に条件、除外 ID、直接一致の理由、subset ごとの除外件数を保存する。`data-receipt.json` に release の revision と hash、`prepared-ids.json` に学習候補 ID と固定 validation ID を保存する。準備段階では token 長と GPU メモリを検証しない。学習時に同じ checksum 検証とフィルタを再実行し、token 長上限による除外を別途記録する。

2モデルを確実に同じデータで比較するには、1つ目の準備で固定した `dataset.revision` を2つ目の入力設定にも指定してから準備する。以前の通常条件との比較では、その学習の `data-receipt.json` と同じ revision を使う。

## 確認した件数

2026-10-10 に `latest` が解決した revision は `906504ac4b72e133a613011318afd8206fb5e8ec`。通常条件の学習データと一致した。

| 項目 | 件数 |
|---|---:|
| 元 train | 33,532 |
| 除外 | 3,889 |
| 残る train | 29,643 |
| validation 全件 | 8,729 |
| 固定 validation：各 subset 最大3件 | 279 |

除外は `jp-jdmb-*` の専用10 subset。今回の snapshot では追加の出典検索・関連グループ除外による件数増加はなかった。データセットのフラグは専用 subset の目印であり、全出典・全合成データの重複検出器ではない。

validation は除外前と同じ各 subset の最大3件を使い、S1MB Japanese 出典の validation も残す。学習途中に10回評価し、学習後は固定した S1MB japanese-v1 revision の全27セットを評価する。validation は開発用測定であり、S1MB Japanese の評価データを学習に使うことはない。

## GPU が空いてからの実行

まず validation を含む smoke を行う。`S1MB_ROOT` は再現手順で用意した S1MB checkout、`GPU_ID` は空いている GPU の番号とする。

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run --no-sync python -m saihai.training \
  --config runs/holdout-llmjp-prepared/config.yaml \
  --output runs/holdout-llmjp-smoke --s1mb-root "$S1MB_ROOT" \
  --smoke --eval-limit 2
```

smoke 成功後、同じ固定設定で `--smoke --eval-limit 2` を外し、新しい出力先で全件学習する。Qwen4B は対応する prepared 設定を使う。2 GPU が空いていれば別モデルを並行実行できる。

通常条件と同じ1 epoch・実効 batch size 16 のため、除外条件は約1,853 optimizer steps、通常条件は2,096 stepsとなる。warmup と validation の位置も総 steps に応じて変わる。これは学習データの除外による比較であり、更新回数を揃えた比較ではない。結果にはデータ量・steps・validation loss・S1MB Task Avgを併記する。
