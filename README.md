# sd-forge-llm-prompt-gen-yoiko

## 日本語ガイド

### 概要

この拡張は Stable Diffusion WebUI Forge / Forge - Neo の `txt2img` と `img2img` に、
LLM ベースの positive prompt 生成を追加します。

- `Gen Prompt` を LLM に渡して positive prompt を生成します
- Forge の `Prompt (Optional)` は最終 positive prompt にそのまま残します
- Forge の `Negative prompt` はそのまま画像生成に使います

基本の合成イメージは次の通りです。

```text
final_positive = processed_gen_prompt + ", " + original_prompt
final_negative = negative_prompt
```

### 対応モデル

- `qwen2.5-7b-instruct`
  - base model: `Qwen/Qwen2.5-7B-Instruct`
  - public LoRA: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `qwen3.5-4b`
  - base model: `Qwen/Qwen3.5-4B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `qwen3.5-9b`
  - base model: `Qwen/Qwen3.5-9B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-9B-lora`

### UI

- `Enable LLM Prompt Gen`
- `LLM Model`
- `LLM Load Mode`
- `LLM Weight Mode`
- `LLM Max New Tokens`
- `LLM Candidate Count`
- `Preview Candidates`
- `Candidate`
- `Gen Prompt`

`LLM Max New Tokens` は LLM の `max_new_tokens` を直接制御します。
初期値は `128`、上限は `225` です。

`LLM Candidate Count` を増やすと、1 回の LLM 推論で複数候補を生成します。
`Preview Candidates` で候補を先に出し、`Candidate` で 1 件を固定して使えます。

`LLM Weight Mode` では、ロード方式を切り替えられます。

- `auto`
  - モデルごとの既定挙動を使います
  - `qwen3.5-9b` では高 VRAM 環境で `bf16 + merge` に自動切り替えされることがあります
- `4bit`
  - 常に 4bit を優先します
- `bf16_merge`
  - 非 4bit + merged LoRA を優先します

`LLM Load Mode = load_then_unload_before_image_gen` のときは、
LLM 実行前に現在の画像生成モデルも GPU から CPU RAM へ退避します。
これは SSD / disk 退避ではなく、Neo の既存 offload 経路を使った RAM 優先の退避です。

### Qwen3.5-9B の高速化

この拡張には `qwen3.5-9b` 向けの自動 throughput 最適化が入っています。

- GPU VRAM が十分に大きい環境では、`4bit + PEFT` の代わりに
  `bf16 + LoRA merge` を自動で使います
- 現在の判定目安は `24GB 以上` の VRAM です
- 自動高速化に失敗した場合は、元のロード方式へ戻します

実機ベンチでは、RTX 5090 / 32GB 環境で `qwen3.5-9b` の生成時間が
おおよそ `10.09s -> 3.58s` まで短縮されました。

ログでは次の値を確認できます。

- `throughput_profile_name`
- `throughput_profile_applied`
- `throughput_profile_reason`
- `throughput_profile_fallback_to_original`
- `requested_weight_mode`
- `effective_weight_mode`
- `effective_load_in_4bit`
- `effective_merge_lora_for_inference`
- `image_model_offload_to_ram`

### 導入

Forge の `extensions/` に clone してください。

```bash
git clone https://github.com/yoikoarmor/sd-forge-llm-prompt-gen-yoiko.git extensions/sd-forge-llm-prompt-gen-yoiko
```

通常ユーザー向けの手順です。

1. 拡張を clone する
2. Forge を一度 `--skip-install` なしで起動する
3. 必要なら `configs/model_registry.example.json` を `configs/model_registry.json` にコピーする
4. base model と adapter を設定する
5. Forge を再起動して使う

`--skip-install` を常用する場合は、初回だけ次を実行してください。

```text
bootstrap_forge_env.bat
```

### 設定ファイル

公開用の既定値は次に入っています。

```text
configs/model_registry.example.json
```

ローカルパスや private repo の設定が必要なときだけ、次を作って使ってください。

```text
configs/model_registry.json
```

`configs/model_registry.json` は gitignore 対象です。
このリポジトリでは公開用の既定値を example 側に置いています。

### 依存関係

この拡張は `install.py` と `requirements.txt` で依存をそろえます。

- `transformers == 5.5.0`
- `huggingface_hub == 1.9.0`
- `peft == 0.18.1`
- `accelerate == 1.13.0`
- `bitsandbytes == 0.48.1`
- `tokenizers == 0.22.2`
- `safetensors >= 0.7.0`

注意:

- `Qwen/Qwen3.5-*` は新しめの `transformers` が必要です
- Windows では Qwen3.5 の fast path 依存がそろわないことがあり、その場合は torch 実装へフォールバックします
- そのため、今回の 9B 高速化は fast path ではなく `bf16 + merge` 側で改善しています

### ログで見る項目

- `llm_load_config`
- `model_load_seconds`
- `llm_generate_seconds`
- `throughput_profile_applied`
- `quantization_fallback_used`
- `final_positive_after_dedupe`

### 既知の制限

- beta / experimental 寄りの拡張です
- Hugging Face 側のダウンロードや認証状態に依存します
- Windows では Hugging Face cache の symlink 警告が出ることがあります
- Qwen3.5 の fast path は環境によって使えない場合があります

### ライセンス

この拡張コードは `AGPL-3.0-or-later` です。
base model と LoRA / adapter weight のライセンスは、それぞれの配布元を確認してください。

---

## English Guide

### Overview

This extension adds LLM-assisted positive prompt generation to Forge / Forge - Neo.

- `Gen Prompt` is sent to the LLM
- `Prompt (Optional)` stays in the final positive prompt
- `Negative prompt` is passed through as-is

```text
final_positive = processed_gen_prompt + ", " + original_prompt
final_negative = negative_prompt
```

### Supported Models

- `qwen2.5-7b-instruct`
- `qwen3.5-4b`
- `qwen3.5-9b`

Public adapters:

- `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `yoikoarmor/yoiko-Qwen3.5-9B-lora`

### UI

- `Enable LLM Prompt Gen`
- `LLM Model`
- `LLM Load Mode`
- `LLM Weight Mode`
- `LLM Max New Tokens`
- `LLM Candidate Count`
- `Preview Candidates`
- `Candidate`
- `Gen Prompt`

`LLM Max New Tokens` defaults to `128` and is capped at `225`.

`LLM Weight Mode` controls how the LLM weights are loaded.

- `auto`
  - uses the model default behavior
  - for `qwen3.5-9b`, this may switch to `bf16 + merge` on high-VRAM GPUs
- `4bit`
  - forces the 4bit path
- `bf16_merge`
  - forces non-4bit + merged LoRA

When `LLM Load Mode = load_then_unload_before_image_gen`, the extension also offloads the currently loaded image generation model from GPU to CPU RAM before running the LLM.
This is RAM-first offloading through Neo's existing model offload path, not an SSD/disk staging feature.

### Qwen3.5-9B Throughput Optimization

For `qwen3.5-9b`, the extension applies an automatic high-VRAM profile on large GPUs.

- on sufficiently large VRAM, it switches from `4bit + PEFT` to `bf16 + merged LoRA`
- current threshold is `24GB+`
- if that path fails, it falls back to the original load strategy

On an RTX 5090 / 32GB machine, this reduced generation time from about
`10.09s` to `3.58s`.

Relevant logs:

- `throughput_profile_name`
- `throughput_profile_applied`
- `throughput_profile_reason`
- `throughput_profile_fallback_to_original`
- `requested_weight_mode`
- `effective_weight_mode`
- `image_model_offload_to_ram`

### Install

Clone into Forge `extensions/`:

```bash
git clone https://github.com/yoikoarmor/sd-forge-llm-prompt-gen-yoiko.git extensions/sd-forge-llm-prompt-gen-yoiko
```

For most users:

1. clone the extension
2. start Forge once without `--skip-install`
3. optionally copy `configs/model_registry.example.json` to `configs/model_registry.json`
4. configure model repos or local paths if needed
5. restart Forge

If you always use `--skip-install`, run this once first:

```text
bootstrap_forge_env.bat
```

This repository ships public defaults in `configs/model_registry.example.json`.
Only create `configs/model_registry.json` when you need local paths or private repository settings.

### Notes

- `Qwen/Qwen3.5-*` requires a newer `transformers`
- on Windows, Qwen3.5 fast-path dependencies may not be available
- the 9B optimization here improves throughput via `bf16 + merge`, not via the unavailable fast path

### License

Extension code: `AGPL-3.0-or-later`

Model and adapter weights follow their respective upstream licenses.
