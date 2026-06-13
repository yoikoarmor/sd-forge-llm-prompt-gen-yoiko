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

### 簡易説明

導入や使い方のざっくりした流れは、こちらの note にもまとめています。  
[sd-forge-llm-prompt-gen-yoiko 簡易説明](https://note.com/yoikoarmor/n/nfa682c31e319)

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

### GGUF / llama.cpp バックエンド

通常の Transformers + LoRA 経路はそのまま利用できます。低VRAM環境や、配布しやすい単体ファイル運用をしたい場合だけ、任意で GGUF バックエンドを使えます。

GGUF は optional 機能です。`requirements.txt` には `llama-cpp-python` を入れていないため、使う場合だけ追加してください。

```bash
python -m pip install -r requirements.gguf.txt
```

CUDA wheel を明示して入れる場合の例です。

```bash
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
```

GGUF を使うには、`configs/model_registry.json` に `backend: "llama_cpp"` のモデルを追加します。

```json
{
  "models": {
    "qwen3.5-4b-gguf-q4km": {
      "enabled": true,
      "backend": "llama_cpp",
      "gguf_path": "<path/to/yoiko-qwen3.5-4b-merged-Q4_K_M.gguf>",
      "n_ctx": 4096,
      "n_gpu_layers": -1,
      "n_batch": 512,
      "thinking_suppression": "auto",
      "flash_attn": true
    }
  }
}
```

Hugging Face 上の GGUF ファイルを使う場合は、`gguf_repo_id` と `gguf_filename` を指定できます。

```json
{
  "backend": "llama_cpp",
  "gguf_repo_id": "your-name/your-gguf-repo",
  "gguf_filename": "model-Q4_K_M.gguf",
  "local_files_only": false
}
```

注意点:

- `LLM Weight Mode` は Transformers 用の設定です。`llama_cpp` バックエンドでは無視され、ログに `weight_mode ignored for llama_cpp backend` が出ます。
- `thinking_suppression` は `"auto"`, `"no_think"`, `"none"` から選べます。既定は `"auto"` です。
- `"auto"` では GGUF メタデータの `general.architecture` が Qwen 系で、chat template に `enable_thinking`, `/no_think`, `<think>` などの thinking marker がある場合だけ `/no_think` を付けます。
- 非対応モデルに `/no_think` が混ざらないよう、非Qwen系または thinking marker なしの template では付与しません。
- 以前の実装で無条件に `/no_think` を足していた理由は、Qwen thinking モデルの実用ケースと smoke を先に通すための暫定策でした。公開版では副作用が強いため、metadata 判定 + registry override に変更しています。
- `qwen2.5` / `qwen3` 系は llama.cpp 側で利用例があります。`qwen3.5` 系は llama.cpp の新しめの変換スクリプトで実機検証してください。
- `llama-cpp-python` が未導入の場合は、LLMロード時に導入方法つきのエラーを出します。

#### GGUF 変換ツール

LoRA を base model に merge して GGUF に変換する補助ツールを同梱しています。
既定では llama.cpp を `1593d5684d077c07fc788e9527ec1bd52287de7f` に pin して使います。この ref の `convert_hf_to_gguf.py` で Qwen3 系分岐があることを確認しています。
別の llama.cpp tag / commit を使う場合は `--llama-cpp-ref` を指定してください。

```bash
python tools/convert_to_gguf.py ^
  --model-key qwen3.5-4b ^
  --outdir E:\gguf\yoiko-qwen3.5-4b ^
  --quant Q4_K_M ^
  --register
```

既存の llama.cpp checkout を使う場合:

```bash
python tools/convert_to_gguf.py ^
  --model-key qwen3.5-4b ^
  --outdir E:\gguf\yoiko-qwen3.5-4b ^
  --llama-cpp-dir E:\tools\llama.cpp ^
  --llama-cpp-ref 1593d5684d077c07fc788e9527ec1bd52287de7f ^
  --quantize-bin E:\tools\llama.cpp\build\bin\Release\llama-quantize.exe ^
  --quant Q4_K_M
```

コマンドだけ確認したい場合:

```bash
python tools/convert_to_gguf.py --model-key qwen3.5-4b --outdir E:\gguf\test --dry-run
```

変換後の簡易テスト:

```bash
python tools/smoke_gguf.py --gguf-path E:\gguf\yoiko-qwen3.5-4b\qwen3.5-4b-merged-Q4_K_M.gguf --prompt "女性、高身長"
```

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

### GGUF / llama.cpp Backend

The default Transformers + LoRA path is unchanged. GGUF support is optional and is useful when you prefer a single converted model file or a llama.cpp-based low-VRAM path.

Install the optional dependency only when you use GGUF:

```bash
python -m pip install -r requirements.gguf.txt
```

Then add a `backend: "llama_cpp"` entry to `configs/model_registry.json`, using either `gguf_path` or `gguf_repo_id` + `gguf_filename`. For remote GGUF files, set `local_files_only` to `false` unless the file is already cached.
`thinking_suppression` defaults to `auto`; it only appends `/no_think` when GGUF metadata indicates a Qwen-family thinking template. Override with `no_think` or `none` if needed.

Conversion helper:

```bash
python tools/convert_to_gguf.py --model-key qwen3.5-4b --outdir E:\gguf\yoiko-qwen3.5-4b --quant Q4_K_M --register
```

The converter pins llama.cpp to `1593d5684d077c07fc788e9527ec1bd52287de7f` by default. Use `--llama-cpp-ref` to override it.

Smoke test:

```bash
python tools/smoke_gguf.py --gguf-path E:\gguf\yoiko-qwen3.5-4b\qwen3.5-4b-merged-Q4_K_M.gguf --prompt "woman, tall"
```

### Notes

- `Qwen/Qwen3.5-*` requires a newer `transformers`
- on Windows, Qwen3.5 fast-path dependencies may not be available
- the 9B optimization here improves throughput via `bf16 + merge`, not via the unavailable fast path

### License

Extension code: `AGPL-3.0-or-later`

Model and adapter weights follow their respective upstream licenses.
