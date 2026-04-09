# sd-forge-llm-prompt-gen-yoiko

Beta Stable Diffusion Forge extension for LLM-assisted positive prompt generation.

This extension adds a small LLM control panel above the built-in prompt area in Forge and can rewrite a `Gen Prompt` into a positive prompt before image generation.

## Beta notice

This repository is currently a beta / experimental pre-release.

- It is usable in practice for local testing and iterative prompt workflows.
- Prompt composition, output cleaning, and LLM integration behavior may still have rough edges in edge cases.
- Feedback and issue reports are welcome.

## What this extension does

- Adds `Gen Prompt` above the built-in prompt area in `txt2img` and `img2img`
- Adds `LLM Model` and `LLM Load Mode` controls
- Uses an LLM to rewrite `Gen Prompt` into a positive prompt
- Keeps the built-in `Prompt (Optional)` field as part of the final positive prompt
- Leaves the built-in `Negative prompt` separate and unchanged

Current final prompt rule when LLM generation succeeds:

```text
final_positive = processed_gen_prompt + ", " + original_prompt
final_negative = negative_prompt
```

If `original_prompt` is empty, the final positive prompt is just the processed LLM output.

If the LLM output is too weak or empty, the extension falls back to `Gen Prompt`.

## Current scope

This repo contains the Forge extension code only.

This repo does **not** include:

- base model weights
- LoRA weights
- training artifacts
- any mandatory local `artifacts/` dependency

Base models and LoRA adapters are expected to be provided separately and configured in `configs/model_registry.json`.
They can be referenced either by local path or by Hugging Face repo ID.

Current model lineup in this beta:

- `qwen2.5-7b-instruct`
  - base model: `Qwen/Qwen2.5-7B-Instruct`
  - public LoRA: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `qwen3.5-4b`
  - base model: `Qwen/Qwen3.5-4B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
  - local path override is still supported
- `qwen3.5-9b`
  - base model: `Qwen/Qwen3.5-9B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-9B-lora`
  - local path override is still supported

Important:

- the repository code is licensed separately from model weights
- base models and LoRA/adapters may have their own licenses
- users must verify weight licenses separately before use or redistribution

## Supported target

- Stable Diffusion Forge
- Forge Script-based extension layout
- Local or Hugging Face-backed Qwen-style text-generation runtime
- Current beta targets:
  - `Qwen/Qwen2.5-7B-Instruct + yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
  - `Qwen/Qwen3.5-4B + yoikoarmor/yoiko-Qwen3.5-4B-lora`
  - `Qwen/Qwen3.5-9B + yoikoarmor/yoiko-Qwen3.5-9B-lora`

## Repository layout

```text
sd-forge-llm-prompt-gen-yoiko/
  README.md
  LICENSE
  .gitignore
  metadata.ini
  install.py
  requirements.txt
  style.css
  javascript/
    llm_prompt_gen_ui.js
  scripts/
    forge_llm_prompt_gen.py
  backend/
    __init__.py
    registry.py
    loader.py
    generator.py
    runtime.py
    prompt_builder.py
  configs/
    model_registry.example.json
    generation_defaults.json
```

Local-only files that are intentionally not tracked:

- `artifacts/`
- `configs/model_registry.json`
- `__pycache__/`

## UI overview

The extension exposes these controls:

- `Enable LLM Prompt Gen`
- `LLM Model`
- `LLM Load Mode`
- `Model Load Type`
- `LLM Max New Tokens`
- `Gen Prompt`
- `Prompt (Optional)` (Forge built-in prompt field)
- `Negative prompt` (Forge built-in negative field)

### Meaning of each field

#### Gen Prompt

Primary input for the LLM.

This is the text that gets rewritten into a positive prompt.

#### Prompt (Optional)

The normal Forge positive prompt field.

- It is not sent into the LLM in the current beta behavior.
- It is preserved in the final positive prompt after LLM generation.

#### Negative prompt

Used as the final negative prompt as usual.

- It is not merged into the positive prompt.
- It is not rewritten in this beta release.

#### LLM Load Mode

- `keep_loaded`
  - Keep the loaded LLM in memory for reuse
- `load_then_unload_before_image_gen`
  - Load for prompt generation, then unload before image generation continues

#### Model Load Type

- `GPU_load`
  - Default
  - Tries to keep the model on GPU first
  - If a quantized load spills beyond VRAM, the loader falls back to GPU+RAM offload
- `GPU_CPU_load`
  - Starts in GPU+RAM offload mode immediately
  - Useful on lower-VRAM systems that would otherwise fail before generation
- `auto`
  - Chooses between the two based on a simple VRAM/model-size heuristic
  - For example, `qwen3.5-9b` on an 8 GB-class GPU will lean toward CPU offload

#### LLM Max New Tokens

- Slider control for per-request `max_new_tokens`
- Default is `128`
- Higher values can produce longer prompts but also increase latency and memory use

## Installation

Clone this repository into your Forge `extensions/` directory.

Example from the Forge root:

```bash
git clone https://github.com/yoikoarmor/sd-forge-llm-prompt-gen-yoiko.git extensions/sd-forge-llm-prompt-gen-yoiko
```

## Quick start

### Normal first launch

This is the recommended path.

1. Clone the repo.
2. Start Forge normally once.
3. Let Forge run this extension's `install.py`.
4. Copy `configs/model_registry.example.json` to `configs/model_registry.json`.
5. Restart Forge and start generating.

If you want the closest thing to "clone and go", do not use `--skip-install` on the first launch.

### If you always use `--skip-install`

Run this once before your first `--skip-install` launch:

```text
bootstrap_forge_env.bat
```

That one-time step installs the pinned LLM dependency set into the Forge venv and writes the compatibility shims needed for Forge's current environment.

## Config setup

Create your local runtime config by copying:

```text
configs/model_registry.example.json
```

to:

```text
configs/model_registry.json
```

`configs/model_registry.json` is intentionally gitignored so you can keep machine-local paths, cache settings, or private repo choices there.

If `configs/model_registry.json` does not exist, the extension falls back to `configs/model_registry.example.json`.

The example file already contains public Hugging Face entries for:

- `qwen2.5-7b-instruct`
- `qwen3.5-4b`
- `qwen3.5-9b`

Path rules:

- If a value resolves to a local path, it is used as a local path.
- If a value looks like `owner/model` and no local path exists, it is treated as a Hugging Face repo ID.
- `cache_dir` is optional. If omitted, the standard Hugging Face cache is used.
- If a local path is missing and `allow_auto_download_missing` is `true`, the extension can fall back to the matching `fallback_*` Hugging Face reference at LLM runtime.

## Published model IDs

Recommended public setup:

- base: `Qwen/Qwen2.5-7B-Instruct`
- adapter: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`

Additional public adapters:

- `Qwen/Qwen3.5-4B` + `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `Qwen/Qwen3.5-9B` + `yoikoarmor/yoiko-Qwen3.5-9B-lora`

On the first run, the extension can download both the base model and the adapter from Hugging Face. Large first-time downloads are expected.

Practical VRAM note:

- `qwen3.5-9b` is a large option and may not fit cleanly on GPUs around 8 GB VRAM.
- `GPU_load` retries with GPU+RAM offload when a 4bit load spills to CPU/disk.
- `GPU_CPU_load` starts in GPU+RAM offload mode immediately.
- If that still fails, use `qwen3.5-4b` or `qwen2.5-7b-instruct` on that machine.

## Dependency notes

The extension uses `install.py` and `requirements.txt` to request missing packages inside the Forge environment.

Pinned dependency set used for the current public adapters:

- `transformers == 5.5.0`
- `huggingface_hub == 1.9.0`
- `peft == 0.18.1`
- `accelerate == 1.13.0`
- `bitsandbytes == 0.48.1`
- `tokenizers == 0.22.2`
- `safetensors >= 0.7.0`

Notes:

- `bitsandbytes` support depends on your platform and CUDA setup.
- The `Qwen/Qwen3.5` paths need a recent `transformers` / `peft` / `bitsandbytes` stack, including a 1.x `huggingface_hub`.
- Forge's current Gradio stack still imports the removed `HfFolder` symbol, so `install.py` writes compatibility patches during setup.
- The extension also applies a runtime compatibility patch for the CLIP causal mask path used by Forge text encoders.
- If Forge pins an older `bitsandbytes` build at startup, the extension can retry in non-4bit mode instead of aborting the LLM call immediately.
- On low-VRAM systems, the loader can also retry once with CPU offload if a 4bit model spills beyond available GPU memory.
- For Qwen 3.5-family templates, the default is `enable_thinking = false`.
- On some high-VRAM Windows setups, `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` can be faster with `load_in_4bit = false` and `merge_lora_for_inference = true`.
- On older Forge runtimes that still use `torch < 2.4`, `install.py` now selects a legacy dependency profile automatically to avoid breaking Forge startup. In that mode, `qwen2.5-7b-instruct` is the intended option and `Qwen/Qwen3.5-*` is not supported until the host Forge/PyTorch stack is upgraded.

## LoRA distribution policy

This repository does **not** bundle:

- base model checkpoints
- adapter weights
- training output directories

Users are expected to configure their own `base_model_name_or_path` and `adapter_path` in `configs/model_registry.json`.

Published public adapter name:

- `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`

Planned distribution split:

- GitHub repository: Forge extension code
- Hugging Face model repository: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- Hugging Face model repository: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- Hugging Face model repository: `yoikoarmor/yoiko-Qwen3.5-9B-lora`
- Current published adapter lineup for this extension:
  - `Qwen/Qwen2.5-7B-Instruct + yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
  - `Qwen/Qwen3.5-4B + yoikoarmor/yoiko-Qwen3.5-4B-lora`
  - `Qwen/Qwen3.5-9B + yoikoarmor/yoiko-Qwen3.5-9B-lora`

Current license split:

- Forge extension code: `AGPL-3.0-or-later`
- `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`: separate adapter package license
- `yoikoarmor/yoiko-Qwen3.5-4B-lora`: separate adapter package license
- `yoikoarmor/yoiko-Qwen3.5-9B-lora`: separate adapter package license

## Migration note

Old local experimental naming may still appear in local-only files under `artifacts/`.

- old experimental name: `fold4_best_eval_adapter`
- public 2.5 adapter name: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- public 4B adapter name: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- public 9B adapter name: `yoikoarmor/yoiko-Qwen3.5-9B-lora`

## Usage

1. Restart Forge after installation.
2. Open `txt2img` or `img2img`.
3. Enable `LLM Prompt Gen`.
4. Choose an entry in `LLM Model`.
5. Enter your LLM request in `Gen Prompt`.
6. Optionally keep extra tags in the normal `Prompt (Optional)` field.
7. Run generation.

### Current behavior summary

#### If `Enable LLM Prompt Gen` is off

- Forge behavior is unchanged.
- If an LLM model is currently loaded in the extension runtime, it is unloaded when you press Generate with the feature turned off.

#### If `LLM Model = none`

- No model is loaded.
- The extension does not call the LLM.
- `Gen Prompt` is used directly as the prompt head in manual prepend behavior.

#### If a real model is selected and `Gen Prompt` is not empty

- The extension calls the configured LLM once.
- The generated text becomes the processed prompt head.
- The normal Forge prompt field is appended after that.
- Exact duplicate comma-separated tags are removed.

### Clean Hugging Face retrieval check

If you want to confirm that local `artifacts/` are not required:

1. Stop Forge.
2. Rename `artifacts/` to something like `artifacts_backup/`.
3. Set `base_model_name_or_path` to either `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3.5-4B`, or `Qwen/Qwen3.5-9B`.
4. Set `adapter_path` and `tokenizer_name_or_path` to `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`, `yoikoarmor/yoiko-Qwen3.5-4B-lora`, or `yoikoarmor/yoiko-Qwen3.5-9B-lora`.
5. Make sure `local_files_only` is `false`.
6. Start Forge and run one prompt generation.
7. Check the logs for Hugging Face download and PEFT attach messages.
8. Restore `artifacts/` later if you still want the local-copy workflow available.

## generation_defaults.json

`configs/generation_defaults.json` holds public-safe defaults for inference behavior.

Current defaults are intentionally conservative for beta use:

- `input_template_mode = simple_chat_template`
- `use_cache = true`
- `enable_thinking = false`
- `seed_mode = random`
- `debug_compare_input_variants = false`

`use_cache = true` means generation-time KV cache is enabled during each `model.generate()` call and cleared immediately after generation completes.
This does not keep previous prompt outputs across separate runs.

Recommended beta interpretation:

- `seed_mode = random` gives more variation between runs
- `use_cache = true` materially improves autoregressive prompt-generation speed, especially for longer 4B outputs
- `enable_thinking = false` is kept as the default for both supported models to reduce reasoning-style replies in prompt generation
- on high-VRAM GPUs, `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` may generate faster with `load_in_4bit = false` and `merge_lora_for_inference = true`
- if you want more reproducible debugging, switch `seed_mode` to `fixed`
- if prompt quality feels weak, adjust prompt wording before changing generation defaults aggressively

## Troubleshooting

### The extension loads but prompt generation does not run

Check:

- `Enable LLM Prompt Gen` is on
- `LLM Model` is not `none`
- `Gen Prompt` is not empty
- `configs/model_registry.json` exists and is valid JSON

### The model fails to load

Check:

- base model path or model ID
- adapter path
- Hugging Face connectivity and available disk space
- `bitsandbytes` availability
- CUDA / GPU availability
- Forge environment dependency installation

On Windows, `huggingface_hub` may print a symlink warning if Developer Mode is disabled.
That warning alone does not necessarily mean the load failed.

If Forge fails very early with an import error mentioning `HfFolder`, your environment likely has `huggingface_hub >= 1.0.0`.
This extension currently fixes that by installing a small compatibility shim during setup.
If that shim is missing, rerun the extension install step or run `bootstrap_forge_env.bat`, then restart Forge.

If the logs show `quantization_fallback_used=True`, Forge has probably kept an older `bitsandbytes` build than the one requested by this extension.
In that case the model can still load in non-4bit mode if enough VRAM is available.

If the logs show `cpu_offload_retry_used=True`, the loader detected that the quantized model did not fit cleanly in VRAM and retried with CPU offload.
That is expected on lower-VRAM GPUs, but for smoother prompt generation you may still want to switch to `qwen3.5-4b` or `qwen2.5-7b-instruct`.

If `Qwen/Qwen3.5-4B` feels unexpectedly slow, check:

- `quantization_fallback_used`
- `cpu_offload_retry_used`
- `cpu_offload_folder`
- `cpu_offload_max_memory`
- `llm_generate_seconds`
- `merge_lora_for_inference_applied`
- `merge_lora_skipped_reason`
- `llm_input_empty_think_block_stripped`
- whether the warning `The fast path is not available ... flash-linear-attention ... causal-conv1d` appears

On some Windows setups, Qwen 3.5 can still be slower than Qwen 2.5 even though it is a smaller model because:

- the Qwen 3.5 fast path is unavailable
- 4bit may not be the fastest choice for that model on high-VRAM GPUs
- merged LoRA inference can outperform unmerged PEFT inference on some high-VRAM setups
- longer generated outputs amplify the effect when cache is disabled or unavailable

### I copied the repo but nothing works

Make sure you did all of the following:

- cloned into Forge `extensions/`
- restarted Forge
- created `configs/model_registry.json`
- pointed it to a real base model and a real LoRA adapter

### Logs to inspect

Useful runtime logs include:

- `llm_load_config`
- `base_model_source`
- `adapter_source`
- `resolved_base_model_reference`
- `resolved_adapter_reference`
- `effective_base_model_reference`
- `effective_adapter_reference`
- `base_missing_local_fallback_used`
- `adapter_missing_local_fallback_used`
- `cache_dir`
- `base_download_started`
- `base_download_finished`
- `adapter_download_started`
- `adapter_download_finished`
- `model_load_seconds`
- `base_model_class`
- `final_model_class`
- `model_load_type_requested`
- `model_load_type_effective`
- `model_load_type_reason`
- `model_load_total_vram_gib`
- `is_peft_model`
- `active_adapter`
- `tokenizer_source`
- `chat_template_source`
- `llm_input_empty_think_block_stripped`
- `llm_generate_seconds`
- `original_prompt_injected_to_llm`
- `original_prompt_appended_after_llm`
- `final_positive_before_dedupe`
- `final_positive_after_dedupe`
- `final_negative_preview`

## Known limitations

- Beta / experimental quality
- Only a narrow local-runtime path has been tested
- Mid-generate interrupt inside a single `model.generate()` call is not implemented
- Prompt shaping may still need manual tuning for some models or adapters
- Initial Hugging Face downloads may take time and require substantial disk space
- Offline use requires the model and adapter to already exist in the local Hugging Face cache or as local paths
- Windows may show a Hugging Face cache symlink warning during downloads

## Security and privacy notes

- This repo intentionally excludes local model files and adapters from version control
- `configs/model_registry.json` is expected to be local-only
- Do not commit your local weights or machine-specific config back into the repository

## Release preparation notes

Before publishing or pushing a public beta:

- do not commit `artifacts/`
- do not commit `configs/model_registry.json`
- do not commit caches, logs, or local-only generated files
- if they were already tracked, remove them from the git index before publishing
- do not commit `publish/`
- always check `git status` before creating a release or tag

Suggested local checks:

```bash
git status
git diff --stat
git rm --cached -r artifacts configs/model_registry.json __pycache__ publish
```

## License

This repository is released under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`).

This applies to the extension code in this repository.

Base models and LoRA/adapters are not bundled in this repository.

The published adapter packages are:

- `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `yoikoarmor/yoiko-Qwen3.5-9B-lora`

Those adapter packages are separate from this repository and may use different licenses for the weight packages themselves.

Model weights may have their own licenses and must be checked separately by users before use, redistribution, or publication.

See [LICENSE](LICENSE).

## Feedback

Bug reports, setup notes, and quality feedback are welcome.

If you open an issue, please include:

- Forge version
- platform and GPU info
- relevant logs
- your `generation_defaults.json` changes, if any
- whether you are using local paths or remote model IDs

---

## Japanese guide

### 概要

この拡張は、Forge の `txt2img` / `img2img` に小さな LLM パネルを追加し、`Gen Prompt` を画像生成向けの positive prompt に書き換えます。

- `Gen Prompt` を LLM に渡して positive prompt を生成
- Forge 標準の `Prompt (Optional)` はそのまま保持
- Forge 標準の `Negative prompt` は変更せずそのまま使用

LLM 生成が成功したときの基本ルールは次のとおりです。

```text
final_positive = processed_gen_prompt + ", " + original_prompt
final_negative = negative_prompt
```

`Prompt (Optional)` が空の場合、最終 positive prompt は LLM の出力のみになります。  
LLM 出力が弱すぎる、または空の場合は `Gen Prompt` にフォールバックします。

### このリポジトリに含まれるもの

このリポジトリには Forge 拡張コードのみが含まれます。  
以下は含みません。

- base model weights
- LoRA / adapter weights
- training artifacts
- ローカル専用の `artifacts/`

モデル本体と LoRA は `configs/model_registry.json` で別途指定してください。  
ローカルパスでも Hugging Face repo ID でも指定できます。

### 現在の対応モデル

- `qwen2.5-7b-instruct`
  - base model: `Qwen/Qwen2.5-7B-Instruct`
  - public LoRA: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `qwen3.5-4b`
  - base model: `Qwen/Qwen3.5-4B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `qwen3.5-9b`
  - base model: `Qwen/Qwen3.5-9B`
  - public LoRA: `yoikoarmor/yoiko-Qwen3.5-9B-lora`

### 導入手順

Forge の `extensions/` にこのリポジトリを clone してください。

```bash
git clone https://github.com/yoikoarmor/sd-forge-llm-prompt-gen-yoiko.git extensions/sd-forge-llm-prompt-gen-yoiko
```

### 最短の初回導入

いちばんおすすめの流れです。

1. リポジトリを clone する
2. Forge を `--skip-install` なしで一度だけ通常起動する
3. この拡張の `install.py` を実行させる
4. `configs/model_registry.example.json` を `configs/model_registry.json` にコピーする
5. base model と adapter を設定する
6. Forge を再起動して使い始める

「clone 後すぐ使える」に一番近いのは、この通常初回起動ルートです。

### `--skip-install` を常用する場合

初回だけ先に次を実行してください。

```text
bootstrap_forge_env.bat
```

これで Forge venv に必要な依存と互換パッチを入れます。  
その後は `--skip-install` 起動に戻して大丈夫です。

### 設定ファイル

まず次をコピーします。

```text
configs/model_registry.example.json
```

コピー先:

```text
configs/model_registry.json
```

`configs/model_registry.json` は gitignore 対象です。  
マシン固有のローカルパスや private repo 設定はここに書いてください。

ルールは次のとおりです。

- ローカルパスとして存在する場合はローカルパスとして使用
- `owner/model` 形式でローカルに存在しない場合は Hugging Face repo ID として扱う
- `cache_dir` は省略可
- ローカルパスが見つからず `allow_auto_download_missing = true` の場合、対応する `fallback_*` の Hugging Face 参照へフォールバック可能

### 公開済み Hugging Face LoRA

公開済み adapter は次の 3 つです。

- `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `yoikoarmor/yoiko-Qwen3.5-9B-lora`

初回は base model と adapter のダウンロードが入るため、時間とディスク容量が必要です。

### 依存関係と互換性

この拡張は `install.py` と `requirements.txt` を使って、Forge 環境に必要な依存を揃えます。  
現在の公開 adapter で想定している主なバージョンは次のとおりです。

- `transformers == 5.5.0`
- `huggingface_hub == 1.9.0`
- `peft == 0.18.1`
- `accelerate == 1.13.0`
- `bitsandbytes == 0.48.1`
- `tokenizers == 0.22.2`
- `safetensors >= 0.7.0`

補足:

- `Qwen/Qwen3.5-*` は新しめの `transformers` 系依存が必要です
- Forge 側との互換性のため、初回セットアップ時に互換パッチを自動で入れます
- Forge が古い `bitsandbytes` を保持した場合でも、拡張側で non-4bit fallback を試みます
- `Qwen 3.5` 系では既定で `enable_thinking = false` です

### よくある確認ポイント

うまく動かないときは、まず次を確認してください。

- Forge を `extensions/` 配下に clone したか
- 初回に Forge を通常起動したか
- `configs/model_registry.json` を作ったか
- base model と adapter の参照先が正しいか

ログでは次の項目が重要です。

- `llm_load_config`
- `resolved_base_model_reference`
- `resolved_adapter_reference`
- `model_load_seconds`
- `llm_generate_seconds`
- `quantization_fallback_used`
- `final_positive_after_dedupe`

### 既知の制限

- まだ beta / experimental 段階です
- 一部のモデルや adapter では prompt shaping の微調整が必要です
- 初回の Hugging Face ダウンロードは重い場合があります
- オフライン利用には事前キャッシュまたはローカル配置が必要です
- Windows では Hugging Face cache の symlink 警告が出ることがあります

### ライセンス

このリポジトリ内の拡張コードは `AGPL-3.0-or-later` です。  
base model や LoRA / adapter weight は同梱しておらず、各 weight package のライセンスは別途確認してください。
