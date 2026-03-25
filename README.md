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

Base models and LoRA adapters are expected to be provided separately and configured in `configs/model_registry.json exists and is valid JSON, or the example fallback is usable`.
They can be referenced either by local path or by Hugging Face repo ID.

Public LoRA release name:

- `yoiko-Qwen2.5-7B-Instruct-lora`

Important:

- the repository code is licensed separately from model weights
- base models and LoRA/adapters may have their own licenses
- users must verify weight licenses separately before use or redistribution

## Supported target

- Stable Diffusion Forge
- Forge Script-based extension layout
- Local Qwen-style text-generation runtime
- Current beta focus: `Qwen/Qwen2.5-7B-Instruct + yoiko-Qwen2.5-7B-Instruct-lora`

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

## Installation

Clone this repository into your Forge `extensions/` directory.

Example from the Forge root:

```bash
git clone https://github.com/yoikoarmor/sd-forge-llm-prompt-gen-yoiko.git extensions/sd-forge-llm-prompt-gen-yoiko
```

Then restart Forge.

## Dependencies

The extension uses `install.py` and `requirements.txt` to request missing packages inside the Forge environment.

Current dependency list:

- `transformers`
- `huggingface_hub`
- `peft`
- `accelerate`
- `bitsandbytes`
- `safetensors`

Notes:

- `bitsandbytes` support depends on your platform and CUDA setup.
- This beta is primarily aimed at local NVIDIA GPU environments where 4-bit inference is available.

## First-time setup

### 1. Copy the example config

Create your local runtime config by copying:

```text
configs/model_registry.example.json
```

to:

```text
configs/model_registry.json
```

`configs/model_registry.json` is intentionally gitignored so you can keep machine-local paths, cache settings, or private repo choices there.

If `configs/model_registry.json` does not exist, the extension will fall back to `configs/model_registry.example.json` as a beta default.

### 2. Edit `configs/model_registry.json`

Set your own base model and adapter references.

Example:

```json
{
  "models": {
    "qwen2.5-7b-instruct": {
      "enabled": true,
      "description": "Recommended beta example using Hugging Face repo IDs",
      "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
      "adapter_path": "yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora",
      "tokenizer_name_or_path": "yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora",
      "cache_dir": null,
      "fallback_base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
      "fallback_adapter_path": "yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora",
      "fallback_tokenizer_name_or_path": "yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora",
      "allow_auto_download_missing": true,
      "load_in_4bit": true,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": "bfloat16",
      "use_double_quant": true,
      "device_map": "auto",
      "torch_dtype": "bfloat16",
      "trust_remote_code": false,
      "local_files_only": false,
      "tokenizer_source": "adapter",
      "chat_template_source": "adapter",
      "use_fast_tokenizer": true
    }
  }
}
```

Path rules in this beta:

- If the value resolves to a local path, it is used as a local path
- If the value looks like `owner/model` and no local path exists, it is treated as a Hugging Face repo ID
- `cache_dir` is optional; if omitted, the standard Hugging Face cache is used
- If a local path is configured but missing, and `allow_auto_download_missing` is `true`, the extension can fall back to the matching `fallback_*` Hugging Face reference at LLM execution time

### 3. Recommended beta setup

Recommended public references:

- base model: `Qwen/Qwen2.5-7B-Instruct`
  - https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- adapter: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
  - https://huggingface.co/yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora

On the first run, the extension can download both the base model and the adapter from Hugging Face.
Large first-time downloads are expected.

### 4. Hugging Face downloads and cache

This extension now supports automatic retrieval for both:

- `base_model_name_or_path`
- `adapter_path`

When either field is set to a Hugging Face repo ID, the extension uses standard Hugging Face cache behavior through `transformers`, `peft`, and `huggingface_hub`.

Recommended beta defaults:

- base model: `Qwen/Qwen2.5-7B-Instruct`
- adapter: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
- tokenizer source: adapter
- chat template source: adapter

Notes:

- the first run may take a while
- on the first run, the extension now waits for the full base-model snapshot download to finish before prompt generation returns control to Forge
- repeated runs should reuse the Hugging Face cache
- if `local_files_only` is `true`, uncached remote IDs will fail instead of downloading
- private or gated repos may require authentication
- this beta is designed so GitHub-distributed code can still recover by downloading weights at LLM runtime if local files are absent and fallback references are configured

## LoRA distribution policy

This repository does **not** bundle:

- base model checkpoints
- adapter weights
- training output directories

Users are expected to configure their own `base_model_name_or_path` and `adapter_path` in `configs/model_registry.json`.

Recommended public adapter name:

- `yoiko-Qwen2.5-7B-Instruct-lora`

Planned distribution split:

- GitHub repository: Forge extension code
- Hugging Face model repository: `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`
  - https://huggingface.co/yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora

Planned license split:

- Forge extension code: `AGPL-3.0-or-later`
- `yoiko-Qwen2.5-7B-Instruct-lora` adapter package: `Apache-2.0`

That local publish directory is staging material for LoRA release work and should not be committed back into the GitHub extension repository.

## Migration note

Old local experimental naming may still appear in local-only files under `artifacts/`.

- old experimental name: `fold4_best_eval_adapter`
- public release name: `yoiko-Qwen2.5-7B-Instruct-lora`

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
3. Set `base_model_name_or_path` to `Qwen/Qwen2.5-7B-Instruct`.
4. Set `adapter_path` and `tokenizer_name_or_path` to `yoikoarmor/yoiko-Qwen2.5-7B-Instruct-lora`.
5. Make sure `local_files_only` is `false`.
6. Start Forge and run one prompt generation.
7. Check the logs for Hugging Face download and PEFT attach messages.
8. Restore `artifacts/` later if you still want the local-copy workflow available.

## generation_defaults.json

`configs/generation_defaults.json` holds public-safe defaults for inference behavior.

Current defaults are intentionally conservative for beta use:

- `input_template_mode = simple_chat_template`
- `use_cache = false`
- `seed_mode = random`
- `debug_compare_input_variants = false`

`use_cache = false` means generation-time KV cache is disabled by default in this repo configuration.

Recommended beta interpretation:

- `seed_mode = random` gives more variation between runs
- `use_cache = false` favors simpler, easier-to-debug generation behavior
- if you want more reproducible debugging, switch `seed_mode` to `fixed`
- if prompt quality feels weak, adjust prompt wording before changing generation defaults aggressively

## Troubleshooting

### The extension loads but prompt generation does not run

Check:

- `Enable LLM Prompt Gen` is on
- `LLM Model` is not `none`
- `Gen Prompt` is not empty
- `configs/model_registry.json` exists and is valid JSON, or the example fallback is usable

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
- `base_model_class`
- `final_model_class`
- `is_peft_model`
- `active_adapter`
- `tokenizer_source`
- `chat_template_source`
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

The intended public LoRA package `yoiko-Qwen2.5-7B-Instruct-lora` is separate from this repository and may use a different license for the weight package itself.

The public Hugging Face package for `yoiko-Qwen2.5-7B-Instruct-lora` is released under `Apache-2.0`.

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
