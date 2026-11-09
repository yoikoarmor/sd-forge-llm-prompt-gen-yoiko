# Neo-3 selective migration

Baseline: `265ea0e03dbe9de36de63df005a9e452181f3841`. The installed Neo extension
and its user settings were not modified. Validation used a separate source copy;
these changes have not been installed into the live UI.

## Preserved Neo behavior

- Native in-process `llama-cpp-python` remains the `llama_cpp` backend. Its
  `gguf_path`, `gguf_repo_id`, `gguf_filename`, `gguf_lora_path`, `n_ctx`,
  `n_gpu_layers`, `n_batch`, `n_threads`, `flash_attn`, and thinking controls remain.
- Example/user registry merging, custom overrides, disabled entries, and the three
  original enabled GGUF entries remain. The user's registry was not rewritten.
- Weight Mode, Max New Tokens (including the existing 225-token cap), Candidate
  Count, Preview Candidates, radio selection, and selected-candidate state remain.
  The script returns eight UI values; the processing object is a separate hook
  argument.
- Image-model CPU RAM offload, the 9B automatic throughput profile, and fallback
  to the original precision remain. A failed throughput attempt now releases its
  traceback and cache before attempting NF4.
- Runtime compatibility shims, install protections, all requirements files, JS,
  CSS, and generation defaults are unchanged.

`artifacts/neo_preservation_evidence.json` records byte equality for ten protected
files and AST equality for eleven preserved functions. The baseline mixed line
endings are retained on unchanged lines to keep the source diff reviewable.

## Selected changes

| Area | Result |
| --- | --- |
| Prompt input | The evaluated family system prompt and raw trimmed user request are used by `simple_chat_template`. Original/negative prompts are excluded from that input. The explicit `forge_prompt_builder` path remains available. |
| Template tokens | Qwen's empty no-thinking block is preserved. Rendered templates do not receive a second set of special tokens. |
| Prompt output | Commas inside balanced weights/schedules are preserved during dedupe. Truncated or mismatched brackets trigger fallback rather than merging into the original prompt. Incomplete thinking markup is discarded. |
| HF generation | Zero pad/top-k values are respected; RNG states are restored; token-level Stop discards partial results. Multiple candidates and greedy beam selection work. The baseline undefined `raw_prompt_text` reference is removed. |
| Native GGUF | Streaming checks Stop during a completion, closes the stream, and discards all partial candidates. Optional repository revision and SHA256 are enforced. `gguf_lora_path` is forwarded to native `lora_path`. |
| Failure cleanup | Tokenization/device transfer and generation OOMs are classified. Model unload occurs after exception handlers release traceback references. Failed loads preserve textual diagnostics while freeing partial models. Persistent CUDA BLAS scratch is isolated using public Torch APIs when available. |
| Forge hooks | Scalar/batched prompt and negative mappings are preserved. Rewrites are reused for identical negative references and committed only after the batch completes. Stop leaves every original prompt intact. Requested unload also occurs when generation is skipped. |
| Model loading | Base/adapter/tokenizer revision pins, standard HF cache snapshots, adapter SHA256, tokenizer assets, optional PEFT, and attention implementation are supported. |

The older external `llama-server` backend, process lifecycle assumptions such as
`is_alive()`, and its incompatible registry fields were deliberately excluded.
No server executable is required for the native Neo route.

## Model configuration

`configs/lora_family.json` records the seven independent best-eval epoch-3
adapters, their hashes, fixed base revision, and exact system prompt. The seven
adapter presets and the merged HF preset are opt-in. They are alternatives, not
seven simultaneously stacked adapters.

The companion [BF16 merged model](https://huggingface.co/yoikoarmor/Qwen3.5-9B-Instruct-7-merged-bf16)
is public at verified commit `6fb87cfd99f44b75bef9b72bf8e94e5c5d7ce72c`.
The preset is pinned but remains disabled by default because the installed Neo
Transformers 4.57.6 cannot load `qwen3_5_text`. A compatible separate runtime is
required; no dependency upgrade is performed by this configuration change.
Public file sizes and LFS hashes, plus the downloaded model card and checksum
manifest, were checked against the release files.

The published Q4 preset `qwen3.5-9b-lora-7-q4km` is enabled using native fields:

- Repository: `yoikoarmor/Qwen3.5-9B-Instruct-7-GGUF`
- Commit: `b34bc0c3e42d9ad91b19f4755b9039c5ec987fbd`
- File: `Qwen3.5-9B-Instruct-7-Q4_K_M.gguf`
- SHA256: `7480cb03124924db66bd72d2dcb78eb73829200e1b7727f273299675c76f49ff`

The public repository, final commit, file size, LFS SHA256, model card, and
upstream license were verified using anonymous access. The native Neo GPU
validation is summarized below.

### Instruct 7 embedded-template opt-in

The new Instruct 7 Q4 preset alone sets `gguf_prompt_mode: "embedded_template"`.
The default is `"legacy"`, preserving the original entries' `/no_think` behavior.
The opt-in route renders GGUF metadata with the installed `Jinja2ChatFormatter`,
passing `enable_thinking=False` from the generation settings. It leaves the user
message unchanged and preserves the template's empty assistant think block.
Explicit `enable_thinking=True` is also passed through to the template.

The rendered prompt is tokenized once using native
`tokenize(add_bos=False, special=True)` and passed as token IDs to completion,
along with the formatter's stop strings. Missing or invalid embedded templates
produce an error without silently switching to the legacy input. Full rendered
text and length are available in `GenerationResult.input_debug.raw_prompt_text`
and `tokenized_input_length`; the native completion helper receives token IDs.

`artifacts/neo_embedded_prompt_validation.json` confirms exact equality with the
previous HF/comparison input: **218 tokens**, compared with legacy's 222. The
validation uses actual Neo Jinja rendering and the actual Q4 vocabulary loaded
with `vocab_only=True, n_gpu_layers=0`; completion is mocked and no GPU inference
is performed. `artifacts/verify_embedded_prompt_cpu.py` reproduces this check.

`tools/configure_lora_family.py` validates all requested files/hashes before an
atomic registry replacement. Existing custom entries are preserved. Example:

```powershell
python tools/configure_lora_family.py --adapter-root C:\models\verified-adapters
python tools/configure_lora_family.py --merged-model C:\models\merged-text-model
python tools/configure_lora_family.py --gguf-model C:\models\Qwen3.5-9B-Instruct-7-Q4_K_M.gguf
```

Remote merged/GGUF configuration requires a verified 40-character commit with
`--merged-revision` / `--gguf-revision`. No downloads or model modification are
performed by this configuration tool.

## Validation and remaining gates

GPU was masked with `CUDA_VISIBLE_DEVICES=-1` and Python bytecode writes were
disabled with `-B`. No dependencies were installed or upgraded.

| Environment | Result |
| --- | --- |
| Actual Neo-3: Python 3.13.12, Torch 2.11.0+cu130, Transformers 4.57.6, Hub 0.36.2 | 95 tests: 94 passed, 1 GPU-only test skipped |
| forge-test: Transformers 5.5.0, Torch 2.9.0+cu128, Hub 1.9.0 | 95 tests: 94 passed, 1 GPU-only test skipped |
| Python source compilation and `git diff --check` | Passed |

Logs: `artifacts/regression-neo313.log`, `artifacts/regression-forge55.log`.
These are unit/regression tests, using real Torch/tokenizers on CPU and mocked
model/native calls. They do not establish live UI or model inference compatibility.

Independent code review identified three issues during migration: lost
rejection of untagged reasoning, ignored tokenizer revisions when sharing a
base/adapter repository, and unclassified native GGUF startup `MemoryError`.
All were fixed and covered by regression tests. The tokenizer revision tests
cover explicit and source-selected references. A subsequent review of the
embedded-template opt-in found no further major issue.

### Real Neo GPU validation

The new embedded-template route was tested in the actual Neo environment using
its installed `llama-cpp-python 0.3.28`, Torch 2.11.0+cu130, and RTX 5090. No
dependencies, installed extension files, or user settings were changed.

- The rendered prompt and all 218 token IDs matched case 1 of the 20-input
  comparison exactly. System text, raw user text, and the empty think block
  were preserved. Existing entries retained their legacy `/no_think` behavior.
- Loading, generation, keep-loaded reuse, streamed Stop with partial output
  discarded, retry, unloading, native object collection, and reloading passed.
- Initial loading took 5.22 seconds. The model used approximately 5.77 GB of GPU
  memory. After final unload, available GPU memory was 65.0 MB below the starting
  reading; Torch allocated and reserved memory were both zero. Device-wide
  readings include background desktop activity and do not promise zero retained
  CUDA context memory.
- A completion deliberately limited to 64 tokens ended with unbalanced syntax
  and correctly used the manual fallback. This verifies the guard, not quality
  of the model's prompts.

The local evidence is in `artifacts/neo_native_validation_summary_embedded.md`,
`neo_native_q4_validation_embedded.json`, and the corresponding script/log.
Validation logs and local settings are not included in the source archive.

Actual Neo-3 Transformers 4.57.6 lacks both `qwen3_5` and `qwen3_5_text` mappings.
The merged model therefore cannot run through that HF runtime. Local unsupported
configs now fail clearly before allocating weights and preserve the original
image prompt. Use a separate compatible environment (5.3/5.5 were previously
verified for the text-only model) or a compatible native GGUF route. This work
does not upgrade the image-generation environment or add a separate HF service.

Live Neo UI, txt2img, and img2img runs remain untested. Model-loading and prompt
prefill cannot be interrupted immediately; Stop is honored after loading and at
streamed output boundaries. OOM recovery tests simulate allocation failures; no
deliberate GPU-memory exhaustion test was performed.

The new weighting system prompt is shared with existing presets, so their text
outputs can change even though the original UI, registry behavior, and runtime
features are retained. Generated prompts can still repeat concepts, ignore
weighting priorities, or reach the output limit. Review them before use.
