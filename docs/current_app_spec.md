# 現行アプリ仕様書

更新日: 2026-04-03

## 1. 概要

このアプリは Stable Diffusion Forge 用の拡張機能 `sd-forge-llm-prompt-gen-yoiko` です。  
`txt2img` / `img2img` の生成前に LLM を使って `Gen Prompt` を整形し、最終 positive prompt を組み立てます。

現行の主対象:

- Forge 拡張コード: `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko`
- base model: `Qwen/Qwen3.5-4B`
- 公開 LoRA adapter: `yoikoarmor/yoiko-Qwen3.5-4B-lora`

## 2. 主要ファイル

- 拡張 UI / Forge フック:
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\scripts\forge_llm_prompt_gen.py`
- model registry:
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\configs\model_registry.example.json`
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\configs\model_registry.json`
- generation defaults:
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\configs\generation_defaults.json`
- backend:
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\backend\registry.py`
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\backend\loader.py`
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\backend\runtime.py`
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\backend\generator.py`
  - `D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\backend\prompt_builder.py`

## 3. UI 仕様

表示位置:

- `Gen Prompt` は既存 `Prompt` 欄の上に表示
- `txt2img` / `img2img` の両方で表示

表示項目:

- `Enable LLM Prompt Gen`
- `LLM Model`
- `LLM Load Mode`
- `Gen Prompt`
- `Prompt (Optional)`:
  - Forge 標準 positive prompt 欄
- `Negative prompt`:
  - Forge 標準 negative prompt 欄

意味:

- `Gen Prompt`
  - LLM に渡す主入力
- `Prompt (Optional)`
  - 最終 positive prompt に残す既存 prompt
  - 現行仕様では LLM 入力には混ぜない
- `Negative prompt`
  - 最終 negative prompt としてそのまま使う

## 4. 生成フロー仕様

### 4.1 `Enable LLM Prompt Gen = off`

- Forge の通常生成挙動を維持
- prompt 差し替えは行わない
- 生成ボタン押下時に、拡張 runtime に LLM が残っていれば `unload` する

期待ログ:

- `runtime unloaded current model`
- `runtime_action=disabled_and_unloaded_model`
  または
- `runtime_action=disabled_no_loaded_model`

### 4.2 `Enable LLM Prompt Gen = on` かつ `LLM Model = none`

- LLM は呼ばない
- manual prepend 動作

最終 prompt:

```text
final_positive = dedupe("Gen Prompt, Prompt")
final_negative = Negative prompt
```

### 4.3 `Enable LLM Prompt Gen = on` かつ実モデル選択

#### `Gen Prompt` が空

- LLM は呼ばない
- `Prompt (Optional)` をそのまま使う

#### `Gen Prompt` が非空

- LLM を 1 回だけ呼ぶ
- `Gen Prompt` を LLM 入力へ渡す
- `Prompt (Optional)` は LLM 入力に入れない
- `Negative prompt` は LLM に「positive に混ぜない参照情報」として渡す

現行の最終組み立て:

```text
processed_gen_prompt = LLM 出力を最小限 cleaning した文字列
final_positive = dedupe(processed_gen_prompt + ", " + original_prompt)
final_negative = negative_prompt
```

補足:

- `original_prompt` が空なら `final_positive = processed_gen_prompt`
- 完全一致タグは comma 区切りで重複除去する

## 5. Prompt builder 仕様

system prompt の方針:

- 元入力の主題・属性・意図を維持する
- 不足する視覚情報や雰囲気を適度に補完する
- 過剰に vivid / dramatic / flashy にしない
- positive prompt だけを返す
- negative prompt 用語を positive に混ぜない

入力テンプレート:

- 既定値: `simple_chat_template`
- 互換用: `forge_prompt_builder`

現行既定の user 側入力構造:

- `User input:` に `Gen Prompt`
- `Negative prompt reference (do not include these terms):` に negative
- 補足 instruction:
  - 元の idea / traits を保つ
  - moderate amount の detail を足す

重要:

- `Prompt (Optional)` は現行の LLM 入力に含めない

## 6. 出力後処理

実装方針:

- aggressive cleaning はしない
- 最小限の整形だけ行う

主な処理:

- trim
- 改行の軽い正規化
- `Prompt:` / `Positive prompt:` など既知接頭辞の除去
- 外側だけの単純な quote 除去
- comma 区切り重複除去

短すぎる出力の fallback 条件:

- 空文字
- 10 文字未満
- comma 区切り要素が 2 未満

fallback 時:

```text
processed_gen_prompt = dedupe(Gen Prompt)
final_positive = dedupe(processed_gen_prompt + ", " + original_prompt)
```

## 7. 推論モデル仕様

現行推奨:

- base model: `Qwen/Qwen3.5-4B`
- adapter: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- tokenizer source: `adapter`
- chat template source: `adapter`
- quantization: `4-bit NF4`
- compute dtype: `bfloat16`
- double quant: `true`
- device_map: `auto`

`model_registry.json` 既定の主項目:

- `base_model_name_or_path`
- `adapter_path`
- `tokenizer_name_or_path`
- `cache_dir`
- `fallback_base_model_name_or_path`
- `fallback_adapter_path`
- `fallback_tokenizer_name_or_path`
- `allow_auto_download_missing`
- `load_in_4bit`
- `bnb_4bit_quant_type`
- `bnb_4bit_compute_dtype`
- `use_double_quant`
- `device_map`
- `torch_dtype`
- `trust_remote_code`
- `local_files_only`
- `tokenizer_source`
- `chat_template_source`
- `use_fast_tokenizer`

## 8. ダウンロード / キャッシュ仕様

対応入力:

- ローカルパス
- Hugging Face repo ID

判定:

- 実在するローカルパスなら local
- `owner/model` 形式でローカル実体がなければ Hugging Face 扱い

現行の自動取得仕様:

- base model / adapter / tokenizer のいずれも HF repo ID 対応
- `allow_auto_download_missing=true` の場合:
  - 設定された local path が存在しなくても
  - `fallback_*` があれば LLM 実行時に HF repo ID へ自動フォールバック

初回実行時の重要仕様:

- base model は **snapshot を最後まで取得してから** load する
- 4 shard など複数ファイル構成でも、download 完了前に画像生成へ戻らない

既定 cache:

- `cache_dir = null` の場合は Hugging Face 標準 cache

## 9. Runtime 管理仕様

モード:

- `keep_loaded`
  - 同一 signature のモデルをメモリ再利用
- `load_then_unload_before_image_gen`
  - prompt generation 後に unload

off 時:

- `Enable LLM Prompt Gen = off` で Generate 押下時にも unload 実行

unload 時の処理:

- `del model`
- `del tokenizer`
- `gc.collect()`
- `torch.cuda.empty_cache()`

## 10. generation_defaults.json の現行既定値

`D:\stablematrix\Data\Packages\forge-test\extensions\sd-forge-llm-prompt-gen-yoiko\configs\generation_defaults.json`

現行値:

- `max_new_tokens = 128`
- `do_sample = true`
- `temperature = 0.7`
- `top_p = 0.9`
- `top_k = null`
- `repetition_penalty = 1.0`
- `input_template_mode = simple_chat_template`
- `cache_implementation = dynamic`
- `use_cache = false`
- `seed_mode = random`
- `llm_seed = 42`
- `debug_compare_input_variants = false`

seed 仕様:

- 既定: `random`
- `fixed` に変えると `llm_seed` を使用

## 11. ログ仕様

主に確認するログ:

- `llm_load_config`
- `base_model_source`
- `adapter_source`
- `resolved_base_model_reference`
- `resolved_adapter_reference`
- `base_download_started`
- `base_download_finished`
- `adapter_download_started`
- `adapter_download_finished`
- `final_model_class`
- `is_peft_model`
- `active_adapter`
- `tokenizer_source`
- `chat_template_source`
- `gen_prompt_raw`
- `negative_prompt_raw`
- `system_prompt_preview`
- `user_prompt_preview`
- `raw_model_output`
- `cleaned_output`
- `fallback_used`
- `final_positive_before_dedupe`
- `final_positive_after_dedupe`
- `final_negative_preview`
- `llm_seed`
- `seed_mode`

## 12. 依存関係

現行 requirements:

- `transformers`
- `huggingface_hub`
- `peft`
- `accelerate`
- `bitsandbytes`
- `safetensors`

前提:

- NVIDIA GPU + CUDA 系環境を主対象
- 4-bit bitsandbytes 推論が使えることを想定

## 13. ライセンスと配布方針

Forge 拡張コード:

- `AGPL-3.0-or-later`

LoRA 公開物:

- repo 同梱しない
- 公開名: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- 配布先: Hugging Face
- LoRA package 側ライセンス: `Apache-2.0`

base model:

- `Qwen/Qwen3.5-4B`
- base model のライセンス条件は別途確認が必要

## 14. 既知の制約

- beta / experimental 品質
- 単一 `model.generate()` の途中 interrupt は未対応
- prompt shaping はモデルや adapter により手調整余地あり
- 初回 Hugging Face download は時間・容量が必要
- offline 利用には local path または HF cache が必要
- Windows では HF cache の symlink warning が出る場合がある

## 15. 現行推奨設定

もっとも現行仕様に合っている推奨セット:

- `Enable LLM Prompt Gen = on`
- `LLM Model = qwen3.5-4b`
- `LLM Load Mode = keep_loaded`
- `Gen Prompt = 主入力`
- `Prompt (Optional) = 既存 positive の補足`
- base model: `Qwen/Qwen3.5-4B`
- adapter: `yoikoarmor/yoiko-Qwen3.5-4B-lora`
- `tokenizer_source = adapter`
- `chat_template_source = adapter`
- `load_in_4bit = true`
- `bnb_4bit_quant_type = nf4`
- `bnb_4bit_compute_dtype = bfloat16`
- `seed_mode = random`

