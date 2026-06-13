# GGUF LoRA Phase 2 Notes

更新日: 2026-06-12

## 目的

今回の GGUF 対応では、base model + LoRA を先に merge し、その merged model を GGUF 化する方式を採用した。
フェーズ2候補として、llama.cpp の `convert_lora_to_gguf.py` で LoRA を GGUF 化し、llama.cpp / llama-cpp-python の LoRA 適用機能で読み込む方式を調査した。

## 候補方式

### 1. merge 済み GGUF 方式

現在採用している方式。

- base model と LoRA を Transformers / PEFT 側で merge
- merged HF model を `convert_hf_to_gguf.py` で GGUF 化
- 必要なら `llama-quantize` で Q4_K_M などへ量子化
- Forge 実行時は単一 GGUF を読むだけ

利点:

- 実行時の構成が単純
- adapter の付け間違いが起きにくい
- 配布・設定・smoke test が分かりやすい
- llama.cpp 側の LoRA 適用差異に依存しにくい

欠点:

- LoRA を差し替えるたびに再merge・再変換が必要
- adapter 単体配布よりファイルサイズが大きくなる

## 2. `convert_lora_to_gguf.py` + `lora_path` 方式

フェーズ2候補。

- base model を GGUF 化
- LoRA adapter を `convert_lora_to_gguf.py` で GGUF LoRA 化
- 実行時に base GGUF + LoRA GGUF をロード

利点:

- base GGUF を共有し、LoRA だけ差し替えられる
- 複数 adapter の検証が速くなる可能性がある
- adapter 単体配布に近い形を保てる

懸念:

- quantized base に LoRA を適用する場合、品質・速度・互換性が merge 済み GGUF と一致しない可能性がある
- llama-cpp-python 側の LoRA API と wheel version 差分に依存する
- Qwen3.5 系 adapter の target module / naming と llama.cpp 側 converter の対応確認が必要
- Forge UI では base GGUF と LoRA GGUF のペア管理が必要になり、設定が複雑になる

## 現時点の判断

公開初期版では merge 済み GGUF 方式を優先する。

理由:

- clone 後の利用説明が単純
- `LLM Model` 1 entry = 1 GGUF という形にできる
- 実行時の失敗点が少ない
- 既存 Transformers + LoRA 経路との責務分離が明確

フェーズ2で `gguf_lora_path` を有効化する場合は、次を確認してから進める。

- `convert_lora_to_gguf.py` が対象 adapter を変換できること
- llama-cpp-python の対象 version で LoRA 適用 API が安定していること
- base GGUF + LoRA GGUF と merged GGUF の出力傾向を比較すること
- `load_then_unload_before_image_gen` で LoRA 適用済みモデルも正しく解放されること

