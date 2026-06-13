# automation — ランダム入力オートメーション + 品質自動検査

`Gen Prompt` の入力からランダム生成し、Forge API 経由で画像を作り、品質を自動採点して SQLite に蓄積するパイプラインです。拡張本体のコードは一切変更していません(`AlwaysVisible` スクリプトなので `alwayson_scripts` でAPI注入できます)。

## 全体フロー

```
wordpool.json からランダムサンプリング → Gen Prompt
        ↓
Forge API /sdapi/v1/txt2img (alwayson_scripts で拡張に注入)
        ↓
画像 + infotext (llm generated / llm fallback) を保存
        ↓
採点: CLIPスコア → (あれば)蒸留Discriminator → VLM判定
        ↓
output/runs.sqlite3 に蓄積 → report.py で弱点分析
```

## 使い方

Forge を `--api` 付きで起動しておくこと。

```bat
rem 20枚生成 → 採点 → レポート まで一括
run_pipeline.bat 20
```

個別実行(Forge venv の python を使う):

```bat
set PY="D:\stablematrix\Data\Packages\Stable Diffusion WebUI Forge - Neo\venv\Scripts\python.exe"
%PY% runner.py --count 20        rem 生成のみ
%PY% scorer.py                   rem 未採点分をまとめて採点
%PY% report.py                   rem レポート(--csv export.csv 可)
%PY% distill.py                  rem Discriminator蒸留(200件以上で)
```

## 品質検査の3層

1. **CLIPスコア**(常時・軽量): `Gen Prompt` と生成画像のCLIP類似度。LLM拡張後も元の意図が画像に残っているかの連続値メトリクス。同時に画像埋め込みをDBに保存します(蒸留用)。
2. **VLM判定**(`Qwen2.5-VL-7B` 4bit): prompt_fidelity / anatomy / artifacts / aesthetics / overall を0-10でJSON採点。`config.json` の `scoring.enable_vlm` で無効化できます。
3. **蒸留Discriminator**: VLM採点が約200件貯まったら `distill.py` で小型MLP(CLIP埋め込み→overall予測)を学習。以後 `scorer.py` が自動で使い、**VLMを読まずに一瞬で採点**できます。VLM教師→生徒の蒸留なので、学習データを別途用意する必要がありません。精度(val MAE)が十分になったら `enable_vlm: false` にしてDiscriminator単独運用に移行できます。

> GAN の Discriminator が欲しいが学習データがない、という問題への回答がこの3層目です。パイプラインを回すこと自体が学習データ収集になります。

## VRAM管理(重要)

- CLIP / VLM はすべてコンテキストマネージャで管理。**例外発生時も含めて** `参照破棄 → gc.collect() → torch.cuda.empty_cache() → torch.cuda.ipc_collect()` が必ず実行されます。アンロード前後のVRAM使用量がログに出ます。
- VLMフェーズの前に Forge 側のチェックポイントもAPIで自動アンロードし(`scoring.unload_forge_checkpoint_during_vlm`)、採点後に自動リロードします。
- 採点プロセスは Forge と別プロセスなので、拡張側LLM(`keep_loaded`)とVLMがVRAMを奪い合うことはありません。気になる場合は `llm_extension.load_mode` を `load_then_unload_before_image_gen` に変更してください。

## config.json の主な項目

| 項目 | 意味 |
|---|---|
| `generation.count` | 1回のバッチ生成枚数 |
| `llm_extension.model` | 拡張側LLM(既定 `qwen3.5-4b`) |
| `scoring.vlm_model` | 判定VLM(HF repo ID) |
| `scoring.enable_vlm` | VLM判定のon/off |
| `scoring.unload_forge_checkpoint_during_vlm` | VLM中にForgeモデルを退避 |

## レポートで分かること

- **LLM fallback率**: 拡張内LLMの出力が短すぎて棄却された率(プロンプト層の品質)
- **VLM軸別平均**: fidelity / anatomy / artifacts / aesthetics
- **弱点ワードプール項目**: スコア平均が低い subject / style など → LoRA学習データ改善の候補
- **ワーストrun一覧**: 画像ファイル名付きで目視確認へ

## wordpool.json

カテゴリ別の語彙プール。`required.subject` は必ず入り、`optional.*` は `probability` で混入率を制御。自分のLoRAの想定ドメインに合わせて自由に編集してください。
