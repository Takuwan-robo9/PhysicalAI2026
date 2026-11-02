# 言葉で指示できる片付けロボット (Language-Driven Tidy-up Robot)

Physical AI 応用講座 最終課題.
MuJoCo 上の Franka Panda アームが、**日本語の自然な指示**（「トマトみたいな色のブロックを、空の色のお皿にのせて」など）を
**ローカル LLM** で作業計画に変換し、**RGB-D カメラ認識**で物体の位置を求めて、ピック&プレースで片付けます。

- API キー不要・GPU 不要（ノート PC の CPU のみで動作）
- シミュレータの「正解位置」は制御に使わず、カメラ画像だけから物体を認識
- LLM の出力を物理的な制約で検証し、誤った順序などを**自動修復**

## システム構成

```
 上部カメラ (RGB + 深度)
        │  perception.py : 逆投影 → HSV 色分割 → 高さで「ブロック / お皿」分離
        ▼                  → 形状 (立方体 / 円柱) と向き (yaw) を推定
 世界モデル (app.py WorldModel)
        │  「何が何の上にあるか」を管理。上に積まれて見えなくなった物体は記憶で補完
        ▼
 日本語の指示 ──▶ ローカル LLM (Qwen2.5-3B-Instruct, GGUF 4bit, llama.cpp)
        │  planner.py : JSON スキーマ制約付き生成 → {"reply", "steps":[{pick, place}]}
        ▼
 検証・修復層 (planner.validate)
        │  存在しない物体 / お皿を持つ / 載っている物の下を抜く 等を検出し、
        │  順序の入れ替えや「一旦どかす」手順の追加で自動修復
        ▼
 動作生成 (robot.py)
        │  減衰最小二乗法の逆運動学 + 最小躍度軌道、把持向きを物体の yaw に合わせる
        ▼
 実行後に再認識して成功判定 → 失敗なら再挑戦
```

## 動かし方

```bash
pip install mujoco numpy opencv-python pillow huggingface_hub
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python download_model.py            # Qwen2.5-3B-Instruct (GGUF, 約2.1GB) を models/ に保存
python run_demo.py                  # 既定シナリオを実行し outputs/demo.mp4 を生成
python run_demo.py --interactive    # キーボードから指示を入力
python run_demo.py --commands "赤いブロックを青いお皿に置いて" "丸いのをピンクのお皿へ"
```

評価:

```bash
python eval/eval_manipulation.py --trials 20   # 認識 + ピック&プレース成功率
python eval/eval_planner.py                    # LLM 計画の正解率 (3B / 1.5B, 修復層の有無)
```

## 評価結果 (Core i5-1235U / メモリ 8GB / GPU なし)

| 評価 | 結果 |
|---|---|
| 認識 + ピック&プレース (ランダム配置 20 試行 × 3 ステップ) | 59/60 ステップ成功 (98.3%) |
| 物体位置の認識誤差 (平均 / 最大) | 0.34 mm / 0.69 mm |
| LLM 計画の正解率 Qwen2.5-3B (生出力 → 検証・修復後) | 80.0% → 93.3% (15 ケース) |
| LLM 計画の正解率 Qwen2.5-1.5B (生出力 → 検証・修復後) | 46.7% → 60.0% |
| LLM 推論時間 (3B, 中央値) | 約 16 秒 |

## ファイル構成

| パス | 内容 |
|---|---|
| `src/perception.py` | RGB-D 画像からの物体認識 (色・形状・3D 位置・向き) |
| `src/planner.py` | ローカル LLM による計画生成、JSON スキーマ制約、検証・自動修復 |
| `src/robot.py` | MuJoCo 環境、逆運動学、軌道生成、ピック&プレース |
| `src/app.py` | 世界モデル、全体ループ (認識→計画→実行→確認)、動画合成 |
| `run_demo.py` | デモ実行 |
| `eval/` | 定量評価スクリプト |
| `assets/franka_emika_panda/` | ロボットモデル ([MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache-2.0) と自作シーン `tidy_scene.xml` |

## クレジット

- Franka Emika Panda モデル: MuJoCo Menagerie (Apache License 2.0) — `assets/franka_emika_panda/LICENSE`
- LLM: [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Qwen Research License, 非商用) / Qwen2.5-1.5B-Instruct (Apache License 2.0)。モデル重みはリポジトリに含めず `download_model.py` で取得
- 推論エンジン: [llama.cpp](https://github.com/ggerganov/llama.cpp) / llama-cpp-python
