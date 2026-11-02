"""デモ実行スクリプト.

例:
  python run_demo.py                       # 既定のシナリオを実行して outputs/demo.mp4 を作る
  python run_demo.py --interactive         # 指示をキーボードから入力 (動画も保存)
  python run_demo.py --commands "赤いブロックを青いお皿に置いて" "丸いのをピンクのお皿へ"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from app import TidyRobotApp  # noqa: E402

DEFAULT_COMMANDS = [
    "トマトみたいな色のブロックを、空の色のお皿にのせて",
    "バナナ色のブロックを、青いブロックの上に積んで",
    "丸いのをピンクのお皿へ",
    "ダンスして",
    "青いブロックもピンクのお皿へ",
]

# 物体の初期配置 (x, y, yaw[rad]). 向きをばらばらにして yaw 推定の効果を見せる
DEFAULT_LAYOUT = {
    "red_cube": (0.56, 0.10, 0.45),
    "yellow_cube": (0.64, -0.12, 0.15),
    "blue_cube": (0.48, -0.06, -0.35),
    "green_cylinder": (0.66, 0.15, 0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", nargs="*", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "demo.mp4"))
    ap.add_argument("--model", default=None, help="GGUF モデルのパス (既定: models/qwen2.5-3b-instruct-q4_k_m.gguf)")
    ap.add_argument("--speed", type=float, default=1.0, help="出力動画の再生速度倍率")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    app = TidyRobotApp(None if args.no_video else out, Path(args.model) if args.model else None, DEFAULT_LAYOUT)
    vc = app.vc
    t0 = time.time()
    if vc:
        vc.card("言葉で指示できる片付けロボット",
                ["MuJoCo + Franka Panda / ローカルLLM (Qwen2.5-3B, CPU) / RGB-D 認識",
                 "",
                 "① 上部カメラの RGB-D 画像から物体を認識",
                 "② 日本語の指示をローカル LLM が作業計画 (JSON) に変換",
                 "③ 計画を物理的に検証・自動修復",
                 "④ 逆運動学でピック&プレース → 再認識で成功を確認",
                 "",
                 f"再生速度: {args.speed:g}倍速" + ("" if args.speed == 1 else " (LLM 推論待ちは短縮表示)")],
                5.0)
    app.observe(move=True)
    print("初期認識誤差 [mm]:", {k: round(v, 2) for k, v in app.perception_error().items()})

    if args.interactive:
        while True:
            try:
                cmd = input("指示> ").strip()
            except EOFError:
                break
            if cmd in ("", "q", "quit", "exit"):
                break
            app.run_command(cmd)
    else:
        for cmd in args.commands or DEFAULT_COMMANDS:
            app.run_command(cmd)

    n_ok = sum(sum(r["success"]) for r in app.log)
    n_all = sum(len(r["success"]) for r in app.log)
    if vc:
        vc.card("結果", [f"指示 {len(app.log)} 件 / ピック&プレース {n_ok}/{n_all} 成功",
                         "LLM 推論時間 (CPU): " + ", ".join(f"{r['latency']:.1f}s" for r in app.log),
                         "", "コード: GitHub (概要欄参照)"], 4.0)
        vc.close(speed=args.speed)
    log = {"model": app.planner.model_name, "commands": app.log, "wall_time_s": time.time() - t0,
           "final_perception_error_mm": app.perception_error()}
    log_path = out.with_suffix(".json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"done: {n_ok}/{n_all} steps succeeded. video={out if vc else '-'} log={log_path}")


if __name__ == "__main__":
    main()
