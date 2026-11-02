"""認識 + ピック&プレースの成功率評価 (LLM なし).

ランダムな位置・向きに物体を配置し、カメラ認識 → 把持 → 目的地へ配置 → 再認識で確認、を繰り返す.
認識誤差はシミュレータ内の真の位置と比較して算出する (評価のためだけに正解を参照).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from app import TidyRobotApp, WorldModel  # noqa: E402
from robot import MOVABLE  # noqa: E402


class _NoPlanner:
    model_name = "none"


def random_layout(rng):
    while True:
        pts = [(rng.uniform(0.45, 0.68), rng.uniform(-0.17, 0.17)) for _ in MOVABLE]
        if all(np.hypot(a[0] - b[0], a[1] - b[1]) > 0.08 for i, a in enumerate(pts) for b in pts[i + 1:]):
            return {n: (x, y, rng.uniform(-np.pi / 4, np.pi / 4)) for n, (x, y) in zip(MOVABLE, pts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows, errs = [], []
    t0 = time.time()
    app = TidyRobotApp(None, planner=_NoPlanner())
    for k in range(args.trials):
        app.env.reset(random_layout(rng))
        app.world = WorldModel()
        app.observe()
        errs += list(v for kk, v in app.perception_error().items() if "plate" not in kk)
        blocks = [b for b in MOVABLE]
        rng.shuffle(blocks)
        # 1 試行 = 3 ステップ: お皿へ / 別のお皿へ / 積み上げ
        task = [(blocks[0], "blue_plate"), (blocks[1], "pink_plate"), (blocks[2], blocks[3])]
        res = []
        for p, q in task:
            if q not in app.world.dets and q != "table":
                res.append(False)
                continue
            res.append(app.execute_step({"pick": p, "place": q}, max_attempts=2))
        rows.append({"trial": k, "task": task, "success": res})
        print(f"trial {k:2d}: {res}", flush=True)
    flat = [s for r in rows for s in r["success"]]
    by_type = {name: float(np.mean([r["success"][i] for r in rows])) for i, name in
               enumerate(["plate_1", "plate_2", "stack"])}
    summary = {"trials": args.trials, "step_success_rate": float(np.mean(flat)), "by_type": by_type,
               "perception_err_mm_mean": float(np.mean(errs)), "perception_err_mm_max": float(np.max(errs)),
               "wall_time_s": time.time() - t0}
    print(summary)
    (ROOT / "outputs" / "eval_manipulation.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
