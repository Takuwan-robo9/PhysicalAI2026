"""LLM プランナーの正解率評価 (モデルサイズ別 / 検証・修復層の有無).

各テストケースは (シーン, 指示, 正解の手順集合) からなる.
完全一致 = 手順の集合と、依存関係 (上の物を先に動かす) を満たす順序が正しいこと.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planner import LLMPlanner, validate  # noqa: E402


def scene(**on):
    base = {"red_cube": ("red", "cube"), "yellow_cube": ("yellow", "cube"),
            "green_cylinder": ("green", "cylinder"), "blue_cube": ("blue", "cube")}
    objs = {k: {"color": c, "shape": s, "on": on.get(k, "table")} for k, (c, s) in base.items()}
    for k, v in on.items():          # 上に載っているブロックは hidden を付ける
        if v in objs:
            objs[v]["hidden"] = k
    objs["blue_plate"] = {"color": "blue", "shape": "plate"}
    objs["pink_plate"] = {"color": "pink", "shape": "plate"}
    return objs


S0 = scene()
S1 = scene(red_cube="blue_plate", yellow_cube="blue_cube")
S2 = scene(yellow_cube="blue_cube")
CASES = [
    (S0, "赤いブロックを青いお皿に置いて", [("red_cube", "blue_plate")]),
    (S0, "トマトみたいな色のブロックを、空の色のお皿にのせて", [("red_cube", "blue_plate")]),
    (S0, "バナナ色のブロックを、青い四角の上に積んで", [("yellow_cube", "blue_cube")]),
    (S0, "丸いのをピンクのお皿へ", [("green_cylinder", "pink_plate")]),
    (S0, "四角いブロックを全部ピンクのお皿に片付けて",
     [("red_cube", "pink_plate"), ("yellow_cube", "pink_plate"), ("blue_cube", "pink_plate")]),
    (S0, "青いブロックを同じ色のお皿に置いて", [("blue_cube", "blue_plate")]),
    (S0, "緑のブロックの上に赤いのを重ねて", [("red_cube", "green_cylinder")]),
    (S0, "ダンスして", []),
    (S0, "コーヒーを入れて", []),
    (S2, "トマトみたいな色のブロックを、空の色のお皿にのせて", [("red_cube", "blue_plate")]),
    (S2, "丸いのをピンクのお皿へ", [("green_cylinder", "pink_plate")]),
    (S1, "丸いのをピンクのお皿へ", [("green_cylinder", "pink_plate")]),
    (S1, "青いブロックもピンクのお皿へ", [("yellow_cube", "table"), ("blue_cube", "pink_plate")]),
    (S1, "積んだ2つのブロックを、両方ともピンクのお皿に移して",
     [("yellow_cube", "pink_plate"), ("blue_cube", "pink_plate")]),
    (S1, "テーブルに残っているものを全部ピンクのお皿に片付けて",
     [("green_cylinder", "pink_plate"), ("yellow_cube", "pink_plate"), ("blue_cube", "pink_plate")]),
]


def correct(steps: list[dict], expected: list[tuple], objs: dict) -> bool:
    got = [(s["pick"], s["place"]) for s in steps]
    if sorted(got) != sorted(expected):
        return False
    # 物理的な順序制約: 上に物が載ったブロックは、上の物を動かした後でないとつかめない
    on = {k: o.get("on") for k, o in objs.items() if o["shape"] != "plate"}
    for p, q in got:
        if any(v == p for v in on.values()):
            return False
        on[p] = q
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5-3b-instruct-q4_k_m.gguf",
                                                     "qwen2.5-1.5b-instruct-q4_k_m.gguf"])
    args = ap.parse_args()
    results = {}
    for m in args.models:
        planner = LLMPlanner(ROOT / "models" / m)
        rows = []
        for objs, instr, exp in CASES:
            plan = planner.plan(instr, objs)
            try:   # 検証・修復層を通す前の LLM 生出力
                raw_steps = [{"pick": s.get("pick"), "place": s.get("place")}
                             for s in json.loads(plan.raw).get("steps", [])]
            except json.JSONDecodeError:
                raw_steps = []
            raw_ok = correct(raw_steps, exp, objs)
            ok = correct(plan.steps, exp, objs)
            rows.append({"instruction": instr, "raw": plan.raw, "final": plan.steps, "notes": plan.rejected,
                         "raw_correct": raw_ok, "final_correct": ok, "latency": plan.latency})
            print(f"[{m[:14]}] raw={raw_ok!s:5} final={ok!s:5} {plan.latency:5.1f}s {instr}", flush=True)
        n = len(rows)
        results[m] = {"rows": rows,
                      "raw_acc": sum(r["raw_correct"] for r in rows) / n,
                      "final_acc": sum(r["final_correct"] for r in rows) / n,
                      "median_latency": sorted(r["latency"] for r in rows)[n // 2]}
        print(m, {k: v for k, v in results[m].items() if k != "rows"}, flush=True)
        del planner
    out = ROOT / "outputs" / "eval_planner.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
