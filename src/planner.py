"""ローカル LLM (Qwen2.5-Instruct, GGUF 4bit 量子化 / llama.cpp) による
言語指示 → 作業計画 (ピック&プレース列) の変換.

- API キー不要・CPU のみで動作 (メモリ 2GB 程度)
- JSON スキーマで出力を制約 (grammar-constrained decoding):
  pick / place に書ける ID を「いま認識できている物体」だけに限定する
- それでも残る論理的に不正な手順 (自分の上に置く等) は validate() で除去する (安全層)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from perception import COLOR_JA, SHAPE_JA

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"

SYSTEM_PROMPT = """あなたはロボットアームの作業計画AIです。
テーブルの上の物体リストと、人間からの日本語の指示が与えられます。
指示を実現するための「つかむ物体(pick)」と「置く場所(place)」の手順を、実行する順番どおりにJSONで出力してください。

ルール:
- pick にはブロック(立方体・円柱)のIDだけを使う。お皿は動かせない。
- place にはお皿のID、または別のブロックのID(そのブロックの上に積む)を使う。
- 指示で言及された物体だけを動かす。指示と関係のない物体は絶対に動かさない。
- すでに目的の場所に置かれている物体は動かさない。
- 「テーブルに残っているもの」とは、場所が「テーブルの上」の物体だけを指す。お皿や他のブロックの上にある物体は含まない。
- 上に物が載っているブロックを動かすときは、先に上の物体を動かす。
- 色や形は、比喩や言い換え(例: 「空の色」=青、「バナナの色」=黄色、「丸いの」=円柱、「四角いの」=立方体)も解釈する。
- ロボットにできない指示(ピック&プレース以外)の場合は steps を空にする。
- まず interpretation に、指示の中の言葉がどの物体を指すかを「言葉=物体の色と形」の形で短く書く。
- reply には、人間への短い返事(30文字以内)を書く。
- 出力は {"interpretation": "...", "reply": "...", "steps": [{"pick": "ID", "place": "ID"}, ...]} の形式のJSONのみ。"""

_SCENE_EX = """物体リスト:
- white_cube: 白の立方体のブロック (四角い形, 場所: テーブルの上)
- black_cylinder: 黒の円柱のブロック (丸い形, 場所: テーブルの上)
- orange_cube: オレンジの立方体のブロック (四角い形, 場所: テーブルの上)
- gray_plate: グレーのお皿
- brown_plate: 茶色のお皿"""

FEWSHOT = [
    (f"{_SCENE_EX}\n\n指示: 白いブロックを茶色のお皿に置いて",
     '{"interpretation": "白いブロック=白の立方体, 茶色のお皿=茶色のお皿", "reply": "白いブロックを茶色のお皿に置きます。", "steps": [{"pick": "white_cube", "place": "brown_plate"}]}'),
    (f"{_SCENE_EX.replace('白の立方体のブロック (四角い形, 場所: テーブルの上)', '白の立方体のブロック (四角い形, 場所: brown_plateの上)')}"
     "\n\n指示: テーブルに残っているものを全部グレーのお皿に片付けて",
     '{"interpretation": "テーブルに残っているもの=黒の円柱とオレンジの立方体(白はすでにお皿の上), グレーのお皿=グレーのお皿", "reply": "残りの2つをグレーのお皿に片付けます。", "steps": [{"pick": "black_cylinder", "place": "gray_plate"}, '
     '{"pick": "orange_cube", "place": "gray_plate"}]}'),
    (f"{_SCENE_EX}\n\n指示: みかん色のやつの上に白いのを重ねて",
     '{"interpretation": "みかん色のやつ=オレンジの立方体, 白いの=白の立方体", "reply": "オレンジのブロックの上に白いブロックを積みます。", "steps": [{"pick": "white_cube", "place": "orange_cube"}]}'),
    (f"{_SCENE_EX}\n\n指示: コーヒーを入れて",
     '{"interpretation": "コーヒーを入れる=ピック&プレースではない", "reply": "ごめんなさい、物を運ぶことしかできません。", "steps": []}'),
]


@dataclass
class Plan:
    interpretation: str
    reply: str
    steps: list[dict]
    raw: str
    rejected: list[str]
    latency: float


def describe_scene(objects: dict[str, dict]) -> str:
    """objects: id -> {"color", "shape", "on", "hidden"} から LLM に渡す物体リスト文字列を作る."""
    lines = ["物体リスト:"]
    for oid, o in objects.items():
        name = f"{COLOR_JA[o['color']]}の{SHAPE_JA[o['shape']]}"
        if o["shape"] == "plate":
            lines.append(f"- {oid}: {name}")
        else:
            where = "場所: テーブルの上" if o.get("on") in (None, "table") else f"場所: {o['on']}の上"
            if o.get("hidden"):
                where += f"、上に{o['hidden']}が載っている"
            form = "丸い形" if o["shape"] == "cylinder" else "四角い形"
            lines.append(f"- {oid}: {name} ({form}, {where})")
    return "\n".join(lines)


def plan_schema(objects: dict[str, dict]) -> dict:
    blocks = [k for k, o in objects.items() if o["shape"] != "plate"]
    return {
        "type": "object",
        "properties": {
            "interpretation": {"type": "string"},
            "reply": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"pick": {"type": "string", "enum": blocks},
                                   "place": {"type": "string", "enum": list(objects)}},
                    "required": ["pick", "place"],
                },
            },
        },
        "required": ["interpretation", "reply", "steps"],
    }


class LLMPlanner:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, n_threads: int | None = None):
        from llama_cpp import Llama

        t = time.time()
        self.llm = Llama(model_path=str(model_path), n_ctx=2048, n_batch=128, n_threads=n_threads, verbose=False, seed=0)
        self.model_name = Path(model_path).name
        print(f"[planner] loaded {self.model_name} in {time.time() - t:.1f}s")

    def plan(self, instruction: str, objects: dict[str, dict]) -> Plan:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in FEWSHOT:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": f"{describe_scene(objects)}\n\n指示: {instruction}"})
        t = time.time()
        out = self.llm.create_chat_completion(
            messages=msgs, temperature=0.0, max_tokens=320,
            response_format={"type": "json_object", "schema": plan_schema(objects)},
        )
        raw = out["choices"][0]["message"]["content"].strip()
        reply, steps, rejected = validate(raw, objects)
        try:
            interp = str(json.loads(raw).get("interpretation", ""))
        except json.JSONDecodeError:
            interp = ""
        return Plan(interpretation=interp, reply=reply, steps=steps, raw=raw, rejected=rejected,
                    latency=time.time() - t)


def validate(raw: str, objects: dict[str, dict]) -> tuple[str, list[dict], list[str]]:
    """LLM 出力をパースし、物理的に実行可能な手順だけを残す."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return "", [], ["JSON が見つからない"]
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return "", [], [f"JSON パース失敗: {e}"]
    rejected: list[str] = []   # 除去・修正した内容 (動画とログに表示)
    ok: list[dict] = []
    stacked_on: dict[str, str] = {k: o.get("on", "table") for k, o in objects.items() if o["shape"] != "plate"}
    queue = [{"pick": s.get("pick"), "place": s.get("place")} for s in js.get("steps", [])]
    for _ in range(32):        # 修復で手順が増えても有限回で止める
        if not queue:
            break
        s = queue.pop(0)
        p, q = s["pick"], s["place"]
        if p not in objects or (q not in objects and q != "table"):
            rejected.append(f"未知のID: {p}->{q}")
            continue
        if objects[p]["shape"] == "plate":
            rejected.append(f"お皿は持ち上げられない: {p}")
            continue
        if p == q:
            rejected.append(f"自分の上には置けない: {p}")
            continue
        tops = [k for k, v in stacked_on.items() if v == p]
        if tops:
            # 上に物が載っている → 上の物体を先に動かす (計画内にあれば並べ替え, なければ退避を追加)
            t = tops[0]
            later = next((i for i, x in enumerate(queue) if x["pick"] == t), None)
            if later is not None:
                queue = [queue.pop(later), s] + queue
                rejected.append(f"[修正] 上の{t}を先に動かすよう順序を入れ替え")
            else:
                queue = [{"pick": t, "place": "table", "auto": True}, s] + queue
                rejected.append(f"[修正] 上の{t}を一旦テーブルに退避する手順を追加")
            continue
        if q != "table" and objects[q]["shape"] != "plate" and any(v == q for v in stacked_on.values()):
            rejected.append(f"置き先の上が塞がっている: {q}")
            continue
        if stacked_on.get(p) == q:
            rejected.append(f"すでに置かれている: {p}->{q}")
            continue
        ok.append(s)
        stacked_on[p] = q
    return str(js.get("reply", "")), ok, rejected


def object_name_ja(o: dict) -> str:
    return f"{COLOR_JA[o['color']]}の{SHAPE_JA[o['shape']]}"


def plan_to_ja(steps: list[dict], objects: dict[str, dict]) -> list[str]:
    lines = []
    for k, s in enumerate(steps):
        if s["place"] == "table":
            dst_ja = "テーブルの空き場所へ退避"
        else:
            dst = objects[s["place"]]
            dst_ja = object_name_ja(dst) + ("へ" if dst["shape"] == "plate" else "の上に積む")
        mark = " [自動追加]" if s.get("auto") else ""
        lines.append(f"{k + 1}. {object_name_ja(objects[s['pick']])} → {dst_ja}{mark}")
    return lines
