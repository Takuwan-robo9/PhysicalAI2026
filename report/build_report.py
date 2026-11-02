"""評価結果 (outputs/*.json) を report.html に埋め込み、Chrome で PDF 化する.

  python report/build_report.py --youtube URL --github URL --account アカウント名
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
CHROME_CANDIDATES = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "google-chrome", "chromium",
]


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--youtube", default="(提出時に記入)")
    ap.add_argument("--github", default="(提出時に記入)")
    ap.add_argument("--account", default="ACCOUNT")
    a = ap.parse_args()

    man = json.loads((ROOT / "outputs" / "eval_manipulation.json").read_text(encoding="utf-8"))["summary"]
    plan = json.loads((ROOT / "outputs" / "eval_planner.json").read_text(encoding="utf-8"))
    demo = json.loads((ROOT / "outputs" / "demo.json").read_text(encoding="utf-8"))

    n_steps = 3 * man["trials"]
    rows = []
    for m, r in plan.items():
        name = "Qwen2.5-3B-Instruct" if "3b" in m else "Qwen2.5-1.5B-Instruct"
        rows.append(f"<tr><td>{name}</td><td class='num'>{pct(r['raw_acc'])}</td>"
                    f"<td class='num'><b>{pct(r['final_acc'])}</b></td><td class='num'>{r['median_latency']:.1f} 秒</td></tr>")
    n_cases = len(next(iter(plan.values()))["rows"])
    r3 = next(v for k, v in plan.items() if "3b" in k)
    r15 = next(v for k, v in plan.items() if "1.5b" in k)
    findings = [
        f"<li>3B モデルでは、検証・修復層によって正解率が {pct(r3['raw_acc'])} → {pct(r3['final_acc'])} に上がった。"
        "LLM は「下のブロックを先に動かす」という物理的に不可能な順序を出しやすく、修復層がそれを直した。</li>",
        f"<li>1.5B モデルは {pct(r15['raw_acc'])} → {pct(r15['final_acc'])} にとどまった。比喩（トマト色・バナナ色）の解釈や「全部」の列挙で誤りが多く、"
        "修復層は物理的な整合性しか直せないため、<b>言葉の意味の理解はモデルの規模に強く依存する</b>ことがわかった。</li>",
        "<li>3B でも誤ったのは「テーブルに残っているものを全部…」（お皿の上の物体まで含めてしまう）で、状態を踏まえた集合の推論が小型 LLM の弱点である。</li>",
    ]
    ok = sum(sum(c["success"]) for c in demo["commands"])
    tot = sum(len(c["success"]) for c in demo["commands"])
    lat = ", ".join(f"{c['latency']:.0f}" for c in demo["commands"])
    demo_txt = (f"ピック&amp;プレース {ok}/{tot} ステップが成功し、不可能な指示（ダンス）には手順を出さずに断った。"
                f"LLM の推論時間は各指示で {lat} 秒（CPU）であった。")

    html = (REP / "report.html").read_text(encoding="utf-8")
    fill = {
        "{{YOUTUBE_URL}}": a.youtube, "{{GITHUB_URL}}": a.github,
        "{{MANIP_ALL}}": f"{pct(man['step_success_rate'])}（{round(man['step_success_rate'] * n_steps)}/{n_steps}）",
        "{{MANIP_PLATE}}": pct((man["by_type"]["plate_1"] + man["by_type"]["plate_2"]) / 2),
        "{{MANIP_STACK}}": pct(man["by_type"]["stack"]),
        "{{PERC_ERR}}": f"{man['perception_err_mm_mean']:.2f} mm / {man['perception_err_mm_max']:.2f} mm",
        "{{MANIP_FAIL_NOTE}}": "円柱を立方体の上に積んだ後の再認識で円柱が見つからず（転倒・転落したと考えられる）、"
                               "再挑戦もできなかった。底面の小さい円柱を積む動作は不安定であり、置く速度や高さの調整、"
                               "見失ったときの探索動作が今後の課題である。",
        "{{N_CASES}}": str(n_cases),
        "{{PLANNER_ROWS}}": "\n  ".join(rows),
        "{{PLANNER_FINDINGS}}": "\n  ".join(findings),
        "{{DEMO_RESULT}}": demo_txt,
    }
    for k, v in fill.items():
        html = html.replace(k, v)
    out_html = REP / "report_filled.html"
    out_html.write_text(html, encoding="utf-8")

    chrome = next((c for c in CHROME_CANDIDATES if Path(c).exists() or shutil.which(c)), None)
    pdf = REP / f"説明資料_PAI最終課題_{a.account}.pdf"
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out_html.as_uri()], check=True)
    print(pdf)


if __name__ == "__main__":
    main()
