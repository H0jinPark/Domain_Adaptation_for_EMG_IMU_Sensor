"""results/ 아래 모든 실험 결과를 한 표(all_results_inventory.csv)로 모은다.

두 종류의 산출물을 읽는다.
  1. multi-seed 결과 json  (results/mean/std 키를 가진 파일)
  2. *_summary.txt         (json 이 없는 옛 런 — EMG / CDAN_Entropy / 일부 Learnable_R)

같은 런이 json 과 summary.txt 로 둘 다 있으면 json 쪽만 쓴다
(seed 평균이 같은지로 판정한다).

test 지표가 없는 파일은 옛 2분할 규약이라 protocol 컬럼에서 구분된다
(`protocol-train-val-test-split`).

    conda run -n DA python scripts/build_results_inventory.py
"""
from pathlib import Path
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "tables" / "all_results_inventory.csv"

# json metric key -> 표 컬럼명
METRICS = {
    "source_acc":      "Src-Val",
    "target_acc":      "Tgt-Val",
    "shift":           "Shift-Val",
    "source_test_acc": "Src-Test",
    "target_test_acc": "Tgt-Test",
    "test_shift":      "Shift-Test",
}


def mtime_str(p):
    return (pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
              .tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"))


def infer_modality(name):
    n = name.lower()
    if "_mm_" in n or n.startswith("mm_") or "multimodal" in n:
        return "multimodal"
    if "emg" in n:
        return "emg_only"
    return "imu_only"


def load_json_runs():
    """multi-seed 결과 json 을 '한 행 = 한 seed' 로 편다."""
    rows = []
    for p in sorted(RESULTS.rglob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        runs = j.get("results")
        if not isinstance(runs, list) or not runs or "mean" not in j:
            continue                     # multi-seed 결과 파일이 아님
        for r in runs:
            if not isinstance(r, dict):
                continue
            row = {
                "group":    p.parent.name,
                "file":     p.relative_to(ROOT).as_posix(),
                "tag":      j.get("tag"),
                "modality": j.get("modality") or infer_modality(p.name),
                "mtime":    mtime_str(p),
                "seed":     r.get("seed"),
            }
            row.update({k: r.get(k) for k in METRICS})
            rows.append(row)
    return rows


SEED_ROW = re.compile(r"^\s*(\d+)\s*\|(.+)$")
PCT = re.compile(r"(-?[\d.]+)\s*%")


def parse_summary(p):
    """*_summary.txt 의 seed 별 표를 json 과 같은 스키마의 행 리스트로 만든다.

    구 포맷 : Seed | Source | Target | Shift            (val 만)
    신 포맷 : Seed | Src-Val | Tgt-Val | Shift | Src-Test | Tgt-Test
    """
    text = p.read_text(encoding="utf-8", errors="replace")

    title = ""
    for line in text.splitlines():
        if "Multi-seed Summary" in line:
            title = line.split("|", 1)[1].strip() if "|" in line else ""
            break
    tag = title.split("|")[-1].strip() if title else ""
    if not tag or "(" in tag:            # 제목에 설정명이 없으면 파일명에서
        tag = p.name.replace("_summary.txt", "")

    rows = []
    for line in text.splitlines():
        m = SEED_ROW.match(line)
        if not m:
            continue
        vals = [float(v) for v in PCT.findall(m.group(2))]
        row = {
            "group":    p.parent.name,
            "file":     p.relative_to(ROOT).as_posix(),
            "tag":      tag,
            "modality": infer_modality(p.name if p.parent.name != "EMG" else "emg"),
            "mtime":    mtime_str(p),
            "seed":     int(m.group(1)),
        }
        row.update({k: None for k in METRICS})
        if len(vals) >= 3:
            row["source_acc"], row["target_acc"], row["shift"] = vals[:3]
        if len(vals) >= 5:
            row["source_test_acc"], row["target_test_acc"] = vals[3], vals[4]
            row["test_shift"] = vals[3] - vals[4]
        if len(vals) >= 3:
            rows.append(row)
    return rows


def run_key(rows):
    """중복 판정용 키 — 같은 그룹에서 seed 별 (Src-Val, Tgt-Val) 이 다 같으면 같은 런.

    summary.txt 는 소수 2자리로 찍히므로 평균이 아니라 seed 값을 2자리로 맞춰 비교한다
    (평균끼리 비교하면 반올림 오차로 어긋난다).
    """
    pairs = sorted((round(r["source_acc"], 2), round(r["target_acc"], 2))
                   for r in rows
                   if r["source_acc"] is not None and r["target_acc"] is not None)
    if not pairs:
        return None
    return (rows[0]["group"], tuple(pairs))


def load_summary_runs(json_rows):
    """json 에 없는 런만 summary.txt 에서 주워 담는다."""
    seen = set()
    by_file = {}
    for r in json_rows:
        by_file.setdefault(r["file"], []).append(r)
    for rows in by_file.values():
        k = run_key(rows)
        if k:
            seen.add(k)

    out, skipped = [], []
    for p in sorted(RESULTS.rglob("*_summary.txt")):
        rows = parse_summary(p)
        if not rows:
            continue
        k = run_key(rows)
        if k in seen:
            skipped.append(p.name)       # 같은 런의 json 이 이미 있다
            continue
        seen.add(k)
        out.extend(rows)
    return out, skipped


def agg_pm(df, keys, nd=2):
    """keys 로 묶어 각 지표를 'mean ± std' 문자열 한 칸으로."""
    g = df.groupby(keys, dropna=False, observed=True)
    out = pd.DataFrame({"n": g.size()})
    for m, col in METRICS.items():
        mu, sd = g[m].mean(), g[m].std()
        out[col] = [
            "—" if pd.isna(a) else (f"{a:.{nd}f} ± {b:.{nd}f}" if pd.notna(b) else f"{a:.{nd}f}")
            for a, b in zip(mu, sd)
        ]
    return out.reset_index()


def main():
    json_rows = load_json_runs()
    sum_rows, skipped = load_summary_runs(json_rows)
    df = pd.DataFrame(json_rows + sum_rows)
    df["protocol"] = ["train/val/test" if pd.notna(v) else "train/val (구)"
                      for v in df["target_test_acc"]]

    inventory = agg_pm(df, ["group", "protocol", "file", "tag", "modality"])
    inventory = inventory.merge(
        df.groupby("file", as_index=False)["mtime"].max(), on="file", how="left")
    inventory = (inventory.sort_values(["group", "mtime"], ascending=[True, False])
                          .reset_index(drop=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT, index=False, encoding="utf-8-sig")

    n_json = len({r["file"] for r in json_rows})
    n_sum = len({r["file"] for r in sum_rows})
    print(f"json {n_json} 개 + summary.txt {n_sum} 개 = {len(inventory)} 런"
          f"  (json 과 중복이라 건너뛴 summary {len(skipped)} 개)")
    print(inventory.groupby(["group", "protocol"]).size().to_string())
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
