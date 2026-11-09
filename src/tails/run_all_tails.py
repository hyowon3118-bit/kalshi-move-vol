"""꼬리·비매끄러움 가설 T1~T11 일괄 실행 + 다중검정 보정.

주 검정(사전 고정, ZT·ZN·MOVE·발표치)은 하나의 묶음으로 Benjamini–Hochberg 보정
추가 수집 종속변수(ES·SR3)와 보조 검정은 따로 보정 (탐색용)
실행 순서: python src/tails/00_tail_features.py → python src/hyp/00_event_windows.py → python src/tails/run_all_tails.py
출력: outputs/tables/tail_T*.csv (가설별), tail_summary_main.csv, tail_summary_secondary.csv
"""
import runpy
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from tail_utils import TABLES, get_logger

HERE = Path(__file__).resolve().parent
FILES = sorted(p.name for p in HERE.glob("T[0-9][0-9]_*.py"))
log = get_logger("tail_run_all")
for i, f in enumerate(FILES, 1):
    log.info("[%d/%d] %s", i, len(FILES), f)
    runpy.run_path(str(HERE / f), run_name="__main__")

allr = pd.concat([pd.read_csv(p) for p in sorted(TABLES.glob("hyp_T[0-9][0-9]_*.csv"))], ignore_index=True)
for name, part in [("main", allr[allr["main"]].copy()), ("secondary", allr[~allr["main"] & (allr["sample"] == "pooled")].copy())]:
    part["p_bh"] = multipletests(part["p"], method="fdr_bh")[1]
    part["sig_bh05"] = part["p_bh"] < 0.05
    part.to_csv(TABLES / f"tail_summary_{name}.csv", index=False)
    log.info("== %s 검정 %d개, BH 보정 후 유의(5%%) %d개", name, len(part), int(part["sig_bh05"].sum()))
    for r in part.itertuples():
        log.info("  %-4s %-36s %-5s coef=%8.3f p=%.4f BH p=%.4f %s %s", r.hyp, r.test, r.target, r.coef, r.p, r.p_bh,
                 "유의" if r.sig_bh05 else "  ", r.note if isinstance(r.note, str) else "")
