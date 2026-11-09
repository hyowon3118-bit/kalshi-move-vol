"""가설 A~F를 한 번에 실행하고, 주 검정 전체를 Benjamini–Hochberg로 함께 보정.

실행 순서: python src/hyp/00_event_windows.py → python src/hyp/run_all.py
출력: outputs/tables/hyp_{A..F}_*.csv (가설별), outputs/tables/hyp_summary.csv (주 검정 + BH 보정 p)
"""
import runpy
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from hyp_utils import TABLES, get_logger

HERE = Path(__file__).resolve().parent
FILES = ["hA_surprise_size.py", "hB_pre_drift.py", "hC_priced_in.py",
         "hD_uncertainty_interaction.py", "hE_vol_resolution.py", "hF_skew_direction.py"]

log = get_logger("hyp_run_all")
for i, f in enumerate(FILES, 1):
    log.info("[%d/%d] %s", i, len(FILES), f)
    runpy.run_path(str(HERE / f), run_name="__main__")

allr = pd.concat([pd.read_csv(p) for p in sorted(TABLES.glob("hyp_[A-F]_*.csv"))], ignore_index=True)
main = allr[allr["main"]].copy()
main["p_bh"] = multipletests(main["p"], method="fdr_bh")[1]
main["sig_bh05"] = main["p_bh"] < 0.05
main.to_csv(TABLES / "hyp_summary.csv", index=False)
log.info("주 검정 %d개, BH 보정 후 유의(5%%) %d개", len(main), int(main["sig_bh05"].sum()))
for r in main.itertuples():
    log.info("  %s %-28s %s coef=%8.3f p=%.4f BH p=%.4f %s", r.hyp, r.test, r.target, r.coef, r.p, r.p_bh,
             "유의" if r.sig_bh05 else "")
