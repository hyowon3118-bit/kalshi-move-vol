"""T1: Kalshi는 꼬리 결과의 확률을 과대평가한다 (favorite–longshot bias).

근거: Bürgi, Deng & Whelan (2025, CESifo WP 12122) — Kalshi 30만여 계약에서 싼(낮은 확률) 계약은 가격이 뜻하는 것보다 훨씬 덜 맞음
T1a  컨센서스±σ 밖에 실제로 떨어진 비율 < Kalshi가 매긴 확률:  (hit1 − tail1) 평균 < 0, h=1·h=7
T1b  구간 단위: Kalshi 확률 10% 미만 구간이 실제로 맞은 비율 < 그 구간들의 평균 확률:  (hit − p) 평균 < 0, h=1
     보정 곡선(확률 구간별 예측 vs 실현)도 저장
추론: 발표일 군집 표준오차 (같은 날 여러 지표, 한 이벤트의 여러 구간)
실행: python src/tails/T01_tail_calibration.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tail_utils import DATA, TABLES, get_logger, load, row, save

log = get_logger("tail_T01")
rows = []
for h in [1, 7]:
    e = load(h).dropna(subset=["tail1", "hit_up1"])
    e["y"] = e["hit_up1"] + e["hit_dn1"] - e["tail1"]
    res = smf.ols("y ~ 1", e).fit(cov_type="cluster", cov_kwds={"groups": e["cluster"]})
    rows.append(row("T1", f"T1a hit(F±σ) − tail1 (h={h})", "-", "pooled", res, "Intercept", True, "less",
                    f"실현 {100*(e['hit_up1']+e['hit_dn1']).mean():.1f}% vs Kalshi {100*e['tail1'].mean():.1f}%"))

b = pd.read_parquet(DATA / "tail_bins.parquet")
b = b[b["h"] == 1].copy()
b["cluster"] = pd.to_datetime(b["release_date"]).dt.strftime("%Y-%m-%d")
low = b[(b["p"] > 0) & (b["p"] < 0.10)].copy()
low["y"] = low["hit"] - low["p"]
res = smf.ols("y ~ 1", low).fit(cov_type="cluster", cov_kwds={"groups": low["cluster"]})
rows.append(row("T1", "T1b 확률<10% 구간 hit − p (h=1)", "-", "pooled", res, "Intercept", True, "less",
                f"구간 {len(low)}개: 실현 {100*low['hit'].mean():.2f}% vs Kalshi {100*low['p'].mean():.2f}%"))
cuts = [0, .02, .05, .10, .20, .35, .50, 1.0]
b["p_bin"] = pd.cut(b["p"], cuts, include_lowest=True)
curve = b[b["p"] > 0].groupby("p_bin", observed=True).agg(n=("p", "size"), mean_p=("p", "mean"), hit_rate=("hit", "mean")).reset_index()
curve.to_csv(TABLES / "tail_T01_calibration_curve.csv", index=False)
for r in curve.itertuples():
    log.info("  보정곡선 %-14s n=%4d  Kalshi %.3f  실현 %.3f", r.p_bin, r.n, r.mean_p, r.hit_rate)
save(rows, "T01_tail_calibration", log)
