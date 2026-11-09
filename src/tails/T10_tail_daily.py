"""T10: 일별로 Kalshi 꼬리 확률이 늘어나는 날 채권 변동성(MOVE, 2년물 실현변동성)도 오른다.

근거 ※ (앞선 일별 분석에서 표준편차·엔트로피 변화는 무관. 꼬리 확률은 극단 위험만 따로 반영할 수 있다는 추론)
변수: 매일 만기 가장 가까운 이벤트 분포에서 tail1 = P(|X−F| > σ) (F·σ는 그 이벤트 값)
      Δtail1: 같은 이벤트 안 직전 MOVE 거래일 대비 변화 (이벤트 바뀌는 날 제외)
모형: ΔY = a + b·Δtail1(지표 안 표준화) + 지표 고정효과, 날짜 군집 표준오차, Y ∈ {MOVE, ZT 연율 실현변동성}
주 검정: b > 0, 두 Y.  보조: ZN, Kalshi 거래 있는 날만
실행: python src/tails/T10_tail_daily.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tail_utils import DATA, fit, get_logger, row, save

spec = importlib.util.spec_from_file_location("feat", Path(__file__).with_name("00_tail_features.py"))
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)

log = get_logger("tail_T10")
log.info("[1/3] 일별 꼬리 확률 계산")
panel = pd.read_parquet(DATA / "daily_panel.parquet")
ev = pd.read_parquet(DATA / "input" / "event_panel.parquet").drop_duplicates("event_ticker").set_index("event_ticker")[["F", "sigma"]]
dist = pd.read_parquet(DATA / "input" / "kalshi_dist.parquet")
dist["date"] = pd.to_datetime(dist["date"])
groups = {k: g for k, g in dist.groupby(["contract_preamble", "date"])}
vals = []
for r in panel.itertuples():
    g = groups.get((r.contract_preamble, r.date))
    if g is None or r.contract_preamble not in ev.index or len(g) < 3:
        vals.append(np.nan)
        continue
    F, sig = ev.loc[r.contract_preamble, ["F", "sigma"]]
    if not np.isfinite(F) or not np.isfinite(sig):
        vals.append(np.nan)
        continue
    f, _ = feat.features(g, F, sig, F, feat.UNIT[r.series])
    vals.append(f["tail1"])
panel["tail1"] = vals
log.info("  tail1 산출 %d / %d", int(panel["tail1"].notna().sum()), len(panel))

log.info("[2/3] 변화량")
rv = pd.read_csv(DATA / "treasury_rv_daily.csv", parse_dates=["date"]).sort_values("date")
for t in ["ZT", "ZN"]:
    rv[f"d_{t}"] = rv[f"vol_{t}"].diff()
panel = panel.merge(rv[["date", "d_ZT", "d_ZN"]], on="date", how="left").sort_values(["series", "date"])
same = panel["contract_preamble"].eq(panel.groupby("series")["contract_preamble"].shift())
panel["d_tail1"] = panel.groupby("series")["tail1"].diff().where(same)
panel["x"] = panel.groupby("series")["d_tail1"].transform(lambda s: (s - s.mean()) / s.std())
panel["cluster"] = panel["date"].dt.strftime("%Y-%m-%d")

log.info("[3/3] 회귀")
rows = []
for y, main in [("dMOVE", True), ("d_ZT", True), ("d_ZN", False)]:
    d = panel.dropna(subset=["x", y])
    rows.append(row("T10", f"{y} ~ Δtail1", y, "pooled", fit(f"{y} ~ x + C(series)", d, True), "x", main, "greater"))
    d2 = d[d["traded"]]
    rows.append(row("T10", f"{y} ~ Δtail1 (거래일만)", y, "pooled", fit(f"{y} ~ x + C(series)", d2, True), "x", False, "greater"))
save(rows, "T10_tail_daily", log)
