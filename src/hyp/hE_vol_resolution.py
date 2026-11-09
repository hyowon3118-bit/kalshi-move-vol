"""가설 E: 발표 전 Kalshi 분포가 넓을수록 발표 후 국채선물 변동성이 발표 전보다 더 많이 떨어진다.

근거: Beber & Brandt (2009) — 경제 파생상품으로 잰 사전 불확실성이 클수록 발표 후 주식·채권 옵션 내재변동성이 더 크게 하락
      원 논문은 내재변동성, 여기선 5분 실현변동성으로 대체 ※
모형: ΔlogRV = log RV(발표 다음 거래일) − log RV(발표 전 거래일) = a + b·W (+ 지표 고정효과)
      W = 발표 1일 전 W_iqr(지표 안 표준화). 발표 당일은 발표 순간 점프가 섞여서 제외 ※
주 검정: b < 0 (단측), 통합, ZT·ZN.  보조: 지표별, 표준편차 기반 폭 W_sd
실행: python src/hyp/hE_vol_resolution.py
"""
import numpy as np
import pandas as pd

from hyp_utils import DATA, SERIES, TARGETS, fit, get_logger, load_events, row, save, zscore_within

log = get_logger("hyp_E")
rv = pd.read_csv(DATA / "treasury_rv_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
e = load_events(1).dropna(subset=["W_iqr"])
e["Wz"] = zscore_within(e, "W_iqr")
e["Wsdz"] = zscore_within(e, "W_sd")
dates = rv["date"].to_numpy()
rows = []
for t in TARGETS:
    s = rv.set_index("date")[f"rv_{t}"]
    def dlog(day):
        i = np.searchsorted(dates, np.datetime64(day))
        if i == 0 or i + 1 >= len(dates) or dates[i] != np.datetime64(day):
            return np.nan
        a, b = s.iloc[i - 1], s.iloc[i + 1]
        return np.log(b) - np.log(a) if a > 0 and b > 0 else np.nan
    d = e.copy()
    d["y"] = [dlog(x) for x in d["release_date_actual"]]
    d = d.dropna(subset=["y"])
    log.info("  %s: ΔlogRV 산출 %d건, 평균 %.3f", t, len(d), d["y"].mean())
    rows.append(row("E", "dlogRV ~ W_iqr", t, "pooled", fit("y ~ Wz + C(series)", d, True), "Wz", True, "less"))
    rows.append(row("E", "dlogRV ~ W_sd", t, "pooled", fit("y ~ Wsdz + C(series)", d.dropna(subset=["Wsdz"]), True), "Wsdz", False, "less"))
    for sname in SERIES:
        g = d[d["series"] == sname]
        rows.append(row("E", "dlogRV ~ W_iqr", t, sname, fit("y ~ Wz", g, False), "Wz", False, "less"))
save(rows, "E_vol_resolution", log)
