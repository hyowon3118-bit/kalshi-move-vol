"""T11: 발표 전 꼬리 확률이 클수록 발표 후 채권 변동성이 더 크게 떨어진다 (Beber & Brandt의 꼬리 버전).

근거: Beber & Brandt (2009) — 경제 파생상품으로 잰 사전 불확실성↑ → 발표 후 옵션 내재변동성 하락폭↑. 꼬리확률로 대체는 ※
모형: ΔlogRV_ZT = log RV(다음 거래일) − log RV(전 거래일) = a + b·tail1 + 지표 고정효과
      ΔMOVE(발표일) = MOVE(발표일) − MOVE(전 거래일) = a + b·tail1 + 지표 고정효과
주 검정: b < 0, 두 모형.  보조: ZN, W_iqr 통제
실행: python src/tails/T11_tail_resolution.py
"""
import numpy as np
import pandas as pd

from tail_utils import DATA, fit, get_logger, load, row, save

log = get_logger("tail_T11")
e = load(1).dropna(subset=["tail1_z"])
rv = pd.read_csv(DATA / "treasury_rv_daily.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
mv = pd.read_csv(DATA / "input" / "move_daily.csv", parse_dates=["date"]).dropna().sort_values("date").reset_index(drop=True)


def around(series: pd.DataFrame, col: str, day, how: str):
    dates = series["date"].to_numpy()
    i = np.searchsorted(dates, np.datetime64(day))
    if i == 0 or i >= len(dates) or dates[i] != np.datetime64(day):
        return np.nan
    if how == "dlog":
        if i + 1 >= len(dates):
            return np.nan
        a, b = series[col].iloc[i - 1], series[col].iloc[i + 1]
        return np.log(b) - np.log(a) if a > 0 and b > 0 else np.nan
    return series[col].iloc[i] - series[col].iloc[i - 1]


rows = []
for t, main in [("ZT", True), ("ZN", False)]:
    d = e.assign(y=[around(rv, f"rv_{t}", x, "dlog") for x in e["release_date_actual"]]).dropna(subset=["y"])
    rows.append(row("T11", "dlogRV ~ tail1", t, "pooled", fit("y ~ tail1_z + C(series)", d, True), "tail1_z", main, "less"))
    rows.append(row("T11", "dlogRV ~ tail1 + W", t, "pooled", fit("y ~ tail1_z + W_iqr_z + C(series)", d.dropna(subset=["W_iqr_z"]), True), "tail1_z", False, "less"))
d = e.assign(y=[around(mv, "value", x, "diff") for x in e["release_date_actual"]]).dropna(subset=["y"])
rows.append(row("T11", "dMOVE ~ tail1", "MOVE", "pooled", fit("y ~ tail1_z + C(series)", d, True), "tail1_z", True, "less"))
rows.append(row("T11", "dMOVE ~ tail1 + W", "MOVE", "pooled", fit("y ~ tail1_z + W_iqr_z + C(series)", d.dropna(subset=["W_iqr_z"]), True), "tail1_z", False, "less"))
save(rows, "T11_tail_resolution", log)
