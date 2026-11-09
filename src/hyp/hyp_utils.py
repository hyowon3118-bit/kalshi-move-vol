"""가설 검증 공용 함수.

이벤트 표 (load_events)
  fixers26-1research의 이벤트 패널(발표 h일 전 Kalshi 분포, S, Gap, W_iqr, skewness 등) +
  00_event_windows.py가 만든 발표 시각 전후 국채선물 수익률 + RQ1 표본외 예측(Ŝ)

방향 맞추기 (pooled 분석용)
  "매파적" 방향 = 금리 상승 압력. CPI·근원 CPI·GDP·고용은 발표치가 높을수록, 실업률은 낮을수록 매파적 ※
  (경제 직관에 따른 고정 부호. 일별 RQ2에서 실업률만 금리 반응 부호가 반대로 나온 것과 일치)
  S_h = sign_k · S 처럼 _h가 붙은 변수는 이 부호를 곱한 것. 선물 가격은 금리와 반대라 기대 부호는 음(−)

추론: 지표별은 HC3, 통합은 발표일 군집 표준오차 (CPI·근원 CPI가 같은 날 발표)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES, get_logger, load_config  # noqa: E402,F401

SIGN = {"KXCPI": 1, "KXCPICORE": 1, "KXGDP": 1, "KXPAYROLLS": 1, "KXU3": -1}
SERIES = list(SIGN)
TARGETS = ["ZT", "ZN"]


def load_events(h: int = 1) -> pd.DataFrame:
    p = pd.read_parquet(DATA / "input" / "event_panel.parquet")
    p = p[p["h"] == h].copy()
    p["release_date_actual"] = pd.to_datetime(p["release_date_actual"])
    w = pd.read_csv(DATA / "event_windows.csv", parse_dates=["date"])
    p = p.merge(w, left_on="release_date_actual", right_on="date", how="left")
    o = pd.read_parquet(DATA / "input" / "rq1_oos_predictions.parquet")
    o = o[o["h"] == h].copy()
    o["release_date_actual"] = pd.to_datetime(o["release_date_actual"])
    # CPI 계열은 CPI+근원 통합 모형 예측, 나머지는 지표 단독 모형 예측 (RQ2와 같은 규칙)
    o = o[((o["sample"] == "primary") & o["series"].isin(["KXCPI", "KXCPICORE"])) |
          ((o["sample"] == o["series"]) & ~o["series"].isin(["KXCPI", "KXCPICORE"]))]
    p = p.merge(o[["event_ticker", "S_hat"]], on="event_ticker", how="left")
    p["u"] = p["S"] - p["S_hat"]
    p["sign"] = p["series"].map(SIGN)
    for v in ["S", "Gap", "S_hat", "u"]:
        p[f"{v}_h"] = p[v] * p["sign"]
    p["cluster"] = p["release_date_actual"].dt.strftime("%Y-%m-%d")
    return p


def zscore_within(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("series")[col].transform(lambda x: (x - x.mean()) / x.std())


def fit(formula: str, d: pd.DataFrame, pooled: bool):
    if pooled:
        return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cluster"]})
    return smf.ols(formula, data=d).fit(cov_type="HC3")


def row(hyp: str, test: str, target: str, sample: str, res, term: str, main: bool,
        alternative: str = "two-sided", note: str = "") -> dict:
    """alternative: 'greater' | 'less' | 'two-sided' (단측은 가설 방향일 때만 p를 절반)."""
    b, se, p2 = res.params[term], res.bse[term], res.pvalues[term]
    if alternative == "greater":
        p = p2 / 2 if b > 0 else 1 - p2 / 2
    elif alternative == "less":
        p = p2 / 2 if b < 0 else 1 - p2 / 2
    else:
        p = p2
    return {"hyp": hyp, "test": test, "target": target, "sample": sample, "term": term, "n": int(res.nobs),
            "coef": b, "se": se, "p": p, "alternative": alternative, "r2": res.rsquared, "main": main, "note": note}


def save(rows: list[dict], name: str, log) -> pd.DataFrame:
    TABLES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / f"hyp_{name}.csv", index=False)
    for r in out.itertuples():
        log.info("  %s %-28s %s %-10s %-10s n=%3d coef=%8.3f se=%7.3f p=%.4f%s", r.hyp, r.test, r.target, r.sample,
                 r.term, r.n, r.coef, r.se, r.p, "  [주]" if r.main else "")
    log.info("저장 outputs/tables/hyp_%s.csv", name)
    return out
