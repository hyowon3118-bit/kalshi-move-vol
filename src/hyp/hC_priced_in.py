"""가설 C: Kalshi가 예측한 서프라이즈는 이미 가격에 반영돼서 발표 순간엔 예측 못 한 부분만 가격을 움직인다.

근거: Campbell & Sharpe (2009)의 예측 가능 성분/예측 불가 성분 분해를 분 단위 국채선물에 적용 ※(분 단위 확장)
모형: r_win = a + c1·Ŝ_h + c2·u_h (+ 지표 고정효과)
  Ŝ = RQ1 표본외 예측(발표 1일 전 Kalshi 괴리로 만든 값, 그 발표 이전 자료로만 추정), u = S − Ŝ, _h는 매파 방향 부호
주 검정: c2 < 0 (단측, 예측 못 한 매파 서프라이즈 → 선물 하락), c1 = 0 (양측)
  c1 동등성(TOST, ±1bp/1σ ※)도 함께 기록. 통합·ZT·ZN.  보조: 지표별
실행: python src/hyp/hC_priced_in.py
"""
import numpy as np
from scipy import stats

from hyp_utils import SERIES, TARGETS, fit, get_logger, load_events, row, save

MARGIN = 1.0
log = get_logger("hyp_C")
e = load_events(1).dropna(subset=["S_hat_h", "u_h"])
rows = []
for t in TARGETS:
    d = e.dropna(subset=[f"{t}_win"]).rename(columns={f"{t}_win": "y"})
    for sample, g, pooled in [("pooled", d, True)] + [(s, d[d["series"] == s], False) for s in SERIES]:
        if len(g) < 10:
            continue
        f = "y ~ S_hat_h + u_h" + (" + C(series)" if pooled else "")
        res = fit(f, g, pooled)
        b, se = res.params["S_hat_h"], res.bse["S_hat_h"]
        p_tost = max(stats.norm.sf((b + MARGIN) / se), stats.norm.cdf((b - MARGIN) / se))
        rows.append(row("C", "c2 r_win ~ u_h", t, sample, res, "u_h", pooled, "less"))
        rows.append(row("C", "c1 r_win ~ S_hat_h", t, sample, res, "S_hat_h", pooled, "two-sided",
                        f"TOST(±{MARGIN}bp) p={p_tost:.3f}"))
save(rows, "C_priced_in", log)
