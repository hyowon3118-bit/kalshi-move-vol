"""가설 D: Kalshi 분포가 넓을수록(사전 불확실성↑) 같은 크기의 서프라이즈에 선물이 더 크게 반응한다.

근거: Pasquariello & Vega (2007) — 의견 분산이 크고 공개 발표가 부정확할 때 주문흐름과 금리 변화의 상관이 더 큼 (간접)
      불확실성이 반응 크기를 키운다는 방향 설정은 ※
모형: |r_win| = a + b1·|S| + b2·W + b3·|S|·W (+ 지표 고정효과),  W = W_iqr(발표 1일 전 연속 IQR/σ)를 지표 안에서 표준화
주 검정: b3 > 0 (단측), 통합, ZT·ZN
보조: 방향 모형 r_win = a + b·S_h + d·S_h·W 의 d < 0 (넓을수록 매파 서프라이즈에 더 하락), 지표별 b3
실행: python src/hyp/hD_uncertainty_interaction.py
"""
from hyp_utils import SERIES, TARGETS, fit, get_logger, load_events, row, save, zscore_within

log = get_logger("hyp_D")
e = load_events(1).dropna(subset=["S", "W_iqr"])
e["absS"] = e["S"].abs()
e["Wz"] = zscore_within(e, "W_iqr")
rows = []
for t in TARGETS:
    d = e.dropna(subset=[f"{t}_win"]).copy()
    d["y_abs"], d["y"] = d[f"{t}_win"].abs(), d[f"{t}_win"]
    rows.append(row("D", "|r_win| ~ |S|*W", t, "pooled", fit("y_abs ~ absS * Wz + C(series)", d, True), "absS:Wz", True, "greater"))
    rows.append(row("D", "r_win ~ S_h*W", t, "pooled", fit("y ~ S_h * Wz + C(series)", d, True), "S_h:Wz", False, "less"))
    for s in SERIES:
        g = d[d["series"] == s]
        rows.append(row("D", "|r_win| ~ |S|*W", t, s, fit("y_abs ~ absS * Wz", g, False), "absS:Wz", False, "greater"))
save(rows, "D_uncertainty_interaction", log)
