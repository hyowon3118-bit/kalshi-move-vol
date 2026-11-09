"""T7: 분포가 들쭉날쭉하거나 원가격 단조성 위반이 많을수록(시장 잡음↑) Kalshi 평균의 오차가 크다.

근거 ※ (단조성 위반은 무차익 조건 위반 = 유동성 부족·오래된 호가의 신호라는 추론)
모형: E_K = a + b·noise + 지표 고정효과,  E_K = |발표치 − Kalshi 평균|/σ
      noise ∈ {viol_raw(원가격 단조성 위반 비율), rough(확률 2차 차분 크기)}, 지표 안 표준화
주 검정: b > 0, 두 측정 각각 (h=1).  보조: kl_normal, 거래량 통제
실행: python src/tails/T07_noise_error.py
"""
import numpy as np

from tail_utils import fit, get_logger, load, row, save

log = get_logger("tail_T07")
e = load(1).dropna(subset=["E_K"])
e["lvol"] = np.log1p(e["volume"])
rows = []
for m, main in [("viol_raw_z", True), ("rough_z", True), ("kl_normal_z", False)]:
    d = e.dropna(subset=[m])
    rows.append(row("T7", f"E_K ~ {m}", "-", "pooled", fit(f"E_K ~ {m} + C(series)", d, True), m, main, "greater"))
    rows.append(row("T7", f"E_K ~ {m} + log거래량", "-", "pooled", fit(f"E_K ~ {m} + lvol + C(series)", d.dropna(subset=["lvol"]), True), m, False, "greater"))
save(rows, "T07_noise_error", log)
