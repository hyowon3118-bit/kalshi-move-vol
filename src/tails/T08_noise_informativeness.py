"""T8: 시장 잡음(원가격 단조성 위반)이 클 때 Kalshi 괴리(Gap)의 예측력이 약하다.

근거 ※ (T7과 같은 추론. RQ1의 S ~ Gap 관계가 잡음 많은 날 약해지는지)
모형: S = a + b·Gap + c·viol + d·Gap·viol + 지표 고정효과 (viol 지표 안 표준화)
주 검정: d < 0 (h=1).  보조: rough로 같은 모형
실행: python src/tails/T08_noise_informativeness.py
"""
from tail_utils import fit, get_logger, load, row, save

log = get_logger("tail_T08")
e = load(1).dropna(subset=["S", "Gap"])
rows = []
for m, main in [("viol_raw_z", True), ("rough_z", False)]:
    d = e.dropna(subset=[m])
    rows.append(row("T8", f"S ~ Gap*{m}", "-", "pooled", fit(f"S ~ Gap * {m} + C(series)", d, True), f"Gap:{m}", main, "less"))
save(rows, "T08_noise_informativeness", log)
