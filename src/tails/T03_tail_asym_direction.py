"""T3: 위쪽 꼬리가 아래쪽보다 두꺼우면(꼬리 비대칭) 실제 발표치도 위로 나온다 (평균 괴리 Gap을 통제해도).

근거 ※ (앞선 F: 왜도는 실패. 왜도는 분포 전체 모양이라 꼬리 확률 차이로 직접 측정)
모형: S = a + b·asym1 + c·Gap + 지표 고정효과,  asym1 = P(X>F+σ) − P(X<F−σ), 지표 안 표준화
주 검정: b > 0, h=1·h=7
실행: python src/tails/T03_tail_asym_direction.py
"""
from tail_utils import SERIES, fit, get_logger, load, row, save

log = get_logger("tail_T03")
rows = []
for h in [1, 7]:
    e = load(h).dropna(subset=["S", "asym1_z", "Gap"])
    rows.append(row("T3", f"S ~ asym1 + Gap (h={h})", "-", "pooled", fit("S ~ asym1_z + Gap + C(series)", e, True), "asym1_z", True, "greater"))
    rows.append(row("T3", f"S ~ asym1 only (h={h})", "-", "pooled", fit("S ~ asym1_z + C(series)", e, True), "asym1_z", False, "greater"))
    for s in SERIES:
        g = e[e["series"] == s]
        rows.append(row("T3", f"S ~ asym1 + Gap (h={h})", "-", s, fit("S ~ asym1_z + Gap", g, False), "asym1_z", False, "greater"))
save(rows, "T03_tail_asym_direction", log)
