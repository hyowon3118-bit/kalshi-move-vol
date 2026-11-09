"""T2: 꼬리가 몸통보다 두꺼운 분포(폭을 고려해도)일수록 실제 서프라이즈가 크다.

근거 ※ (Gürkaynak & Wolfers 2005: 경제 파생상품 분포가 잘 보정됨 → 꼬리 모양에도 정보가 있을 것이라는 추론)
모형: |S| = a + b·tail_ratio + 지표 고정효과,  tail_ratio = (q90−q10)/(q75−q25), 지표 안 표준화
주 검정: b > 0 (h=1).  보조: |S| ~ tail1(±σ 밖 확률), h=7
실행: python src/tails/T02_tail_shape_size.py
"""
from tail_utils import SERIES, fit, get_logger, load, row, save

log = get_logger("tail_T02")
rows = []
for h in [1, 7]:
    e = load(h).dropna(subset=["absS"])
    d = e.dropna(subset=["tail_ratio_z"])
    rows.append(row("T2", f"|S| ~ tail_ratio (h={h})", "-", "pooled", fit("absS ~ tail_ratio_z + C(series)", d, True), "tail_ratio_z", h == 1, "greater"))
    d = e.dropna(subset=["tail1_z"])
    rows.append(row("T2", f"|S| ~ tail1 (h={h})", "-", "pooled", fit("absS ~ tail1_z + C(series)", d, True), "tail1_z", False, "greater"))
    for s in SERIES:
        g = e[e["series"] == s].dropna(subset=["tail_ratio_z"])
        if len(g) >= 10:
            rows.append(row("T2", f"|S| ~ tail_ratio (h={h})", "-", s, fit("absS ~ tail_ratio_z", g, False), "tail_ratio_z", False, "greater"))
save(rows, "T02_tail_shape_size", log)
