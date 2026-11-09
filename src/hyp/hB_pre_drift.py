"""가설 B: 발표 전 30분(8:00→8:30) 선물 가격이 발표 내용 방향으로 미리 움직인다.

B1 (재현) Kurov, Sancetta, Strasser & Wolfe (2019): 주가지수·국채선물이 발표 약 30분 전부터 '맞는' 방향으로 움직임
   r_pre = a + b·S_h (+ 지표 고정효과),  S_h = 매파 방향으로 부호 맞춘 서프라이즈. 매파적이면 금리↑·선물↓ → b < 0
B2 (확장 ※) 발표 1일 전 Kalshi 괴리 Gap_h 방향으로 발표 전 30분에 움직이나: r_pre = a + b·Gap_h, b < 0
주 검정: 통합 B1·B2, ZT·ZN (단측 b<0).  보조: 지표별
실행: python src/hyp/hB_pre_drift.py
"""
from hyp_utils import SERIES, TARGETS, fit, get_logger, load_events, row, save

log = get_logger("hyp_B")
e = load_events(1)
rows = []
for t in TARGETS:
    for x, name in [("S_h", "B1 r_pre ~ S_h"), ("Gap_h", "B2 r_pre ~ Gap_h")]:
        d = e.dropna(subset=[f"{t}_pre", x]).rename(columns={f"{t}_pre": "y"})
        rows.append(row("B", name, t, "pooled", fit(f"y ~ {x} + C(series)", d, True), x, True, "less"))
        for s in SERIES:
            g = d[d["series"] == s]
            rows.append(row("B", name, t, s, fit(f"y ~ {x}", g, False), x, False, "less"))
save(rows, "B_pre_drift", log)
