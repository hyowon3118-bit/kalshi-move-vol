"""가설 A: 서프라이즈가 클수록 발표 직후 국채선물이 크게 움직인다.

근거: Balduzzi, Elton & Green (2001) — 정기 경제지표 발표가 2년·10년 국채 가격을 유의하게 움직이고 조정은 약 1분 안에 끝남
모형: |r_win| = a + b·|S| (+ 지표 고정효과),  r_win = 8:20→8:50 ET 선물 수익률(bp), S = (발표치−컨센서스)/σ
주 검정: 통합(다섯 지표) b > 0 (단측), ZT·ZN 각각.  보조: 지표별
의미: 데이터·시점이 맞는지 확인하는 기준점. 이게 안 나오면 이벤트 시각이나 데이터에 문제
실행: python src/hyp/hA_surprise_size.py
"""
from hyp_utils import SERIES, TARGETS, fit, get_logger, load_events, row, save

log = get_logger("hyp_A")
e = load_events(1).dropna(subset=["S"])
e["absS"] = e["S"].abs()
rows = []
for t in TARGETS:
    d = e.dropna(subset=[f"{t}_win"]).assign(y=lambda x: x[f"{t}_win"].abs())
    rows.append(row("A", "|r_win| ~ |S|", t, "pooled", fit("y ~ absS + C(series)", d, True), "absS", True, "greater"))
    for s in SERIES:
        g = d[d["series"] == s]
        rows.append(row("A", "|r_win| ~ |S|", t, s, fit("y ~ absS", g, False), "absS", False, "greater"))
save(rows, "A_surprise_size", log)
