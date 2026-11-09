"""T6: 꼬리 비대칭 정보는 이미 가격에 반영돼 있다 → 발표 구간 선물 수익률을 예측하지 못한다.

근거: 효율적 시장 가정 하의 Campbell & Sharpe식 논리 ※ (예측 가능한 정보는 사전 반영)
모형: r_win = a + b·asym1_h + 지표 고정효과,  asym1_h = 매파 방향 꼬리 비대칭(지표 안 표준화 × 부호)
주 검정: b = 0 (양측; 유의하면 '반영 안 됨'), ZT·ZN
보조: 발표 전 30분 r_pre ~ asym1_h (b<0: 매파 비대칭이면 미리 하락), ES·SR3
실행: python src/tails/T06_asym_priced.py
"""
from tail_utils import fit, get_logger, load, row, save, targets

log = get_logger("tail_T06")
e = load(1).dropna(subset=["asym1_h"])
rows = []
for t, main in targets(e):
    d = e.dropna(subset=[f"{t}_win"]).rename(columns={f"{t}_win": "y"})
    rows.append(row("T6", "r_win ~ asym1_h", t, "pooled", fit("y ~ asym1_h + C(series)", d, True), "asym1_h", main, "two-sided"))
    d = e.dropna(subset=[f"{t}_pre"]).rename(columns={f"{t}_pre": "y"})
    rows.append(row("T6", "r_pre ~ asym1_h", t, "pooled", fit("y ~ asym1_h + C(series)", d, True), "asym1_h", False, "less"))
save(rows, "T06_asym_priced", log)
