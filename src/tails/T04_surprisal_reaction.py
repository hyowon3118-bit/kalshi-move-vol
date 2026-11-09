"""T4: 실제 발표치가 Kalshi가 낮은 확률을 준 구간에 떨어질수록(놀람 정도↑) 선물이 더 크게 움직인다 (|S|를 통제해도).

근거 ※ (정보량 −ln p 개념. 컨센서스 대비 거리 |S|와 별개로 '시장이 예상한 분포 대비 놀람'이 가격을 움직이는지)
모형: |r_win| = a + b1·|S| + b2·surprisal + 지표 고정효과,  surprisal = −ln p(실현 구간), 최소 1%
주 검정: b2 > 0, ZT·ZN.  보조: ES·SR3 (있으면)
실행: python src/tails/T04_surprisal_reaction.py
"""
from tail_utils import fit, get_logger, load, row, save, targets

log = get_logger("tail_T04")
e = load(1).dropna(subset=["absS", "surprisal"])
rows = []
for t, main in targets(e):
    d = e.dropna(subset=[f"{t}_win"]).assign(y=lambda x: x[f"{t}_win"].abs())
    rows.append(row("T4", "|r_win| ~ |S| + surprisal", t, "pooled", fit("y ~ absS + surprisal + C(series)", d, True), "surprisal", main, "greater"))
    rows.append(row("T4", "|r_win| ~ surprisal only", t, "pooled", fit("y ~ surprisal + C(series)", d, True), "surprisal", False, "greater"))
save(rows, "T04_surprisal_reaction", log)
