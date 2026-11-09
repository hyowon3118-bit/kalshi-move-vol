"""T5: 발표 전 꼬리 확률이 클수록(폭을 고려해도) 발표 구간 선물이 크게 움직인다 (사전 정보만 사용).

근거 ※ (앞선 D·E에서 폭은 실패. 꼬리 확률 자체가 극단 결과 가능성을 따로 담는지)
모형: |r_win| = a + b·tail1 + c·W_iqr + 지표 고정효과 (둘 다 지표 안 표준화)
주 검정: b > 0, ZT·ZN.  보조: ±2σ 꼬리, ES·SR3
실행: python src/tails/T05_tail_exante_reaction.py
"""
from tail_utils import fit, get_logger, load, row, save, targets, zscore_within

log = get_logger("tail_T05")
e = load(1).dropna(subset=["tail1_z", "W_iqr_z"])
e["tail2_z"] = zscore_within(e, "tail2")
rows = []
for t, main in targets(e):
    d = e.dropna(subset=[f"{t}_win"]).assign(y=lambda x: x[f"{t}_win"].abs())
    rows.append(row("T5", "|r_win| ~ tail1 + W", t, "pooled", fit("y ~ tail1_z + W_iqr_z + C(series)", d, True), "tail1_z", main, "greater"))
    rows.append(row("T5", "|r_win| ~ tail2 + W", t, "pooled", fit("y ~ tail2_z + W_iqr_z + C(series)", d, True), "tail2_z", False, "greater"))
save(rows, "T05_tail_exante_reaction", log)
