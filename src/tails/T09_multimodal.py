"""T9: 분포가 봉우리 둘 이상(시장 의견이 갈림)이면 실제 서프라이즈와 선물 반응이 크다.

근거: Pasquariello & Vega (2007) — 의견 분산이 클 때 주문흐름-금리 관계가 강함 (간접). 다봉을 의견 분열로 보는 건 ※
모형: |S| = a + b·multimodal + c·W_iqr + 지표 고정효과
      |r_win| = a + b·multimodal + c·|S| + 지표 고정효과
주 검정: b > 0 (|S|, ZT).  보조: ZN·ES·SR3
실행: python src/tails/T09_multimodal.py
"""
from tail_utils import fit, get_logger, load, row, save, targets

log = get_logger("tail_T09")
e = load(1).dropna(subset=["multimodal", "absS"])
rows = [row("T9", "|S| ~ multimodal + W", "-", "pooled",
            fit("absS ~ multimodal + W_iqr_z + C(series)", e.dropna(subset=["W_iqr_z"]), True), "multimodal", True, "greater")]
for t, main in targets(e):
    d = e.dropna(subset=[f"{t}_win"]).assign(y=lambda x: x[f"{t}_win"].abs())
    rows.append(row("T9", "|r_win| ~ multimodal + |S|", t, "pooled", fit("y ~ multimodal + absS + C(series)", d, True),
                    "multimodal", main and t == "ZT", "greater"))
save(rows, "T09_multimodal", log)
