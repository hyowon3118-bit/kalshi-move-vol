"""가설 F: Kalshi 분포의 왜도 방향으로 실제 발표치가 나온다 (평균 괴리 Gap을 통제해도).

근거: 선행연구 확인 못 함 ※ (탐색용)
모형: S = a + b·skew + c·Gap (+ 지표 고정효과),  skew = Kalshi 분포 왜도(DKW 적률), 지표 안 표준화
      양(+)의 왜도 = 높은 값 쪽 꼬리가 김 → 발표치가 컨센서스보다 높게 나올 것 → b > 0
주 검정: 통합 b > 0 (단측), 발표 1일 전(h=1)·7일 전(h=7).  보조: 지표별
실행: python src/hyp/hF_skew_direction.py
"""
from hyp_utils import SERIES, fit, get_logger, load_events, row, save, zscore_within

log = get_logger("hyp_F")
rows = []
for h in [1, 7]:
    e = load_events(h).dropna(subset=["S", "skewness", "Gap"])
    e["skz"] = zscore_within(e, "skewness")
    rows.append(row("F", f"S ~ skew + Gap (h={h})", "-", "pooled", fit("S ~ skz + Gap + C(series)", e, True), "skz", True, "greater"))
    for s in SERIES:
        g = e[e["series"] == s]
        rows.append(row("F", f"S ~ skew + Gap (h={h})", "-", s, fit("S ~ skz + Gap", g, False), "skz", False, "greater"))
save(rows, "F_skew_direction", log)
