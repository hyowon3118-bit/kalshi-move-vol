"""가설 공용 0단계: 발표일마다 오전 8:30(미 동부시간) 전후 국채선물 수익률.

다섯 지표(CPI, 근원 CPI, 고용, 실업률, GDP 속보치) 모두 8:30 ET 발표
가격 P(T) = 시각 T 직전에 끝난 1분봉의 종가 (1분봉 시각은 시작 시각이라, T−1분 봉의 종가 = T 직전 가격)
            T 앞 10분 안에 봉이 없으면 결측 ※
수익률(bp) = 10,000 · ln(P(t1)/P(t0)),  구간 안에서 근월물이 바뀌면 결측
구간 ※
  pre   8:00 → 8:30  발표 전 30분 (Kurov 외 2019: 발표 약 30분 전부터 가격 이동)
  post  8:30 → 8:50  발표 후 20분 (Balduzzi·Elton·Green 2001: 조정은 대체로 1분 안에 끝남 → 여유 있게 20분)
  win   8:20 → 8:50  발표를 포함한 30분 (발표 10분 전 ~ 20분 후)

입력: data/raw/glbx_treasury_ohlcv1m.dbn.zst (+ glbx_extra_ohlcv1m.dbn.zst 있으면), data/input/event_panel.parquet
출력: data/event_windows.csv (date, {ZT,ZN,ES,SR3}_{pre,post,win})
실행: python src/hyp/00_event_windows.py
"""
from __future__ import annotations

import databento as db
import numpy as np
import pandas as pd

from hyp_utils import DATA, get_logger, load_config

WINDOWS = {"pre": ("08:00", "08:30"), "post": ("08:30", "08:50"), "win": ("08:20", "08:50")}
MAX_GAP = pd.Timedelta(minutes=10)


def price_at(bars: pd.DataFrame, t: pd.Timestamp) -> tuple[float, float]:
    before = bars.loc[:t - pd.Timedelta(minutes=1)]
    if before.empty or t - before.index[-1] > MAX_GAP:
        return np.nan, np.nan
    return float(before["close"].iloc[-1]), float(before["instrument_id"].iloc[-1])


def main() -> None:
    cfg = load_config()
    log = get_logger("hyp_00_event_windows")
    log.info("[1/3] 선물 1분봉 읽기 (동부시간 변환; 국채선물 + 있으면 ES·SR3)")
    bars = {}
    for block in ["databento", "databento_extra"]:
        path = DATA.parent / cfg[block]["out_file"]
        if not path.exists():
            log.info("  %s 없음 → 건너뜀", path.name)
            continue
        df = db.DBNStore.from_file(path).to_df(tz="America/New_York")
        bars.update({s.split(".")[0]: g[["close", "instrument_id"]].sort_index() for s, g in df.groupby("symbol")})
    log.info("  %s", {k: len(v) for k, v in bars.items()})

    log.info("[2/3] 발표일 목록")
    p = pd.read_parquet(DATA / "input" / "event_panel.parquet")
    dates = sorted(pd.to_datetime(p["release_date_actual"]).dt.normalize().unique())
    log.info("  발표일 %d개 (%s ~ %s)", len(dates), dates[0].date(), dates[-1].date())

    log.info("[3/3] 구간 수익률")
    rows = []
    for d in dates:
        r = {"date": d.date()}
        for k, b in bars.items():
            for w, (a, z) in WINDOWS.items():
                t0 = pd.Timestamp(f"{d.date()} {a}", tz="America/New_York")
                t1 = pd.Timestamp(f"{d.date()} {z}", tz="America/New_York")
                p0, i0 = price_at(b, t0)
                p1, i1 = price_at(b, t1)
                r[f"{k}_{w}"] = 1e4 * np.log(p1 / p0) if i0 == i1 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(DATA / "event_windows.csv", index=False)
    for c in out.columns[1:]:
        log.info("  %-8s 결측 %d | |수익률| 중앙값 %.1fbp", c, int(out[c].isna().sum()), out[c].abs().median())
    log.info("저장 data/event_windows.csv (%d일)", len(out))


if __name__ == "__main__":
    main()
