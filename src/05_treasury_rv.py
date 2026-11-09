"""5단계: 국채선물 1분봉 → 일별 실현변동성.

정의: RV_d = Σ r_k²,  r_k = ln(P_k / P_{k−1}) 5분 로그수익률 (Andersen–Bollerslev–Diebold–Labys 2003의 정의,
      5분 간격은 Liu, Patton & Sheppard 2015)
절차
  1. 시간을 시카고 시간으로 바꾸고, 17:00 이후는 다음 거래일로 배정 (CME 거래일: 전날 17:00 ~ 당일 16:00) ※
  2. 5분 구간마다 마지막 1분봉 종가와 계약 번호(instrument_id). 거래 없는 구간은 직전 가격 유지 ※
  3. 같은 거래일 안의 연속 구간 사이 수익률만 사용 → 장 휴식 시간·주말을 넘는 갭 수익률은 제외 ※
  4. 근월물 교체(계약 번호가 바뀜)로 생긴 수익률은 제외: 가격이 조정되지 않은 원래 가격이라 교체 순간 가짜 움직임이 생김
  5. 거래 없는 5분 구간 비율 > max_empty_share인 날(휴일 단축장 등)은 결측 ※
  6. 연율화 변동성(%) = 100·√(252·RV) ※

입력: config.yaml databento(·databento_gap).out_file (04_databento_treasury.py 출력)
출력: data/treasury_rv_daily.csv  (date, rv_ZT, vol_ZT, n_ZT, empty_ZT, rv_ZN, ...)
실행: python src/05_treasury_rv.py
"""
from __future__ import annotations

import databento as db
import numpy as np
import pandas as pd

from common import DATA, ROOT, get_logger, load_config


def daily_rv(bars: pd.DataFrame, minutes: int, session_end: str, max_empty: float) -> pd.DataFrame:
    """bars: index=ts(시카고 시간, tz-aware), columns close, volume, instrument_id."""
    end_h, end_m = map(int, session_end.split(":"))
    # 마감 1시간 뒤(17:00)에 다음 거래일 시작 → 그 시각을 자정으로 옮기는 이동량 (16:00 마감이면 7시간)
    shift = pd.Timedelta(hours=24) - pd.Timedelta(hours=end_h + 1, minutes=end_m)
    b = bars.copy()
    b["bucket"] = b.index.floor(f"{minutes}min")
    agg = b.groupby("bucket").agg(close=("close", "last"), inst=("instrument_id", "last"), volume=("volume", "sum"))
    agg["session"] = (agg.index.tz_localize(None) + shift).normalize()
    n_grid = int(23 * 60 / minutes)  # 17:00~16:00 = 23시간

    rows = []
    for day, g in agg.groupby("session"):
        start = (day - shift).tz_localize(bars.index.tz, ambiguous="NaT", nonexistent="shift_forward")
        grid = pd.date_range(start, periods=n_grid, freq=f"{minutes}min")
        g = g.reindex(grid)
        traded = g["close"].notna()
        g[["close", "inst"]] = g[["close", "inst"]].ffill()
        r = np.log(g["close"]).diff()
        roll = g["inst"].ne(g["inst"].shift()) & g["inst"].shift().notna()
        r = r.where(~roll)
        empty = 1 - traded.mean()
        rv = float((r.dropna() ** 2).sum())
        rows.append({"date": day.date(), "rv": rv if empty <= max_empty else np.nan,
                     "n": int(r.notna().sum()), "empty": empty, "rolls": int(roll.sum())})
    out = pd.DataFrame(rows)
    out["vol"] = 100 * np.sqrt(252 * out["rv"])
    return out


def main() -> None:
    cfg = load_config()
    dcfg, rcfg = cfg["databento"], cfg["rv"]
    log = get_logger("05_treasury_rv")
    paths = [ROOT / cfg[b]["out_file"] for b in ["databento_gap", "databento"] if b in cfg and (ROOT / cfg[b]["out_file"]).exists()]
    if not paths:
        raise FileNotFoundError("국채선물 DBN 파일 없음. src/04_databento_treasury.py를 먼저 실행해")

    log.info("[1/3] DBN 읽기: %s", [p.name for p in paths])
    df = pd.concat([db.DBNStore.from_file(p).to_df(tz="America/Chicago") for p in paths]).sort_index()
    log.info("  1분봉 %d개, 종목 %s", len(df), sorted(df["symbol"].unique()))

    log.info("[2/3] 5분 실현변동성 (근월물 교체·장 휴식 갭 제외)")
    merged = None
    for sym, g in df.groupby("symbol"):
        root = sym.split(".")[0]
        d = daily_rv(g[["close", "volume", "instrument_id"]], rcfg["bar_minutes"], rcfg["session_end_ct"],
                     rcfg["max_empty_share"])
        log.info("  %s: 거래일 %d, 결측(거래 희박) %d, 근월물 교체 제외 %d회, 5분 수익률 중앙값 %d개/일, 연율 변동성 중앙값 %.2f%%",
                 root, len(d), int(d["rv"].isna().sum()), int(d["rolls"].sum()), int(d["n"].median()), d["vol"].median())
        d = d.drop(columns="rolls").set_index("date").add_suffix(f"_{root}")
        merged = d if merged is None else merged.join(d, how="outer")

    log.info("[3/3] 저장")
    merged.reset_index().to_csv(DATA / "treasury_rv_daily.csv", index=False)
    log.info("  data/treasury_rv_daily.csv (%d일)", len(merged))


if __name__ == "__main__":
    main()
