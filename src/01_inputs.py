"""1단계: 입력 자료 준비.

- Kalshi: fixers26-1research가 만든 data/interim/kalshi_dist.parquet, kalshi_moments.parquet, kalshi_events.parquet(발표일) 복사
          (연준 DKW 방식으로 만든 일별 확률분포와 적률. 다시 수집하지 않음)
- MOVE: fixers26-1research의 data/raw/move/move_daily.csv가 있으면 복사, 없으면 yfinance(^MOVE)로 다운로드

출력: data/input/kalshi_*.parquet, event_panel.parquet, rq1_oos_predictions.parquet, move_daily.csv
실행: python src/01_inputs.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from common import DATA, ROOT, get_logger, load_config


def main() -> None:
    cfg = load_config()
    log = get_logger("01_inputs")
    src = Path(cfg["source_repo"])
    src = src if src.is_absolute() else (ROOT / src).resolve()
    out = DATA / "input"
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/2] Kalshi 분포 복사 (원본: %s)", src)
    snap = src / "data" / "interim" / "kalshi_release_snapshot.parquet"
    if snap.exists():
        shutil.copy2(snap, out / snap.name)
        log.info("  kalshi_release_snapshot.parquet (발표 직전 60분 IQR): %d행", len(pd.read_parquet(snap)))
    for name in ["kalshi_dist.parquet", "kalshi_moments.parquet", "kalshi_events.parquet"]:
        f = src / "data" / "interim" / name
        if not f.exists():
            raise FileNotFoundError(f"{f} 없음. fixers26-1research에서 src/clean/01_kalshi_distributions.py를 먼저 실행해")
        shutil.copy2(f, out / name)
        log.info("  %s: %d행", name, len(pd.read_parquet(out / name)))

    log.info("  가설 검증용 이벤트 패널 (fixers26-1research data/processed)")
    for name in ["event_panel.parquet", "rq1_oos_predictions.parquet"]:
        f = src / "data" / "processed" / name
        if f.exists():
            shutil.copy2(f, out / name)
            log.info("  %s: %d행", name, len(pd.read_parquet(out / name)))
        else:
            log.warning("  %s 없음 (fixers26-1research에서 01_event_panel.py, rq1_forecast.py 실행 필요). src/hyp 가설 검증에만 필요", name)

    log.info("[2/2] MOVE 일별")
    f = src / "data" / "raw" / "move" / "move_daily.csv"
    if f.exists():
        shutil.copy2(f, out / "move_daily.csv")
        log.info("  fixers26-1research에서 복사")
    else:
        import yfinance as yf
        log.info("  yfinance ^MOVE 다운로드 (%s~)", cfg["move"]["start"])
        d = yf.download("^MOVE", start=cfg["move"]["start"], interval="1d", progress=False, auto_adjust=False)
        if d.empty:
            raise RuntimeError("MOVE 다운로드 실패")
        close = d["Close"].squeeze()
        pd.DataFrame({"date": close.index.date, "value": close.to_numpy()}).to_csv(out / "move_daily.csv", index=False)
    m = pd.read_csv(out / "move_daily.csv")
    log.info("  MOVE %d일 (%s ~ %s)", len(m), m["date"].min(), m["date"].max())


if __name__ == "__main__":
    main()
