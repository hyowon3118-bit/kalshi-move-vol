"""4단계: Databento에서 국채선물 1분봉 받기 (비용 조회 → 예산 안이면 다운로드 1회).

1) metadata.get_cost로 요청 비용(USD) 조회. 조회 자체는 과금되지 않음
2) 비용 ≤ budget_usd 이고 저장 파일이 아직 없을 때만 timeseries.get_range를 딱 한 번 호출해
   원자료(DBN, zstd 압축)를 config의 out_file에 그대로 저장. 이후 계산은 전부 이 파일로 함
   - 파일이 이미 있으면 다시 받지 않음 (중복 과금 방지)
   - 비용이 예산을 넘으면 받지 않고 종료

설정: config.yaml의 databento (GLBX.MDP3, ZT.v.0·ZN.v.0 거래량 기준 연속 근월물, ohlcv-1m)
API 키: 환경변수 DATABENTO_API_KEY 또는 repo 루트 .env 파일 (DATABENTO_API_KEY=db-...)

실행: python src/04_databento_treasury.py            (비용 확인 후 예산 안이면 다운로드)
      python src/04_databento_treasury.py --cost-only (비용만 확인)
"""
from __future__ import annotations

import argparse
import os
import time

import databento as db

from common import ROOT, get_logger, load_config


def load_key() -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DATABENTO_API_KEY="):
                os.environ.setdefault("DATABENTO_API_KEY", line.split("=", 1)[1].strip().strip('"'))
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise RuntimeError("DATABENTO_API_KEY가 없어. .env에 DATABENTO_API_KEY=db-... 한 줄 넣어줘")
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-only", action="store_true", help="비용만 조회하고 다운로드하지 않음")
    args = ap.parse_args()
    cfg = load_config()["databento"]
    log = get_logger("04_databento_treasury")
    out = ROOT / cfg["out_file"]
    req = dict(dataset=cfg["dataset"], symbols=cfg["symbols"], stype_in=cfg["stype_in"],
               schema=cfg["schema"], start=cfg["start"], end=cfg["end"])

    log.info("[1/3] 요청 내용: %s", req)
    client = db.Historical(load_key())

    log.info("[2/3] 비용 조회 (과금 없음)")
    cost = float(client.metadata.get_cost(**req))
    log.info("  예상 비용 $%.2f / 예산 $%.2f", cost, cfg["budget_usd"])

    if args.cost_only:
        log.info("--cost-only라 여기서 종료")
        return
    if out.exists():
        log.info("[3/3] 이미 받은 파일이 있어서 다운로드 안 함: %s (%.1f MB)", out.relative_to(ROOT), out.stat().st_size / 1e6)
        return
    if cost > cfg["budget_usd"]:
        log.warning("[3/3] 예산 초과라 다운로드 안 함. config.yaml의 기간(start/end)이나 symbols를 줄여서 다시 조회해")
        return

    log.info("[3/3] 다운로드 (API 호출 1회) → %s", out.relative_to(ROOT))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = client.timeseries.get_range(**req, path=out)
    log.info("  완료 %.0f초, %.1f MB", time.time() - t0, out.stat().st_size / 1e6)

    df = store.to_df()
    for sym, g in df.groupby("symbol"):
        log.info("  %s: 1분봉 %d개, %s ~ %s, 계약 %d개", sym, len(g), g.index.min(), g.index.max(), g["instrument_id"].nunique())


if __name__ == "__main__":
    main()
