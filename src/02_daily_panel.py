"""2단계: 지표×거래일 패널.

각 지표·날짜에서 만기가 가장 가까운(아직 만기 전인) Kalshi 이벤트 하나의 분포를 씀 ※
  sd      분포 표준편차 = √variance (DKW 적률)
  H       섀넌 엔트로피 −Σ p_j ln p_j (p_j: 각 구간 확률, 0인 구간은 기여 0)
  H_norm  H / ln J (J: 그 분포의 구간 수). 이벤트마다 기준선 개수가 달라서 정규화 ※
  days_to_expiry  만기까지 남은 일수 (발표가 가까워지면 분포가 저절로 좁아지는 효과 통제용)
  step    기준선 간격 중앙값 (계약 설계 변경 통제용) ※
  traded  그날 Kalshi 거래량 > min_volume (거래 없는 날은 DKW 방식상 전날 가격을 이어 씀)

MOVE 거래일에만 행을 둠. 변화량:
  dMOVE   MOVE − 직전 MOVE 거래일 값
  d_sd 등  같은 이벤트 안에서 직전 MOVE 거래일 대비 변화 (이벤트가 바뀌는 날은 결측: 분포가 새 이벤트로 바뀌어 생긴 변화라서)

입력: data/input/*   출력: data/daily_panel.parquet
실행: python src/02_daily_panel.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DATA, get_logger, load_config


def entropy_table(dist: pd.DataFrame) -> pd.DataFrame:
    p = dist["probability"] / 100.0
    dist = dist.assign(plogp=np.where(p > 0, -p * np.log(p.where(p > 0, 1.0)), 0.0))
    g = dist.groupby(["series", "contract_preamble", "date"])
    out = g.agg(H=("plogp", "sum"), J=("probability", "size"), psum=("probability", "sum")).reset_index()
    out["H_norm"] = out["H"] / np.log(out["J"])
    # 기준선 간격(중앙값): Kalshi가 계약 설계를 바꾸면(예: 고용 10만→1만 명) 폭·엔트로피가 기계적으로 변함
    step = (dist.sort_values("strike").groupby(["series", "contract_preamble", "date"])["strike"]
            .apply(lambda x: float(np.median(np.diff(x.to_numpy()))) if len(x) > 1 else np.nan)
            .rename("step").reset_index())
    return out.merge(step, on=["series", "contract_preamble", "date"])


def main() -> None:
    cfg = load_config()
    log = get_logger("02_daily_panel")
    inp = DATA / "input"

    log.info("[1/4] 입력 읽기")
    mom = pd.read_parquet(inp / "kalshi_moments.parquet")
    dist = pd.read_parquet(inp / "kalshi_dist.parquet")
    move = pd.read_csv(inp / "move_daily.csv", parse_dates=["date"]).dropna().sort_values("date")
    mom = mom[mom["series"].isin(cfg["series"])]
    log.info("  적률 %d행, 분포 %d행, MOVE %d일", len(mom), len(dist), len(move))

    log.info("[2/4] 엔트로피 계산")
    ent = entropy_table(dist)
    bad = (ent["psum"] - 100).abs() > 1e-6
    log.info("  분포 %d개 | 확률 합 100 아닌 분포 %d개 | J 중앙값 %d", len(ent), int(bad.sum()), int(ent["J"].median()))

    log.info("[3/4] 지표·날짜별로 만기 가장 가까운 이벤트 선택")
    mom = mom.merge(ent[["series", "contract_preamble", "date", "H", "H_norm", "J", "step"]],
                    on=["series", "contract_preamble", "date"], how="left")
    mom["days_to_expiry"] = (mom["expiry_date"] - mom["date"]).dt.days
    mom = mom[(mom["days_to_expiry"] >= 0) & (mom["days_to_expiry"] <= cfg["daily"]["max_days_to_expiry"])]
    n_before = len(mom)
    mom = mom.sort_values(["series", "date", "days_to_expiry"]).drop_duplicates(["series", "date"], keep="first")
    log.info("  %d행 → %d행 (같은 날 여러 이벤트면 만기 가까운 것)", n_before, len(mom))
    mom["sd"] = np.sqrt(mom["variance"])
    mom["traded"] = mom["daily_volume"] > cfg["daily"]["min_volume"]

    log.info("[4/4] MOVE 거래일과 결합, 변화량")
    move = move.rename(columns={"value": "MOVE"})
    move["dMOVE"] = move["MOVE"].diff()
    rows = []
    for s, g in mom.groupby("series"):
        g = move.merge(g, on="date", how="inner").sort_values("date")
        same = g["contract_preamble"].eq(g["contract_preamble"].shift())
        for v in ["sd", "H", "H_norm"]:
            g[f"d_{v}"] = g[v].diff().where(same)
        rows.append(g)
        log.info("  %s: %d거래일 (%s ~ %s), 이벤트 %d개, 거래 있는 날 %.0f%%, 변화량 산출 %d",
                 s, len(g), g["date"].min().date(), g["date"].max().date(),
                 g["contract_preamble"].nunique(), 100 * g["traded"].mean(), int(g["d_sd"].notna().sum()))
    panel = pd.concat(rows, ignore_index=True)
    cols = ["series", "date", "contract_preamble", "expiry_date", "days_to_expiry", "daily_volume", "traded",
            "sd", "H", "H_norm", "J", "step", "MOVE", "dMOVE", "d_sd", "d_H", "d_H_norm"]
    panel[cols].to_parquet(DATA / "daily_panel.parquet", index=False)
    log.info("저장 data/daily_panel.parquet (%d행)", len(panel))


if __name__ == "__main__":
    main()
