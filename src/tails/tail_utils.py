"""꼬리 가설 공용: 이벤트 표 + 꼬리 특징량 + 발표 구간 선물 수익률."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hyp"))
from hyp_utils import DATA, SERIES, SIGN, TABLES, fit, get_logger, load_events, row, save, zscore_within  # noqa: E402,F401

MAIN_TARGETS = ["ZT", "ZN"]           # 주 검정 종속변수 (사전에 고정)
EXTRA_TARGETS = ["ES", "SR3"]         # 추가 수집 종속변수 (있으면 보조 검정, 별도 BH)


def load(h: int = 1) -> pd.DataFrame:
    e = load_events(h)
    f = pd.read_parquet(DATA / "tail_features.parquet")
    e = e.merge(f[f["h"] == h].drop(columns=["series", "h"]), on="event_ticker", how="left")
    for c in ["tail1", "tail2", "asym1", "tail_ratio", "rough", "kl_normal", "viol_raw", "open_tail", "W_iqr"]:
        if c == "tail_ratio":
            e[c] = (e["q90"] - e["q10"]) / (e["q75"] - e["q25"])
        e[f"{c}_z"] = zscore_within(e, c)
    e["asym1_h"] = e["asym1_z"] * e["sign"]
    e["absS"] = e["S"].abs()
    return e


def targets(e: pd.DataFrame) -> list[tuple[str, bool]]:
    """(종속변수, 주 검정 여부). 추가 종속변수는 event_windows에 있을 때만."""
    out = [(t, True) for t in MAIN_TARGETS]
    out += [(t, False) for t in EXTRA_TARGETS if f"{t}_win" in e.columns and e[f"{t}_win"].notna().any()]
    return out
