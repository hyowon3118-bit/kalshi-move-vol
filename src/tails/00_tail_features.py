"""꼬리·비매끄러움 0단계: 이벤트마다 발표 h일 전(h=1, 7) Kalshi 분포에서 특징량 계산.

분포 구간 정의 (fixers26-1research src/features/01_event_panel.py의 bin_edges와 같음)
  행 j의 확률 = 발표치가 (s_j + u/2, s_{j+1} + u/2] 에 들 확률 (u = 발표 단위: 0.1%p, 고용 1000명)
  맨 아래·맨 위 행은 열린 구간. 꼬리확률 계산에서만 바로 옆 구간 폭으로 닫음 ※

특징량 (모두 ※ = 레퍼런스 없는 내 정의, 근거 있는 건 표시)
  꼬리
    tail_up1, tail_dn1   P(X > F+σ), P(X < F−σ)  (F 컨센서스, σ 과거 서프라이즈 표준편차; 구간 안 균등분포)
    tail1, asym1         합, 차(위−아래)
    tail_up2, tail_dn2, tail2   ±2σ 버전
    open_tail            열린 양끝 구간 확률 합
    tail_ratio           (q90−q10)/(q75−q25)  꼬리가 몸통보다 얼마나 넓나 (정규분포면 약 1.90)
  비매끄러움
    n_modes              확률 1% 이상인 국소 최댓값 개수, multimodal = n_modes ≥ 2
    rough                Σ(Δ²p)² / Σp²  (인접 구간 확률의 2차 차분 크기, 들쭉날쭉할수록 큼)
    holes                양옆에 확률이 있는데 0인 내부 구간 수
    kl_normal            같은 평균·표준편차의 정규분포를 같은 구간에 나눈 것과의 KL 발산 (정규성에서 벗어난 정도)
    viol_raw             원가격(정리 전 마지막 체결가)에서 기준선이 높은데 가격이 더 높은 인접 쌍 비율 (단조성 위반)
    adj_l1               DKW 정리 과정에서 바뀐 가격 크기 평균 |정리 후 − 원가격| (¢)
  실현값 관련 (사후 변수, 결과 변수로만 사용)
    surprisal            −ln p(실제 발표치가 든 구간). 최소 1% (Kalshi 최소 호가 1¢) ※
    hit_up1, hit_dn1, hit_up2, hit_dn2   실제로 F±σ, F±2σ 밖에 떨어졌나
  bins (보정 분석용): 구간별 예측확률과 실현 여부 → data/tail_bins.parquet

입력: data/input/kalshi_dist.parquet, event_panel.parquet
출력: data/tail_features.parquet (event_ticker, h, 특징량), data/tail_bins.parquet
실행: python src/tails/00_tail_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, get_logger  # noqa: E402

UNIT = {"KXCPI": 0.1, "KXCPICORE": 0.1, "KXU3": 0.1, "KXGDP": 0.1, "KXPAYROLLS": 1000.0}
P_FLOOR = 0.01


def edges(strikes: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
    """닫힌 구간 경계. 열린 양끝은 바로 옆 구간 폭으로 닫음 ※."""
    lo = np.r_[np.nan, strikes[1:] + u / 2]
    hi = np.r_[strikes[1:] + u / 2, np.nan]
    if len(strikes) >= 3:
        lo[0] = hi[0] - (hi[1] - lo[1])
        hi[-1] = lo[-1] + (hi[-2] - lo[-2])
    else:
        lo[0], hi[-1] = hi[0] - u, lo[-1] + u
    return lo, hi


def prob_above(lo, hi, p, c) -> float:
    frac = np.clip((hi - c) / (hi - lo), 0, 1)
    return float((p * frac).sum())


def features(g: pd.DataFrame, F: float, sig: float, A: float, u: float) -> tuple[dict, list[dict]]:
    g = g.sort_values("strike")
    s = g["strike"].to_numpy(float)
    p = g["probability"].to_numpy(float)
    p = p / p.sum()
    lo, hi = edges(s, u)
    mid = (lo + hi) / 2
    f = {}
    for k in (1, 2):
        up = prob_above(lo, hi, p, F + k * sig)
        dn = 1 - prob_above(lo, hi, p, F - k * sig)
        f[f"tail_up{k}"], f[f"tail_dn{k}"], f[f"tail{k}"] = up, dn, up + dn
        f[f"hit_up{k}"] = float(A > F + k * sig)
        f[f"hit_dn{k}"] = float(A < F - k * sig)
    f["asym1"] = f["tail_up1"] - f["tail_dn1"]
    f["asym2"] = f["tail_up2"] - f["tail_dn2"]
    f["open_tail"] = p[0] + p[-1]

    # 비매끄러움
    inner = np.r_[False, (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]), False]
    edge_max = np.r_[p[0] > p[1] if len(p) > 1 else False, np.zeros(len(p) - 2, bool), p[-1] > p[-2] if len(p) > 1 else False]
    f["n_modes"] = int(((inner | edge_max) & (p >= 0.01)).sum())
    f["multimodal"] = float(f["n_modes"] >= 2)
    d2 = np.diff(p, 2)
    f["rough"] = float((d2 ** 2).sum() / (p ** 2).sum()) if len(p) > 2 else np.nan
    nz = np.flatnonzero(p > 0)
    f["holes"] = int(((p[nz[0]:nz[-1] + 1]) == 0).sum()) if len(nz) else 0
    mu = (p * mid).sum()
    sd = np.sqrt((p * (mid - mu) ** 2).sum())
    q = np.diff(stats.norm.cdf(np.r_[-np.inf, hi[:-1], np.inf], mu, sd)) if sd > 0 else np.full(len(p), np.nan)
    q = np.clip(q, 1e-6, None)
    f["kl_normal"] = float((p[p > 0] * np.log(p[p > 0] / q[p > 0])).sum())
    raw = g["yes_price"].to_numpy(float)
    pairs = (raw[:-1] > 0) | (raw[1:] > 0)
    f["viol_raw"] = float(((raw[1:] > raw[:-1]) & pairs).sum() / max(pairs.sum(), 1))
    f["adj_l1"] = float(np.abs(g["adjusted_yes_price"].to_numpy(float) - raw).mean())

    # 실현 구간
    j = np.flatnonzero((A > np.r_[-np.inf, lo[1:]]) & (A <= np.r_[hi[:-1], np.inf]))
    pj = p[j[0]] if len(j) else np.nan
    f["p_realized"] = pj
    f["surprisal"] = -np.log(max(pj, P_FLOOR)) if np.isfinite(pj) else np.nan
    bins = [{"p": float(p[i]), "hit": float(len(j) > 0 and i == j[0]), "is_open": i in (0, len(p) - 1)}
            for i in range(len(p))]
    return f, bins


def main() -> None:
    log = get_logger("tails_00_features")
    log.info("[1/3] 입력")
    dist = pd.read_parquet(DATA / "input" / "kalshi_dist.parquet")
    dist["date"] = pd.to_datetime(dist["date"])
    ev = pd.read_parquet(DATA / "input" / "event_panel.parquet")
    ev["obs_date"] = pd.to_datetime(ev["obs_date"])
    ev = ev.dropna(subset=["F", "sigma", "A", "obs_date"])
    log.info("  이벤트×h %d행 (컨센서스·σ·실제치·관측일 있음)", len(ev))

    log.info("[2/3] 특징량 계산")
    groups = {k: g for k, g in dist.groupby(["contract_preamble", "date"])}
    rows, bins = [], []
    miss = 0
    for r in ev.itertuples():
        g = groups.get((r.event_ticker, r.obs_date))
        if g is None or len(g) < 3:
            miss += 1
            continue
        f, b = features(g, r.F, r.sigma, r.A, UNIT[r.series])
        rows.append({"event_ticker": r.event_ticker, "series": r.series, "h": r.h, **f})
        bins += [{"event_ticker": r.event_ticker, "series": r.series, "h": r.h, "release_date": r.release_date_actual, **x} for x in b]
    out = pd.DataFrame(rows)
    log.info("  계산 %d행, 분포 못 찾음 %d", len(out), miss)

    log.info("[3/3] 요약·저장")
    for c in ["tail1", "tail2", "asym1", "open_tail", "n_modes", "rough", "holes", "kl_normal", "viol_raw", "adj_l1", "surprisal"]:
        log.info("  %-10s 평균 %.3f  중앙값 %.3f  (h=1)", c, out.loc[out.h == 1, c].mean(), out.loc[out.h == 1, c].median())
    log.info("  다봉 분포 비율 %.1f%% | 실제 F±σ 밖 %.1f%% vs Kalshi 예측 %.1f%% (h=1)",
             100 * out.loc[out.h == 1, "multimodal"].mean(),
             100 * (out.loc[out.h == 1, "hit_up1"] + out.loc[out.h == 1, "hit_dn1"]).mean(),
             100 * out.loc[out.h == 1, "tail1"].mean())
    out.to_parquet(DATA / "tail_features.parquet", index=False)
    pd.DataFrame(bins).to_parquet(DATA / "tail_bins.parquet", index=False)
    log.info("저장 data/tail_features.parquet, data/tail_bins.parquet")


if __name__ == "__main__":
    main()
