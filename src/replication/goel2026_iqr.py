"""Goel (2026, IREF 110, 105577) 재현: Kalshi 내재 IQR이 서프라이즈 크기를 예측하나.

원 논문 주 설계 (Table 3): 헤드라인 CPI, |발표치 − 컨센서스|(%p) = α + β·IQR(%p), OLS + HC1,
  wild-bootstrap-t 95% CI (Rademacher, 5000회), 순열 p값 (10,000회). 결과 β=0.753, N=45
  IQR = 발표 직전 60분 안 마지막 체결가로 만든 CDF에서 Q75−Q25 (인접 기준선 사이 선형 보간)

우리 데이터로 바꾼 점 (차이 원인 후보)
  R0  원 설계에 가장 가까운 것: 헤드라인 CPI, 원단위 |A−F| ~ 원단위 IQR, 발표 1일 전 분포 (발표 직전 60분 분포는 없음)
  R1  발표 7일 전 분포
  R2  우리 RQ3 방식: |S| ~ W_iqr (둘 다 σ로 나눔)
  R3  섞어서: 원단위 |A−F| ~ W_iqr, |S| ~ 원단위 IQR  (어느 쪽 정규화가 결과를 바꾸는지)
  R4  이산 IQR (구간 값 기준, 보간 없음)
  R5  CPI + 근원 CPI 통합 (원 논문 pooled 57건과 대응, 같은 날 발표라 날짜 군집 표준오차 추가 보고)
  R6  기간: 2023년 vs 2024~26년, IQR×2023 상호작용 (원 논문은 2022~23 고인플레에 효과 집중)
※ 우리 Kalshi 자료는 2022-12 시작 → 원 논문 표본의 2022년 발표가 없음

입력: data/input/event_panel.parquet (fixers26-1research)
출력: outputs/tables/replication_goel2026.csv
실행: python src/replication/goel2026_iqr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, TABLES, get_logger  # noqa: E402

RNG = np.random.default_rng(20261109)
N_BOOT, N_PERM = 5000, 10000


def wild_t_ci(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """wild-bootstrap-t 95% CI (Rademacher, 무제약 잔차)."""
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit(cov_type="HC1")
    b, se, res, yhat = fit.params[1], fit.bse[1], fit.resid, fit.fittedvalues
    ts = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ystar = yhat + res * RNG.choice([-1.0, 1.0], len(y))
        f = sm.OLS(ystar, X).fit(cov_type="HC1")
        ts[i] = (f.params[1] - b) / f.bse[1]
    lo, hi = np.quantile(ts, [0.975, 0.025])
    return b - lo * se, b - hi * se


def perm_p(y: np.ndarray, x: np.ndarray) -> float:
    b = np.polyfit(x, y, 1)[0]
    bs = np.array([np.polyfit(RNG.permutation(x), y, 1)[0] for _ in range(N_PERM)])
    return float((np.abs(bs) >= abs(b)).mean())


def spec(name: str, d: pd.DataFrame, y: str, x: str, log, cluster: bool = False, boot: bool = True) -> dict:
    d = d.dropna(subset=[y, x])
    fit = smf.ols(f"{y} ~ {x}", d).fit(cov_type="HC1")
    r = {"spec": name, "y": y, "x": x, "n": len(d), "beta": fit.params[x], "se_hc1": fit.bse[x],
         "t": fit.tvalues[x], "p_hc1": fit.pvalues[x], "r2": fit.rsquared,
         "start": d["release_date_actual"].min().date(), "end": d["release_date_actual"].max().date()}
    if boot:
        r["wild_ci_lo"], r["wild_ci_hi"] = wild_t_ci(d[y].to_numpy(float), d[x].to_numpy(float))
        r["perm_p"] = perm_p(d[y].to_numpy(float), d[x].to_numpy(float))
    if cluster:
        fc = smf.ols(f"{y} ~ {x}", d).fit(cov_type="cluster", cov_kwds={"groups": d["release_date_actual"]})
        r["se_cluster"], r["p_cluster"] = fc.bse[x], fc.pvalues[x]
    log.info("  %-34s n=%2d β=%7.3f HC1 se=%.3f t=%5.2f p=%.4f R²=%.3f%s", name, r["n"], r["beta"], r["se_hc1"],
             r["t"], r["p_hc1"], r["r2"],
             f" | wild CI [{r['wild_ci_lo']:.3f}, {r['wild_ci_hi']:.3f}] 순열 p={r['perm_p']:.4f}" if boot else "")
    return r


def main() -> None:
    log = get_logger("replication_goel2026")
    p = pd.read_parquet(DATA / "input" / "event_panel.parquet")
    p["release_date_actual"] = pd.to_datetime(p["release_date_actual"])
    p["absAF"] = (p["A"] - p["F"]).abs()
    p["absS"] = p["S"].abs()
    p["iqr"] = p["q75"] - p["q25"]
    p["iqr_disc"] = p["q75_disc"] - p["q25_disc"]
    p["y2023"] = (p["release_date_actual"].dt.year == 2023).astype(float)
    rows = []

    log.info("원 논문: β=0.753 (HC1 0.169, t=4.44, R²=0.199, N=45), wild CI [0.356, 1.217], 순열 p=0.0015")
    for h in [1, 7]:
        c = p[(p["series"] == "KXCPI") & (p["h"] == h)]
        log.info("[h=%d] 헤드라인 CPI", h)
        rows.append(spec(f"R0 원단위 |A−F|~IQR (h={h})" if h == 1 else f"R1 원단위 |A−F|~IQR (h={h})", c, "absAF", "iqr", log))
        rows.append(spec(f"R2 |S|~W_iqr σ정규화 (h={h})", c, "absS", "W_iqr", log))
        rows.append(spec(f"R3a 원단위 |A−F|~W_iqr (h={h})", c, "absAF", "W_iqr", log, boot=False))
        rows.append(spec(f"R3b |S|~원단위 IQR (h={h})", c, "absS", "iqr", log, boot=False))
        rows.append(spec(f"R4 원단위 |A−F|~이산 IQR (h={h})", c, "absAF", "iqr_disc", log, boot=False))
        cc = p[p["series"].isin(["KXCPI", "KXCPICORE"]) & (p["h"] == h)]
        rows.append(spec(f"R5 CPI+근원 통합 (h={h})", cc, "absAF", "iqr", log, cluster=True, boot=False))
        for lab, sub in [("2023", c[c["y2023"] == 1]), ("2024~26", c[c["y2023"] == 0])]:
            rows.append(spec(f"R6 {lab} (h={h})", sub, "absAF", "iqr", log, boot=False))
        d = c.dropna(subset=["absAF", "iqr"])
        fi = smf.ols("absAF ~ iqr * y2023", d).fit(cov_type="HC1")
        rows.append({"spec": f"R6 상호작용 IQR×2023 (h={h})", "y": "absAF", "x": "iqr:y2023", "n": len(d),
                     "beta": fi.params["iqr:y2023"], "se_hc1": fi.bse["iqr:y2023"], "t": fi.tvalues["iqr:y2023"],
                     "p_hc1": fi.pvalues["iqr:y2023"], "r2": fi.rsquared})
        log.info("  R6 상호작용 IQR×2023 (h=%d)           β=%7.3f HC1 se=%.3f p=%.4f", h, fi.params["iqr:y2023"],
                 fi.bse["iqr:y2023"], fi.pvalues["iqr:y2023"])
        log.info("  참고: |A−F| 표준편차 2023 %.3f / 2024~26 %.3f, IQR 평균 2023 %.3f / 2024~26 %.3f", d.loc[d.y2023 == 1, "absAF"].std(),
                 d.loc[d.y2023 == 0, "absAF"].std(), d.loc[d.y2023 == 1, "iqr"].mean(), d.loc[d.y2023 == 0, "iqr"].mean())
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "replication_goel2026.csv", index=False)
    log.info("저장 outputs/tables/replication_goel2026.csv")


if __name__ == "__main__":
    main()
