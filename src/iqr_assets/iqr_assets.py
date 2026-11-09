"""Kalshi 내재 IQR과 자산가격 반응 (처음부터 다시, 2026-11-09 설계 고정 후 실행).

질문: 발표 전 Kalshi가 매긴 결과 분포의 폭(IQR)이 클수록 발표 순간 자산가격이 크게 움직이나
      (Goel 2026이 후속 과제로 남긴 "내재 분산과 고빈도 자산가격 반응의 연결")

1단계 (이 파일, 재수집 없는 자료): 2023-01 이후, IQR = 발표 전날 Kalshi 분포
2단계 (1단계가 유의하지 않으면): 2022년부터 재수집 + 발표 직전 60분 IQR

변수
  IQR      Kalshi 분포의 Q75 − Q25, 원단위(%p, 고용은 명). 구간 안 균등분포 보간 (fixers26-1research 이벤트 패널 q25·q75)
  |r_win|  발표(8:30 ET)를 포함한 8:20→8:50 선물 수익률 절댓값(bp)  ※
  ΔMOVE    발표일 MOVE − 직전 거래일 MOVE

주 검정 (하나)
  헤드라인 CPI, 발표 전날 IQR:  |r_win(ZT, 2년물 국채선물)| = α + β·IQR
  OLS + HC1, wild-bootstrap-t 95% CI(Rademacher 5000회), 순열 p(10,000회), 양측
  (추론 방식은 Goel 2026 Table 3과 같음. 헤드라인 CPI를 주 표본으로 둔 것도 같음)
  ※ 같은 데이터로 비슷한 회귀를 이미 본 적 있음(가설 D·탐색 회귀) → 완전한 사전 등록은 아님

보조 검정 (한 묶음으로 Benjamini–Hochberg 보정)
  S1  ZN(10년물) |r_win|
  S2  ΔMOVE (부호 있음)
  S3  |ΔMOVE|
  S4  ZT, 네 지표 통합(CPI·근원 CPI·고용·실업률; GDP는 표본이 작아 제외, Goel도 제외). IQR 지표 안 표준화, 발표일 군집 SE
  S5  ZT, 서프라이즈 크기 |A−F| 통제 (IQR이 실현된 서프라이즈 크기 말고도 정보가 있나)
  S6  ZT, 발표 7일 전 IQR
  S7  ES(S&P500 선물), S8 SR3(SOFR 선물) — data/raw/glbx_extra_ohlcv1m.dbn.zst가 있을 때만

입력: data/input/event_panel.parquet, data/event_windows.csv (src/hyp/00_event_windows.py), data/input/move_daily.csv
출력: outputs/tables/iqr_assets.csv, outputs/figures/iqr_assets_main.png
실행: python src/hyp/00_event_windows.py → python src/iqr_assets/iqr_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, FIGURES, TABLES, get_logger  # noqa: E402

RNG = np.random.default_rng(20261109)
N_BOOT, N_PERM = 5000, 10000
POOL = ["KXCPI", "KXCPICORE", "KXPAYROLLS", "KXU3"]


def wild_t_ci(y, x):
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit(cov_type="HC1")
    b, se, res, yhat = fit.params[1], fit.bse[1], fit.resid, fit.fittedvalues
    ts = np.empty(N_BOOT)
    for i in range(N_BOOT):
        f = sm.OLS(yhat + res * RNG.choice([-1.0, 1.0], len(y)), X).fit(cov_type="HC1")
        ts[i] = (f.params[1] - b) / f.bse[1]
    lo, hi = np.quantile(ts, [0.975, 0.025])
    return b - lo * se, b - hi * se


def perm_p(y, x):
    b = np.polyfit(x, y, 1)[0]
    bs = np.array([np.polyfit(RNG.permutation(x), y, 1)[0] for _ in range(N_PERM)])
    return float((np.abs(bs) >= abs(b)).mean())


def load(h: int) -> pd.DataFrame:
    p = pd.read_parquet(DATA / "input" / "event_panel.parquet")
    p = p[p["h"] == h].copy()
    p["release_date_actual"] = pd.to_datetime(p["release_date_actual"])
    p["iqr"] = p["q75"] - p["q25"]
    p["absAF"] = (p["A"] - p["F"]).abs()
    p["iqr_z"] = p.groupby("series")["iqr"].transform(lambda s: (s - s.mean()) / s.std())
    w = pd.read_csv(DATA / "event_windows.csv", parse_dates=["date"])
    p = p.merge(w, left_on="release_date_actual", right_on="date", how="left")
    mv = pd.read_csv(DATA / "input" / "move_daily.csv", parse_dates=["date"]).dropna().sort_values("date")
    mv["dMOVE"] = mv["value"].diff()
    p = p.merge(mv[["date", "dMOVE"]], left_on="release_date_actual", right_on="date", how="left", suffixes=("", "_mv"))
    for c in [c for c in p.columns if c.endswith("_win")]:
        p[f"abs_{c}"] = p[c].abs()
    p["absdMOVE"] = p["dMOVE"].abs()
    p["cluster"] = p["release_date_actual"].dt.strftime("%Y-%m-%d")
    return p


def ols(name, d, y, x, log, controls="", cluster=False, boot=False):
    d = d.dropna(subset=[y, x] + [c.strip() for c in controls.split("+") if c.strip() and "C(" not in c])
    f = f"{y} ~ {x}" + (f" + {controls}" if controls else "")
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": d["cluster"]}} if cluster else {"cov_type": "HC1"}
    fit = smf.ols(f, d).fit(**kw)
    r = {"test": name, "y": y, "x": x, "controls": controls, "n": len(d), "beta": fit.params[x], "se": fit.bse[x],
         "t": fit.tvalues[x], "p": fit.pvalues[x], "r2": fit.rsquared}
    if boot:
        r["wild_ci_lo"], r["wild_ci_hi"] = wild_t_ci(d[y].to_numpy(float), d[x].to_numpy(float))
        r["perm_p"] = perm_p(d[y].to_numpy(float), d[x].to_numpy(float))
    log.info("  %-40s n=%3d β=%9.3f se=%8.3f t=%5.2f p=%.4f R²=%.3f%s", name, r["n"], r["beta"], r["se"], r["t"], r["p"], r["r2"],
             f" | wild CI [{r['wild_ci_lo']:.2f}, {r['wild_ci_hi']:.2f}] 순열 p={r['perm_p']:.4f}" if boot else "")
    return r, d


def main() -> None:
    log = get_logger("iqr_assets")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    e1, e7 = load(1), load(7)
    cpi1 = e1[e1["series"] == "KXCPI"]
    log.info("헤드라인 CPI 발표 %d건 (%s ~ %s), IQR 있는 것 %d", len(cpi1), cpi1["release_date_actual"].min().date(),
             cpi1["release_date_actual"].max().date(), int(cpi1["iqr"].notna().sum()))

    log.info("[1/3] 주 검정")
    main_r, dmain = ols("MAIN 헤드라인CPI |r_win ZT| ~ IQR (h=1)", cpi1, "abs_ZT_win", "iqr", log, boot=True)
    main_r["family"] = "main"

    log.info("[2/3] 보조 검정")
    sec = []
    sec.append(ols("S1 |r_win ZN| ~ IQR", cpi1, "abs_ZN_win", "iqr", log)[0])
    sec.append(ols("S2 ΔMOVE ~ IQR", cpi1, "dMOVE", "iqr", log)[0])
    sec.append(ols("S3 |ΔMOVE| ~ IQR", cpi1, "absdMOVE", "iqr", log)[0])
    sec.append(ols("S4 ZT 네 지표 통합 (IQR 표준화)", e1[e1["series"].isin(POOL)], "abs_ZT_win", "iqr_z", log,
                   controls="C(series)", cluster=True)[0])
    sec.append(ols("S5 ZT + |A−F| 통제", cpi1, "abs_ZT_win", "iqr", log, controls="absAF")[0])
    sec.append(ols("S6 ZT 발표 7일 전 IQR", e7[e7["series"] == "KXCPI"], "abs_ZT_win", "iqr", log)[0])
    for k, t in [("S7", "ES"), ("S8", "SR3")]:
        if f"abs_{t}_win" in cpi1.columns and cpi1[f"abs_{t}_win"].notna().any():
            sec.append(ols(f"{k} |r_win {t}| ~ IQR", cpi1, f"abs_{t}_win", "iqr", log)[0])
        else:
            log.info("  %s %s 자료 없음 → 건너뜀", k, t)
    sec = pd.DataFrame(sec)
    sec["p_bh"] = multipletests(sec["p"], method="fdr_bh")[1]
    sec["family"] = "secondary"
    out = pd.concat([pd.DataFrame([main_r]), sec], ignore_index=True)
    out.to_csv(TABLES / "iqr_assets.csv", index=False)

    log.info("[3/3] 요약")
    log.info("  주 검정: β=%.1f bp/%%p, p=%.4f, wild CI [%.1f, %.1f], 순열 p=%.4f → %s", main_r["beta"], main_r["p"],
             main_r["wild_ci_lo"], main_r["wild_ci_hi"], main_r["perm_p"],
             "유의(5%)" if main_r["perm_p"] < 0.05 and not (main_r["wild_ci_lo"] < 0 < main_r["wild_ci_hi"]) else "유의하지 않음")
    for r in sec.itertuples():
        log.info("  %-36s β=%9.3f p=%.4f BH p=%.4f", r.test, r.beta, r.p, r.p_bh)

    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.scatter(dmain["iqr"], dmain["abs_ZT_win"], s=22, alpha=.8)
    xx = np.linspace(dmain["iqr"].min(), dmain["iqr"].max(), 50)
    b, a = np.polyfit(dmain["iqr"], dmain["abs_ZT_win"], 1)
    ax.plot(xx, a + b * xx, color="#E07A28")
    ax.set(xlabel="Kalshi implied IQR, day before (pp)", ylabel="|2Y futures return| 8:20–8:50 ET (bp)",
           title=f"Headline CPI (n={len(dmain)}), β={b:.0f}, perm p={main_r['perm_p']:.3f}")
    fig.savefig(FIGURES / "iqr_assets_main.png", dpi=180)
    log.info("저장 outputs/tables/iqr_assets.csv, outputs/figures/iqr_assets_main.png")


if __name__ == "__main__":
    main()
