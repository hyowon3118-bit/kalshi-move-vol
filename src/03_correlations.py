"""3단계: Kalshi 분포의 불확실성(표준편차·엔트로피)과 채권 변동성의 상관.

대상 Y (--target)
  MOVE   ICE MOVE 지수 (미국채 옵션 내재변동성, 일별 종가)            [기본]
  ZT     2년물 국채선물 5분 실현변동성의 연율화 값(%)  (05_treasury_rv.py 출력)
  ZN     10년물 국채선물, 같은 방식

가설: 예측시장 분포의 표준편차/엔트로피가 채권 변동성과 양(+)의 상관을 갖는다.

분석 (지표별, 측정치 m ∈ {sd, H, H_norm})
  A 수준       corr(m_t, Y_t). Pearson·Spearman. p값은 z(Y) ~ z(m) OLS의 Newey–West(HAC) 표준오차
               둘 다 추세가 있으면 상관이 부풀려지므로(가성 상관) 참고용 ※
  B 수준+통제  Y ~ m + 발표까지 남은 일수 더미, HAC.  분포가 발표 직전 저절로 좁아지는 효과 제거 ※
  C 변화 (주)  corr(Δm_t, ΔY_t), 같은 이벤트 안 변화만. HAC p값.  추세 영향이 없어 주 결과로 씀 ※
               강건성: Kalshi 거래가 있었던 날만 / 다섯 지표 통합(지표별 표준화, 날짜 군집 표준오차)
  D 이벤트 단위 발표 h일 전 분포의 m과 그날 Y 수준의 상관 (이벤트당 1개 관측, 남은 일수 고정) ※
               통제: 기준선 간격 log(step), 시간 추세 → 편상관 ※
               (고용은 기준선 간격이 10만→1만 명으로 바뀌는 동안 MOVE도 하락해서, 통제 없이는 가성 상관 가능)
  E 발표일 변동성 (ZT·ZN만) 발표 h일 전 분포의 m이 발표 당일 실현변동성을 예고하나 ※
               편상관 통제: 발표 전 평상시 변동성(직전 pre_days 거래일 평균, h일 전 시점 기준), log(step), 시간 추세
               "분포가 넓을수록 발표일에 금리가 크게 움직인다"를 직접 보는 분석

HAC 시차 L = floor(4·(T/100)^(2/9)) (Newey–West 1994의 관행 규칙)
다중검정: 주 결과(C, D, E)를 각 묶음 안에서 Benjamini–Hochberg 보정

입력: data/daily_panel.parquet (+ data/treasury_rv_daily.csv)
출력: outputs/tables/{target}_corr_level.csv, _change.csv, _event.csv (+ _release.csv), outputs/figures/{target}_*.png
실행: python src/03_correlations.py --target MOVE | ZT | ZN
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import DATA, FIGURES, TABLES, get_logger, load_config

MEASURES = ["sd", "H", "H_norm"]
NAMES = {"KXCPI": "Headline CPI", "KXCPICORE": "Core CPI", "KXGDP": "GDP",
         "KXPAYROLLS": "Payrolls", "KXU3": "Unemployment"}
LABEL = {"MOVE": "MOVE", "ZT": "2Y futures realized vol (%)", "ZN": "10Y futures realized vol (%)"}


def hac_lag(n: int) -> int:
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def corr_row(x: pd.Series, y: pd.Series, hac: bool = True) -> dict:
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 10 or d.iloc[:, 0].std() == 0:
        return {"n": n}
    r, p_r = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
    rho, p_rho = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    out = {"n": n, "pearson_r": r, "p_pearson_iid": p_r, "spearman_rho": rho, "p_spearman_iid": p_rho}
    if hac:
        X = sm.add_constant(z(d.iloc[:, 0]).rename("x"))
        fit = sm.OLS(z(d.iloc[:, 1]), X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag(n)})
        out.update({"beta_z": fit.params["x"], "se_hac": fit.bse["x"], "p_hac": fit.pvalues["x"],
                    "hac_lags": hac_lag(n)})
    return out


def level_with_controls(g: pd.DataFrame, m: str) -> dict:
    d = g[["Y", m, "days_to_release"]].dropna().copy()
    if len(d) < 30:
        return {}
    d["mz"] = z(d[m])
    fit = smf.ols("Y ~ mz + C(days_to_release)", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag(len(d))})
    return {"coef_ctrl": fit.params["mz"], "se_ctrl": fit.bse["mz"], "p_ctrl": fit.pvalues["mz"]}


def event_level(panel: pd.DataFrame, h: int, max_stale: int) -> pd.DataFrame:
    """이벤트마다 발표 h일 전 이하에서 가장 가까운 관측 1개."""
    d = panel[panel["days_to_release"] >= h].sort_values("days_to_release")
    d = d.groupby(["series", "contract_preamble"], as_index=False).first()
    return d[d["days_to_release"] - h <= max_stale]


def partial_corr(g: pd.DataFrame, m: str, y: str, extra: list[str] | None = None) -> dict:
    """y와 m에서 log(step)·시간 추세(+extra)를 각각 제거한 잔차끼리의 상관."""
    extra = extra or []
    d = g[[m, y, "step", "date", *extra]].dropna()
    d = d[d["step"] > 0]
    if len(d) < 10:
        return {}
    C = pd.DataFrame({"lstep": np.log(d["step"]), "t": (d["date"] - d["date"].min()).dt.days / 365.25})
    for e in extra:
        C[e] = d[e]
    C = sm.add_constant(C.loc[:, C.std() > 0])
    rx = sm.OLS(d[m], C).fit().resid
    ry = sm.OLS(d[y], C).fit().resid
    r, _ = stats.pearsonr(rx, ry)
    df = len(d) - C.shape[1] - 1
    t = r * np.sqrt(df / (1 - r ** 2))
    return {"partial_r": r, "p_partial": 2 * stats.t.sf(abs(t), df),
            "controls": "+".join(c for c in C.columns if c != "const")}


def add_bh(df: pd.DataFrame, pcol: str, out: str = "p_bh") -> pd.DataFrame:
    ok = df[pcol].notna()
    df.loc[ok, out] = multipletests(df.loc[ok, pcol], method="fdr_bh")[1]
    return df


def load_target(target: str, panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """panel에 Y(수준), dY(직전 거래일 대비 변화) 추가. RV 대상이면 일별 RV 표도 반환."""
    if target == "MOVE":
        return panel.rename(columns={"MOVE": "Y", "dMOVE": "dY"}), None
    rv = pd.read_csv(DATA / "treasury_rv_daily.csv", parse_dates=["date"]).sort_values("date")
    rv = rv[["date", f"vol_{target}"]].rename(columns={f"vol_{target}": "Y"}).dropna()
    rv["dY"] = rv["Y"].diff()
    # 발표 전 평상시 변동성: 그날 포함 직전 pre_days 거래일 평균 (E에서 h일 전 시점 값을 씀)
    rv["Y_pre"] = rv["Y"].rolling(cfg["rv"]["pre_days"], min_periods=3).mean()
    p = panel.drop(columns=["MOVE", "dMOVE"]).merge(rv, on="date", how="inner")
    return p, rv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="MOVE", choices=["MOVE", "ZT", "ZN"])
    target = ap.parse_args().target
    cfg = load_config()
    log = get_logger(f"03_correlations_{target}")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel, rv = load_target(target, pd.read_parquet(DATA / "daily_panel.parquet"), cfg)
    # 이벤트가 바뀌는 날 Δm은 이미 결측. RV 대상이면 d_m을 RV 거래일 기준으로 다시 맞출 필요 없음(같은 날짜 결합)
    log.info("대상 %s | 패널 %d행, 지표 %s", target, len(panel), sorted(panel["series"].unique()))
    tag = target

    log.info("[1/5] A·B 수준 상관 (참고용)")
    rows = []
    for s, g in panel.groupby("series"):
        for m in MEASURES:
            r = {"series": s, "measure": m, **corr_row(g[m], g["Y"]), **level_with_controls(g, m)}
            rows.append(r)
            log.info("  %s %-6s n=%d  r=%.2f (HAC p=%.3f) | 남은일수 통제 계수=%.2f (p=%.3f)",
                     s, m, r["n"], r.get("pearson_r", np.nan), r.get("p_hac", np.nan),
                     r.get("coef_ctrl", np.nan), r.get("p_ctrl", np.nan))
    pd.DataFrame(rows).to_csv(TABLES / f"{tag}_corr_level.csv", index=False)

    log.info("[2/5] C 변화 상관 (주 결과)")
    rows = []
    for sample, d in [("all_days", panel), ("traded_only", panel[panel["traded"]])]:
        for s, g in d.groupby("series"):
            for m in MEASURES:
                r = {"sample": sample, "series": s, "measure": m, **corr_row(g[f"d_{m}"], g["dY"])}
                rows.append(r)
                log.info("  [%s] %s %-6s n=%d  r=%.3f (HAC p=%.3f)", sample, s, m, r["n"],
                         r.get("pearson_r", np.nan), r.get("p_hac", np.nan))
        for m in MEASURES:  # 다섯 지표 통합: 지표별로 표준화한 Δm, 같은 날짜 관측은 군집
            st = d.dropna(subset=[f"d_{m}", "dY"]).copy()
            st["x"] = st.groupby("series")[f"d_{m}"].transform(z)
            st["y"] = z(st["dY"])
            fit = smf.ols("y ~ x + C(series)", data=st).fit(cov_type="cluster", cov_kwds={"groups": st["date"].astype(str)})
            rows.append({"sample": sample, "series": "pooled", "measure": m, "n": len(st),
                         "beta_z": fit.params["x"], "se_hac": fit.bse["x"], "p_hac": fit.pvalues["x"],
                         "pearson_r": np.corrcoef(st["x"], st["y"])[0, 1]})
            log.info("  [%s] 통합 %-6s n=%d  β=%.3f (날짜 군집 p=%.3f)", sample, m, len(st),
                     fit.params["x"], fit.pvalues["x"])
    change = pd.DataFrame(rows)
    change = pd.concat([add_bh(x.copy(), "p_hac") for _, x in change.groupby("sample")], ignore_index=True)
    change.to_csv(TABLES / f"{tag}_corr_change.csv", index=False)

    log.info("[3/5] D 이벤트 단위 (발표 h일 전 분포 vs 그날 %s)", target)
    rows = []
    for h in cfg["event_level"]["horizons"]:
        ev = event_level(panel, h, cfg["event_level"]["max_stale_days"])
        for s, g in ev.groupby("series"):
            for m in MEASURES:
                r = {"h": h, "series": s, "measure": m, **corr_row(g[m], g["Y"], hac=False), **partial_corr(g, m, "Y")}
                rows.append(r)
                log.info("  h=%d %s %-6s 이벤트 %d  r=%.2f (p=%.3f)  ρ=%.2f | 간격·추세 통제 편상관 %.2f (p=%.3f)",
                         h, s, m, r["n"], r.get("pearson_r", np.nan), r.get("p_pearson_iid", np.nan),
                         r.get("spearman_rho", np.nan), r.get("partial_r", np.nan), r.get("p_partial", np.nan))
    event = add_bh(pd.DataFrame(rows), "p_pearson_iid")
    event = add_bh(event, "p_partial", "p_partial_bh")
    event.to_csv(TABLES / f"{tag}_corr_event.csv", index=False)

    rel = None
    if rv is not None:
        log.info("[4/5] E 발표일 실현변동성 (발표 h일 전 분포 → 발표 당일 %s 변동성)", target)
        rv_day = rv.set_index("date")["Y"]
        rows = []
        for h in cfg["event_level"]["horizons"]:
            ev = event_level(panel, h, cfg["event_level"]["max_stale_days"]).copy()
            ev["Y_release"] = ev["release_date"].map(rv_day)
            for s, g in ev.groupby("series"):
                for m in MEASURES:
                    r = {"h": h, "series": s, "measure": m, **corr_row(g[m], g["Y_release"], hac=False),
                         **partial_corr(g, m, "Y_release", ["Y_pre"])}
                    rows.append(r)
                    log.info("  h=%d %s %-6s 이벤트 %d  r=%.2f (p=%.3f) | 평상시 변동성·간격·추세 통제 편상관 %.2f (p=%.3f)",
                             h, s, m, r["n"], r.get("pearson_r", np.nan), r.get("p_pearson_iid", np.nan),
                             r.get("partial_r", np.nan), r.get("p_partial", np.nan))
        rel = add_bh(pd.DataFrame(rows), "p_partial", "p_partial_bh")
        rel.to_csv(TABLES / f"{tag}_corr_release.csv", index=False)
    else:
        log.info("[4/5] E는 실현변동성 대상(ZT·ZN)에서만")

    log.info("[5/5] 그림")
    series = list(NAMES)
    c = change[(change["sample"] == "all_days") & (change["series"] != "pooled")].dropna(subset=["pearson_r"])
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    width = 0.25
    for i, m in enumerate(MEASURES):  # 95% CI는 Fisher z 근사(자기상관 무시라 참고용)
        sub = c[c["measure"] == m].set_index("series").reindex(series)
        zr, se = np.arctanh(sub["pearson_r"]), 1 / np.sqrt(sub["n"] - 3)
        lo, hi = np.tanh(zr - 1.96 * se), np.tanh(zr + 1.96 * se)
        x = np.arange(len(series)) + (i - 1) * width
        ax.bar(x, sub["pearson_r"], width, label=m, alpha=.85)
        ax.errorbar(x, sub["pearson_r"], yerr=[sub["pearson_r"] - lo, hi - sub["pearson_r"]], fmt="none", ecolor="#333", lw=1)
    ax.axhline(0, color="#777", lw=.8)
    ax.set_xticks(range(len(series)), [NAMES[s] for s in series])
    ax.set(ylabel=f"corr(Δmeasure, Δ{target})", title=f"Daily changes: Kalshi uncertainty vs {LABEL[target]}")
    ax.legend(frameon=False)
    fig.savefig(FIGURES / f"{tag}_change_correlations.png", dpi=180)
    plt.close(fig)

    # 그림 2: 발표 h일 전 분포 표준편차 vs (MOVE: 그날 수준 / RV: 발표 당일 실현변동성)
    h = cfg["event_level"]["horizons"][0]
    ev = event_level(panel, h, cfg["event_level"]["max_stale_days"]).copy()
    ycol, ylab = "Y", LABEL[target]
    if rv is not None:
        ev["Y_release"] = ev["release_date"].map(rv.set_index("date")["Y"])
        ycol, ylab = "Y_release", f"Release-day {LABEL[target]}"
    fig, axes = plt.subplots(1, len(series), figsize=(16, 3.4), constrained_layout=True)
    for ax, s in zip(axes, series):
        g = ev[ev["series"] == s].dropna(subset=["sd", ycol])
        ax.scatter(g["sd"], g[ycol], s=18, alpha=.8)
        if len(g) > 2:
            b, a = np.polyfit(g["sd"], g[ycol], 1)
            xx = np.linspace(g["sd"].min(), g["sd"].max(), 50)
            ax.plot(xx, a + b * xx, color="#E07A28")
            ax.set_title(f"{NAMES[s]} (n={len(g)}, r={g['sd'].corr(g[ycol]):.2f})", fontsize=9)
        ax.set_xlabel("Kalshi distribution SD")
    axes[0].set_ylabel(ylab)
    fig.suptitle(f"Event level, {h} days before release")
    fig.savefig(FIGURES / f"{tag}_event_sd.png", dpi=180)
    plt.close(fig)
    log.info("저장 outputs/tables/%s_corr_*.csv | 그림 2개", tag)


if __name__ == "__main__":
    main()
