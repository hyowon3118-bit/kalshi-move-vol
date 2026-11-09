"""3단계: Kalshi 분포의 불확실성(표준편차·엔트로피)과 MOVE의 상관.

가설: 예측시장 분포의 표준편차/엔트로피가 MOVE와 양(+)의 상관을 갖는다.

분석 (지표별, 측정치 m ∈ {sd, H, H_norm})
  A 수준       corr(m_t, MOVE_t). Pearson·Spearman. p값은 z(MOVE) ~ z(m) OLS의 Newey–West(HAC) 표준오차
               둘 다 추세가 있으면 상관이 부풀려지므로(가성 상관) 참고용 ※
  B 수준+통제  MOVE ~ m + 만기까지 남은 일수 더미, HAC.  분포가 발표 직전 저절로 좁아지는 효과 제거 ※
  C 변화 (주)  corr(Δm_t, ΔMOVE_t), 같은 이벤트 안 변화만. HAC p값.  추세 영향이 없어 주 결과로 씀 ※
               강건성: Kalshi 거래가 있었던 날만 / 다섯 지표 통합(지표별 표준화, 날짜 군집 표준오차)
  D 이벤트 단위 만기 h일 전 분포의 m과 그날 MOVE 수준의 상관 (이벤트당 1개 관측, 남은 일수 고정) ※
               통제: 기준선 간격 log(step), 시간 추세 → 편상관 ※
               (고용은 기준선 간격이 10만→1만 명으로 바뀌는 동안 MOVE도 하락해서, 통제 없이는 가성 상관 가능)

HAC 시차 L = floor(4·(T/100)^(2/9)) (Newey–West 1994의 관행 규칙)
다중검정: 주 결과(C, D)를 각 묶음 안에서 Benjamini–Hochberg 보정

입력: data/daily_panel.parquet
출력: outputs/tables/corr_level.csv, corr_change.csv, corr_event.csv, outputs/figures/*.png
실행: python src/03_correlations.py
"""
from __future__ import annotations

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
    d = g[["MOVE", m, "days_to_expiry"]].dropna().copy()
    if len(d) < 30:
        return {}
    d["mz"] = z(d[m])
    fit = smf.ols("MOVE ~ mz + C(days_to_expiry)", data=d).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag(len(d))})
    return {"coef_ctrl": fit.params["mz"], "se_ctrl": fit.bse["mz"], "p_ctrl": fit.pvalues["mz"]}


def event_level(panel: pd.DataFrame, h: int, max_stale: int) -> pd.DataFrame:
    """이벤트마다 만기 h일 전 이하에서 가장 가까운 관측 1개."""
    d = panel[panel["days_to_expiry"] >= h].sort_values("days_to_expiry")
    d = d.groupby(["series", "contract_preamble"], as_index=False).first()
    return d[d["days_to_expiry"] - h <= max_stale]


def partial_corr(g: pd.DataFrame, m: str) -> dict:
    """MOVE와 m에서 log(step)·시간 추세를 각각 제거한 잔차끼리의 상관."""
    d = g[[m, "MOVE", "step", "date"]].dropna()
    d = d[d["step"] > 0]
    if len(d) < 10:
        return {}
    C = sm.add_constant(pd.DataFrame({"lstep": np.log(d["step"]),
                                      "t": (d["date"] - d["date"].min()).dt.days / 365.25}))
    if C["lstep"].std() == 0:
        C = C.drop(columns="lstep")
    rx = sm.OLS(d[m], C).fit().resid
    ry = sm.OLS(d["MOVE"], C).fit().resid
    r, _ = stats.pearsonr(rx, ry)
    df = len(d) - C.shape[1] - 1
    t = r * np.sqrt(df / (1 - r ** 2))
    return {"partial_r": r, "p_partial": 2 * stats.t.sf(abs(t), df), "controls": "+".join(c for c in C.columns if c != "const")}


def add_bh(df: pd.DataFrame, pcol: str) -> pd.DataFrame:
    ok = df[pcol].notna()
    df.loc[ok, "p_bh"] = multipletests(df.loc[ok, pcol], method="fdr_bh")[1]
    return df


def main() -> None:
    cfg = load_config()
    log = get_logger("03_correlations")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA / "daily_panel.parquet")
    log.info("패널 %d행, 지표 %s", len(panel), sorted(panel["series"].unique()))

    log.info("[1/4] A·B 수준 상관 (참고용)")
    rows = []
    for s, g in panel.groupby("series"):
        for m in MEASURES:
            r = {"series": s, "measure": m, **corr_row(g[m], g["MOVE"]), **level_with_controls(g, m)}
            rows.append(r)
            log.info("  %s %-6s n=%d  r=%.2f (HAC p=%.3f) | 남은일수 통제 계수=%.2f (p=%.3f)",
                     s, m, r["n"], r.get("pearson_r", np.nan), r.get("p_hac", np.nan),
                     r.get("coef_ctrl", np.nan), r.get("p_ctrl", np.nan))
    level = pd.DataFrame(rows)
    level.to_csv(TABLES / "corr_level.csv", index=False)

    log.info("[2/4] C 변화 상관 (주 결과)")
    rows = []
    for sample, d in [("all_days", panel), ("traded_only", panel[panel["traded"]])]:
        for s, g in d.groupby("series"):
            for m in MEASURES:
                r = {"sample": sample, "series": s, "measure": m, **corr_row(g[f"d_{m}"], g["dMOVE"])}
                rows.append(r)
                log.info("  [%s] %s %-6s n=%d  r=%.3f (HAC p=%.3f)", sample, s, m, r["n"],
                         r.get("pearson_r", np.nan), r.get("p_hac", np.nan))
        # 다섯 지표 통합: 지표별로 표준화한 Δm, 같은 날짜 관측은 군집
        for m in MEASURES:
            st = d.dropna(subset=[f"d_{m}", "dMOVE"]).copy()
            st["x"] = st.groupby("series")[f"d_{m}"].transform(z)
            st["y"] = z(st["dMOVE"])
            fit = smf.ols("y ~ x + C(series)", data=st).fit(cov_type="cluster", cov_kwds={"groups": st["date"].astype(str)})
            rows.append({"sample": sample, "series": "pooled", "measure": m, "n": len(st),
                         "beta_z": fit.params["x"], "se_hac": fit.bse["x"], "p_hac": fit.pvalues["x"],
                         "pearson_r": np.corrcoef(st["x"], st["y"])[0, 1]})
            log.info("  [%s] 통합 %-6s n=%d  β=%.3f (날짜 군집 p=%.3f)", sample, m, len(st),
                     fit.params["x"], fit.pvalues["x"])
    change = pd.DataFrame(rows)
    change = pd.concat([add_bh(x.copy(), "p_hac") for _, x in change.groupby("sample")], ignore_index=True)
    change.to_csv(TABLES / "corr_change.csv", index=False)

    log.info("[3/4] D 이벤트 단위 (만기 h일 전 분포 vs 그날 MOVE)")
    rows = []
    for h in cfg["event_level"]["horizons"]:
        ev = event_level(panel, h, cfg["event_level"]["max_stale_days"])
        for s, g in ev.groupby("series"):
            for m in MEASURES:
                r = {"h": h, "series": s, "measure": m, **corr_row(g[m], g["MOVE"], hac=False), **partial_corr(g, m)}
                rows.append(r)
                log.info("  h=%d %s %-6s 이벤트 %d  r=%.2f (p=%.3f)  ρ=%.2f | 간격·추세 통제 편상관 %.2f (p=%.3f)",
                         h, s, m, r["n"], r.get("pearson_r", np.nan), r.get("p_pearson_iid", np.nan),
                         r.get("spearman_rho", np.nan), r.get("partial_r", np.nan), r.get("p_partial", np.nan))
    event = add_bh(pd.DataFrame(rows), "p_pearson_iid")
    ok = event["p_partial"].notna()
    event.loc[ok, "p_partial_bh"] = multipletests(event.loc[ok, "p_partial"], method="fdr_bh")[1]
    event.to_csv(TABLES / "corr_event.csv", index=False)

    log.info("[4/4] 그림")
    # 그림 1: 변화 상관 r과 95% 신뢰구간 (Fisher z 근사; 자기상관 무시라 참고용)
    c = change[(change["sample"] == "all_days") & (change["series"] != "pooled")].dropna(subset=["pearson_r"])
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    series = list(NAMES)
    width = 0.25
    for i, m in enumerate(MEASURES):
        sub = c[c["measure"] == m].set_index("series").reindex(series)
        zr, se = np.arctanh(sub["pearson_r"]), 1 / np.sqrt(sub["n"] - 3)
        lo, hi = np.tanh(zr - 1.96 * se), np.tanh(zr + 1.96 * se)
        x = np.arange(len(series)) + (i - 1) * width
        ax.bar(x, sub["pearson_r"], width, label=m, alpha=.85)
        ax.errorbar(x, sub["pearson_r"], yerr=[sub["pearson_r"] - lo, hi - sub["pearson_r"]], fmt="none", ecolor="#333", lw=1)
    ax.axhline(0, color="#777", lw=.8)
    ax.set_xticks(range(len(series)), [NAMES[s] for s in series])
    ax.set(ylabel="corr(Δmeasure, ΔMOVE)", title="Daily changes: Kalshi uncertainty vs MOVE")
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "change_correlations.png", dpi=180)
    plt.close(fig)

    # 그림 2: 만기 7일 전 분포 표준편차 vs 그날 MOVE (지표별)
    h = cfg["event_level"]["horizons"][0]
    ev = event_level(panel, h, cfg["event_level"]["max_stale_days"])
    fig, axes = plt.subplots(1, len(series), figsize=(16, 3.4), constrained_layout=True)
    for ax, s in zip(axes, series):
        g = ev[ev["series"] == s].dropna(subset=["sd", "MOVE"])
        ax.scatter(g["sd"], g["MOVE"], s=18, alpha=.8)
        if len(g) > 2:
            b, a = np.polyfit(g["sd"], g["MOVE"], 1)
            xx = np.linspace(g["sd"].min(), g["sd"].max(), 50)
            ax.plot(xx, a + b * xx, color="#E07A28")
            ax.set_title(f"{NAMES[s]} (n={len(g)}, r={g['sd'].corr(g['MOVE']):.2f})", fontsize=9)
        ax.set_xlabel("Kalshi distribution SD")
    axes[0].set_ylabel("MOVE")
    fig.suptitle(f"Event level, {h} days before expiry")
    fig.savefig(FIGURES / "event_sd_vs_move.png", dpi=180)
    plt.close(fig)
    log.info("저장 outputs/tables/corr_level.csv, corr_change.csv, corr_event.csv | 그림 2개")


if __name__ == "__main__":
    main()
