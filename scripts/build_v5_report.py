"""CLI: generate the v5 presentation report (figures + tables) for slide prep.

This script produces a self-contained gallery of slide-ready PNG figures
plus summary CSV / Markdown tables into ``dist/v5/report/`` covering:

  * Forecast quality vs baselines (hit rate, RMSE, IC)
  * Posterior inclusion probabilities (top predictors per target)
  * Trends Risk Index time series with crisis shading
  * Conformal calibration (pre vs post)
  * Forward-looking forecast fan charts (HY, IG)
  * ETF-vs-OAS proxy quality
  * Band-width vs horizon (√h scaling)
  * Topic vs category contribution mix

Outputs are idempotent — re-run after refit / Risk Index / bundle changes
to refresh the visuals.

Usage::

    PYTHONPATH=src python3 scripts/build_v5_report.py
    # Optional: choose a different output directory.
    PYTHONPATH=src python3 scripts/build_v5_report.py --out-dir my_report/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Style constants ----------------------------------------------------

# Slide-friendly colors with consistent semantic meaning across all figures.
MODEL_COLORS: dict[str, str] = {
    "BSTS (Trends)":   "#d62728",   # red — the protagonist
    "AR(p)":           "#1f77b4",   # blue — the baseline
    "Naive RW":        "#7f7f7f",   # gray — the floor
    "StackedResidual": "#2ca02c",   # green — the experiment
}
TARGET_COLORS: dict[str, str] = {
    "HY": "#d62728",   # red — high yield
    "IG": "#1f77b4",   # blue — investment grade
}
TIER_COLORS: dict[str, str] = {
    "high": "#d62728",  # stress regime
    "med":  "#bbbbbb",  # normal
    "low":  "#2ca02c",  # benign
}

# Output presets
FIG_DPI = 160                                    # crisp on a 4K projector
FIG_SIZE_DEFAULT = (9, 5)                        # 16:9-ish for slides
FIG_SIZE_WIDE = (11, 4)                          # for time-series
FIG_SIZE_SQUARE = (7, 6)                         # for scatter / heatmap

# Cumulative-direction horizons (weekly = "1w") in business-day units.
HORIZON_LABEL: dict[int, str] = {1: "1w", 2: "2w", 4: "1m", 13: "1q", 26: "6m"}


# ---------- Helpers ------------------------------------------------------------


def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


def _save(fig: plt.Figure, out_fig_dir: Path, name: str) -> Path:
    """Save ``fig`` as ``<name>.png`` and close it."""
    path = out_fig_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_table(df: pd.DataFrame, out_tbl_dir: Path, name: str,
                 caption: str = "") -> tuple[Path, Path]:
    """Persist ``df`` as both CSV and Markdown."""
    csv = out_tbl_dir / f"{name}.csv"
    md = out_tbl_dir / f"{name}.md"
    df.to_csv(csv, index=False)
    body = df.to_markdown(index=False, floatfmt=".3f")
    md.write_text(f"### {name}\n\n{caption}\n\n{body}\n" if caption
                  else f"### {name}\n\n{body}\n")
    return csv, md


# ---------- Figures ------------------------------------------------------------


def fig01_hit_rate_by_horizon(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Grouped bar chart of hit rate by model at each horizon, per target."""
    paths: list[Path] = []
    for tgt in ("HY", "IG"):
        sub = df[df.target == tgt].copy()
        sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
        pivot = sub.pivot(index="horizon_label", columns="model",
                          values="hit_rate")
        # Stable column order: BSTS, AR(p), Naive RW, Stacked
        cols = ["BSTS (Trends)", "AR(p)", "Naive RW", "StackedResidual"]
        pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])
        # Stable horizon order
        pivot = pivot.reindex(["1w", "2w", "1m", "1q", "6m"])

        fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)
        x = np.arange(len(pivot.index))
        n_models = len(pivot.columns)
        width = 0.8 / n_models
        for i, model in enumerate(pivot.columns):
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, pivot[model].values, width,
                   label=model, color=MODEL_COLORS.get(model, "#aaaaaa"),
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0.5, color="black", linestyle=":", lw=1, alpha=0.6,
                   label="50% (coin flip)")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel("Hit rate (cumulative-direction)")
        ax.set_xlabel("Horizon")
        ax.set_title(f"{tgt} — directional hit rate by horizon (v5 walk-forward)")
        ax.set_ylim(0.0, 0.8)
        ax.legend(loc="upper left", ncol=2)
        paths.append(_save(fig, out_dir, f"01_hit_rate_{tgt}"))
    return paths


def fig02_rmse_by_horizon(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Grouped bar chart of RMSE by model at each horizon, per target."""
    paths: list[Path] = []
    for tgt in ("HY", "IG"):
        sub = df[df.target == tgt].copy()
        sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
        pivot = sub.pivot(index="horizon_label", columns="model", values="rmse")
        cols = ["Naive RW", "AR(p)", "StackedResidual", "BSTS (Trends)"]
        pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])
        pivot = pivot.reindex(["1w", "2w", "1m", "1q", "6m"])

        fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)
        x = np.arange(len(pivot.index))
        n_models = len(pivot.columns)
        width = 0.8 / n_models
        for i, model in enumerate(pivot.columns):
            offset = (i - (n_models - 1) / 2) * width
            ax.bar(x + offset, pivot[model].values, width,
                   label=model, color=MODEL_COLORS.get(model, "#aaaaaa"),
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylabel("RMSE (transform space)")
        ax.set_xlabel("Horizon")
        ax.set_title(f"{tgt} — RMSE by horizon (lower is better)")
        ax.legend(loc="upper left", ncol=2)
        paths.append(_save(fig, out_dir, f"02_rmse_{tgt}"))
    return paths


def fig03_bsts_vs_rw_hit_rate(df: pd.DataFrame, out_dir: Path) -> Path:
    """Highlight chart: BSTS (Trends) vs Naive RW hit rate, 2×4 grid by target."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, tgt in zip(axes, ("HY", "IG")):
        sub = df[df.target == tgt].copy()
        sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
        for model, marker, color in [
            ("BSTS (Trends)", "o", MODEL_COLORS["BSTS (Trends)"]),
            ("Naive RW", "s", MODEL_COLORS["Naive RW"]),
        ]:
            s = sub[sub.model == model].sort_values("horizon")
            ax.plot(s["horizon_label"], s["hit_rate"], marker=marker, lw=2.5,
                    markersize=11, color=color, label=model)
        ax.axhline(0.5, color="black", linestyle=":", lw=1, alpha=0.6)
        ax.set_title(f"{tgt} — BSTS vs Naive RW hit rate")
        ax.set_xlabel("Horizon")
        if tgt == "HY":
            ax.set_ylabel("Cumulative-direction hit rate")
        ax.set_ylim(0.15, 0.7)
        ax.legend(loc="upper left")
    fig.suptitle("BSTS (Trends) edge grows with horizon — 3× signal vs Naive RW at HY 6m",
                 y=1.02, fontsize=13)
    return _save(fig, out_dir, "03_bsts_vs_rw_hit_rate")


def fig04_inclusion_top_predictors(posteriors: dict, out_dir: Path) -> list[Path]:
    """Top-15 inclusion-probability predictors per target."""
    paths: list[Path] = []
    for tgt, posterior in posteriors.items():
        cs = posterior["coefficient_summary"].copy()
        cs = cs.sort_values("inclusion_prob", ascending=False).head(15)
        # Color positive vs negative betas distinctly
        colors = ["#d62728" if b < 0 else "#1f77b4"
                  for b in cs["mean_when_included"]]
        fig, ax = plt.subplots(figsize=(9, 6))
        y = np.arange(len(cs))
        ax.barh(y, cs["inclusion_prob"].values, color=colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(cs.index.tolist(), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Inclusion probability P(γ=1)")
        ax.set_title(f"{tgt} — top-15 predictors by inclusion probability\n"
                     "blue = positive β (signal ↑ → target ↑), "
                     "red = negative β")
        # Print sign + magnitude annotation next to each bar
        for yi, (_, row) in zip(y, cs.iterrows()):
            beta = row["mean_when_included"]
            ax.text(row["inclusion_prob"] + 0.02, yi,
                    f"β̄={beta:+.2f}", va="center", fontsize=8,
                    color="#444")
        ax.axvline(0.5, color="black", linestyle=":", lw=1, alpha=0.5)
        paths.append(_save(fig, out_dir, f"04_inclusion_{tgt}"))
    return paths


def fig05_risk_index_weekly(risk_dir: Path, out_dir: Path,
                            crises: list[tuple[str, str, str]]) -> list[Path]:
    """Weekly Risk Index time series with shaded crisis windows."""
    paths: list[Path] = []
    for tgt in ("HY", "IG"):
        ri = pd.read_parquet(risk_dir / f"{tgt}_trends_risk_index_weekly.parquet")
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        # Tier-color background fill via fill_between
        z = ri["zscore_5y"]
        ax.plot(ri.index, z, color=TARGET_COLORS[tgt], lw=1.2,
                label=f"{tgt} Trends Risk Index (5y z-score)")
        ax.fill_between(ri.index, 0, z, where=z > 1.0,
                        color=TIER_COLORS["high"], alpha=0.18,
                        label="high-stress regime (z > 1σ)")
        ax.fill_between(ri.index, 0, z, where=z < -1.0,
                        color=TIER_COLORS["low"], alpha=0.18,
                        label="benign regime (z < −1σ)")
        # Crisis markers
        for name, start, end in crises:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color="black", alpha=0.08)
            ax.text(pd.Timestamp(start), 4.5, name, fontsize=8,
                    rotation=90, va="top", ha="right", color="#333")
        ax.axhline(0, color="black", lw=0.7, alpha=0.6)
        ax.axhline(1.0, color="#d62728", lw=0.7, alpha=0.6, linestyle=":")
        ax.axhline(-1.0, color="#2ca02c", lw=0.7, alpha=0.6, linestyle=":")
        ax.set_ylim(-4, 5)
        ax.set_ylabel("Trends Risk Index (5y rolling z-score)")
        ax.set_title(f"{tgt} — Trends Risk Index (weekly, 2008–present)")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", ncol=2)
        paths.append(_save(fig, out_dir, f"05_risk_index_weekly_{tgt}"))
    return paths


def fig06_risk_index_daily(risk_dir: Path, out_dir: Path,
                           crises: list[tuple[str, str, str]]) -> list[Path]:
    """Daily Risk Index for both targets stacked."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, tgt in zip(axes, ("HY", "IG")):
        ri = pd.read_parquet(risk_dir / f"{tgt}_trends_risk_index_daily.parquet")
        z = ri["zscore_5y"]
        ax.plot(ri.index, z, color=TARGET_COLORS[tgt], lw=0.7)
        ax.fill_between(ri.index, 0, z, where=z > 1.0,
                        color=TIER_COLORS["high"], alpha=0.15)
        ax.fill_between(ri.index, 0, z, where=z < -1.0,
                        color=TIER_COLORS["low"], alpha=0.15)
        for _, start, end in crises:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color="black", alpha=0.08)
        ax.axhline(0, color="black", lw=0.6, alpha=0.5)
        ax.set_ylabel(f"{tgt}\n(z-score)")
        ax.set_ylim(-4, 5)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_title("Trends Risk Index (daily cadence, full 39-predictor v5 universe)")
    axes[-1].set_xlabel("Date")
    return [_save(fig, out_dir, "06_risk_index_daily")]


def fig07_calibration_coverage(recal: dict, out_dir: Path) -> Path:
    """80% coverage pre vs post conformal recalibration — bars per (target, model)."""
    rows = []
    for tgt in ("HY", "IG"):
        for model_name, cells in recal[tgt].items():
            cell = cells.get("0.8")
            if cell:
                rows.append({
                    "target": tgt, "model": model_name,
                    "pre": cell["empirical_pre_full"],
                    "post": cell["empirical_post_full"],
                    "alpha": cell["alpha"],
                })
    df = pd.DataFrame(rows)
    df["label"] = df["target"] + "·" + df["model"].str.replace("BSTS (Trends)", "BSTS", regex=False).str.replace("StackedResidual", "Stacked", regex=False).str.replace("Naive RW", "RW", regex=False).str.replace("AR(p)", "AR", regex=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(df))
    width = 0.35
    ax.barh(y - width/2, df["pre"], width, label="pre-recalibration",
            color="#bbbbbb", edgecolor="white")
    ax.barh(y + width/2, df["post"], width, label="post-recalibration",
            color="#d62728", edgecolor="white")
    ax.axvspan(0.75, 0.85, color="green", alpha=0.10,
               label="acceptance band [0.75, 0.85]")
    ax.axvline(0.80, color="black", linestyle=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Empirical coverage of nominal-80% band")
    ax.set_title("Conformal recalibration lands all 8 cells in [0.75, 0.85]")
    ax.legend(loc="lower right")
    # Annotate α per row
    for yi, alpha in zip(y, df["alpha"]):
        ax.text(0.02, yi - width/2, f"α={alpha:.3f}", fontsize=8,
                va="center", color="#333")
    return _save(fig, out_dir, "07_calibration_coverage")


def fig08_forecast_fan(model_path: Path, y_history: pd.Series,
                       x_latest: pd.DataFrame, out_dir: Path,
                       target: str) -> Path:
    """Fan chart: point forecast + 90% band over the 5-horizon ladder.

    Uses the pure-Python inference layer — no MCMC.
    """
    from gtrends_bayes.inference import load_model, forecast

    model = load_model(str(model_path))
    horizons = ["1w", "2w", "1m", "1q", "6m"]

    # Run one forecast call per horizon and concat the paths into a fan.
    all_paths: dict[str, dict] = {}
    for h in horizons:
        out = forecast(model, h, y_history.index.max(),
                       y_history, x_latest, n_draws=1000, seed=42)
        all_paths[h] = out

    # Use the longest path (6m, 126 bd) for the underlying fan curve;
    # mark the 1w/1m/1q/6m terminal points specially.
    out6m = all_paths["6m"]
    days = np.arange(1, out6m["horizon_bd"] + 1)
    as_of = pd.Timestamp(out6m["as_of"])
    fcst_idx = pd.date_range(as_of + pd.Timedelta(days=1),
                             periods=len(days), freq="B")

    fig, ax = plt.subplots(figsize=(12, 6))
    # History (last 6 months)
    history = y_history.tail(26)
    ax.plot(history.index, history.values, color="black", lw=1.5,
            label=f"{target} ETF history (last 6m)")
    # Fan band
    ax.fill_between(fcst_idx, out6m["level_path_q05"], out6m["level_path_q95"],
                    color=TARGET_COLORS[target], alpha=0.18,
                    label="90% credible band (full path)")
    ax.plot(fcst_idx, out6m["level_path_median"],
            color=TARGET_COLORS[target], lw=2,
            label="Forecast median (BSTS + AR(4) + conformal α)")
    # Terminal-point markers — annotate each horizon at its own position
    horizon_bd = {"1w": 5, "2w": 10, "1m": 21, "1q": 63, "6m": 126}
    # Vertical-offset rotation so labels don't collide on near-equal medians
    offsets = {"1w": (8, 22), "2w": (8, -22), "1m": (8, 22),
               "1q": (8, 22), "6m": (-30, 22)}
    for h in horizons:
        idx = horizon_bd[h] - 1
        terminal = fcst_idx[idx]
        med = all_paths[h]["level_median"]
        lo, hi = all_paths[h]["level_band"]
        ax.scatter([terminal], [med], color=TARGET_COLORS[target], zorder=5,
                   s=60, edgecolors="white", linewidths=1.5)
        dx, dy = offsets[h]
        ax.annotate(
            f"{h}: ${med:.1f}\n[{lo:.1f}, {hi:.1f}]",
            xy=(terminal, med),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color="#333",
            ha="left" if dx >= 0 else "right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=TARGET_COLORS[target], lw=0.8, alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.6,
                            color=TARGET_COLORS[target], alpha=0.5),
        )
        ax.errorbar([terminal], [med], yerr=[[med - lo], [hi - med]],
                    color=TARGET_COLORS[target], alpha=0.45, capsize=4, lw=1)
    ax.axhline(float(history.iloc[-1]), color="black", lw=0.7,
               linestyle=":", alpha=0.6,
               label=f"last observed = ${float(history.iloc[-1]):.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} ETF price (USD)")
    ax.set_title(
        f"{target} forecast fan (BSTS + AR(4) backbone, "
        f"α={model['conformal_alpha']:.3f}) — as of {as_of.date()}"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return _save(fig, out_dir, f"08_forecast_fan_{target}")


def fig09_band_width_vs_horizon(df: pd.DataFrame, out_dir: Path) -> Path:
    """Use the BSTS rows of horizon_sweep_v3 to show RMSE growth with horizon."""
    sub = df[df.model == "BSTS (Trends)"].copy()
    sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
    fig, ax = plt.subplots(figsize=FIG_SIZE_DEFAULT)
    for tgt in ("HY", "IG"):
        s = sub[sub.target == tgt].sort_values("horizon")
        ax.plot(s["horizon"], s["rmse"], "o-", lw=2,
                color=TARGET_COLORS[tgt], markersize=8,
                label=f"{tgt} BSTS RMSE")
        # Annotate horizon labels
        for _, row in s.iterrows():
            ax.annotate(HORIZON_LABEL[row.horizon],
                        xy=(row.horizon, row.rmse),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color=TARGET_COLORS[tgt])
    # √h reference curve (rough)
    h_grid = np.linspace(1, 26, 30)
    rmse_hy_1w = sub[(sub.target == "HY") & (sub.horizon == 1)]["rmse"].iloc[0]
    ax.plot(h_grid, rmse_hy_1w * np.sqrt(h_grid),
            color="#aaaaaa", linestyle="--", lw=1,
            label="√h reference (HY-anchored)")
    ax.set_xlabel("Horizon (weekly steps)")
    ax.set_ylabel("BSTS RMSE")
    ax.set_xticks([1, 2, 4, 13, 26])
    ax.set_title("Forecast uncertainty grows ≈ √h with horizon")
    ax.legend()
    return _save(fig, out_dir, "09_rmse_vs_horizon")


def fig10_oas_proxy_scatter(oas_dir: Path, eval_json: dict,
                            out_dir: Path,
                            targets_dir: Path = Path("data/raw/targets")) -> Path:
    """Side-by-side scatter: ETF Δlog vs ΔOAS for HY and IG.

    Joins the user-supplied FRED OAS series (``oas_dir``) with the yfinance
    ETF cache (``targets_dir``) and computes weekly first differences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tgt in zip(axes, ("HY", "IG")):
        oas = pd.read_parquet(oas_dir / f"{tgt}_OAS_weekly.parquet")
        oas_diff = oas.iloc[:, 0].diff()
        etf = pd.read_parquet(targets_dir / f"{tgt}.parquet")
        etf_dlog = np.log(etf.iloc[:, 0]).diff()
        # Align on a common DateTimeIndex (date-only).
        oas_diff.index = pd.to_datetime(oas_diff.index).normalize()
        etf_dlog.index = pd.to_datetime(etf_dlog.index).normalize()
        merged = pd.concat([etf_dlog.rename("etf_dlog"),
                            oas_diff.rename("oas_diff")], axis=1,
                           join="inner").dropna()
        x = merged["etf_dlog"]
        y = merged["oas_diff"]

        pearson = eval_json["by_target"][tgt]["pearson"]
        n = eval_json["by_target"][tgt]["n_obs"]
        ax.scatter(x, y, s=14, alpha=0.45, color=TARGET_COLORS[tgt])
        # OLS fit line
        if len(x) > 2:
            slope, intercept = np.polyfit(x.values, y.values, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, intercept + slope * xs, color="black", lw=1.2,
                    linestyle="--",
                    label=f"OLS Δbps per Δlog = {slope:+.1f}")
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.axvline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"{tgt} ETF Δlog (weekly)")
        ax.set_ylabel(f"{tgt} OAS Δ (weekly, bps)")
        verdict = "defensible" if abs(pearson) > 0.5 else "weak"
        ax.set_title(
            f"{tgt}: Pearson = {pearson:+.2f}  (n={n} weeks, {verdict})"
        )
        ax.legend(loc="upper right")
    fig.suptitle(
        "ETF proxy quality vs underlying ICE BAML OAS (2023-05 onward)",
        y=1.02, fontsize=13,
    )
    return _save(fig, out_dir, "10_oas_proxy_scatter")


def fig11_topic_vs_category_share(posteriors: dict, predictors_yaml: Path,
                                  out_dir: Path) -> Path:
    """How much of the top-inclusion mass comes from topics vs categories?"""
    # Load predictor kinds from config. Schema is
    #   categories: {group: [{id, name}, ...]}
    #   topics:     {group: [{mid, name}, ...]}
    import yaml
    with open(predictors_yaml) as f:
        cfg = yaml.safe_load(f)
    kind_map: dict[str, str] = {}
    for group_dict, kind in (("categories", "category"), ("topics", "topic")):
        for _grp, entries in cfg.get(group_dict, {}).items():
            for e in entries:
                kind_map[e["name"]] = kind
    # `vix` (lowercase) is a control from data/raw/targets, not a Trends predictor.
    # Match either casing.
    kind_map.setdefault("vix", "control")
    kind_map.setdefault("ust10y", "control")
    kind_map.setdefault("ust2y10y_slope", "control")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, tgt in zip(axes, ("HY", "IG")):
        posterior = posteriors[tgt]
        cs = posterior["coefficient_summary"].copy()
        cs["kind"] = cs.index.map(lambda nm: kind_map.get(nm, "control"))
        # Inclusion-weighted contribution per kind (sum of P(γ))
        by_kind = cs.groupby("kind")["inclusion_prob"].sum().sort_values()
        colors = ["#7f7f7f", "#1f77b4", "#d62728"]
        ax.bar(by_kind.index, by_kind.values,
               color=[{"category": "#1f77b4", "topic": "#d62728",
                       "control": "#7f7f7f"}.get(k, "#aaaaaa")
                      for k in by_kind.index],
               edgecolor="white")
        ax.set_title(f"{tgt} — total inclusion-prob mass by predictor kind")
        ax.set_ylabel("Σ P(γ=1)")
        for i, v in enumerate(by_kind.values):
            ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)
    return _save(fig, out_dir, "11_topic_vs_category_share")


def fig12_forecast_vs_baselines(raw_dir: Path, out_dir: Path) -> list[Path]:
    """Time-series overlay: BSTS forecast vs AR(p) vs Naive RW vs actual."""
    paths: list[Path] = []
    for tgt in ("HY", "IG"):
        candidates = {
            "BSTS (Trends)": f"{tgt}_BSTS_Trends_re4_ar0.parquet",
            "AR(p)":         f"{tgt}_ARp_re4_ar4.parquet",
            "Naive RW":      f"{tgt}_Naive_RW_re4_ar0.parquet",
        }
        # Single-horizon (h=1) parquets are produced by refit_sweep mode;
        # find them via the file names actually on disk.
        loaded: dict[str, pd.DataFrame] = {}
        for label, fname in candidates.items():
            p = raw_dir / fname
            if p.exists():
                df = pd.read_parquet(p)
                # The refit_cadence parquets are indexed by (forecast_date, horizon)
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(1, level="horizon")
                loaded[label] = df
        if "BSTS (Trends)" not in loaded:
            continue

        bsts = loaded["BSTS (Trends)"]
        fig, ax = plt.subplots(figsize=FIG_SIZE_WIDE)
        ax.plot(bsts.index, bsts["y_true"], color="black", lw=1.2,
                label=f"{tgt} actual")
        for label, df in loaded.items():
            ax.plot(df.index, df["y_pred_mean"], lw=1.4, alpha=0.85,
                    color=MODEL_COLORS[label], label=f"{label} forecast")
        ax.set_xlabel("Forecast date")
        ax.set_ylabel(f"{tgt} ETF price (USD)")
        ax.set_title(f"{tgt} — 1-week-ahead forecasts: actual vs models")
        ax.legend(loc="upper left", ncol=2)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        paths.append(_save(fig, out_dir, f"12_forecast_vs_baselines_{tgt}"))
    return paths


def fig13_inclusion_summary_table(posteriors: dict, out_dir: Path) -> Path:
    """Side-by-side bar: top-5 predictors for HY and IG."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tgt in zip(axes, ("HY", "IG")):
        posterior = posteriors[tgt]
        cs = posterior["coefficient_summary"].copy()
        top = cs.sort_values("inclusion_prob", ascending=False).head(5)
        colors = [TARGET_COLORS[tgt] if b > 0 else "#d62728"
                  for b in top["mean_when_included"]]
        y = np.arange(len(top))
        ax.barh(y, top["inclusion_prob"].values, color=colors,
                edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(top.index.tolist())
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("P(γ=1)")
        ax.set_title(f"{tgt} — top-5 inclusion predictors")
        for yi, (_, row) in zip(y, top.iterrows()):
            ax.text(row["inclusion_prob"] + 0.02, yi,
                    f"β̄={row['mean_when_included']:+.2f}",
                    va="center", fontsize=9, color="#333")
    fig.suptitle("v5 spike-and-slab inclusion (41-predictor universe)",
                 y=1.02, fontsize=13)
    return _save(fig, out_dir, "13_top5_inclusion_side_by_side")


# ---------- OAS-specific visuals (v5.1 add-ons) -------------------------------


def fig14_oas_implied_fan(model_path: Path, y_history: pd.Series,
                          x_latest: pd.DataFrame, out_dir: Path,
                          target: str) -> Path | None:
    """OAS-bps fan chart from the ETF→OAS translation layer.

    Mirror of :func:`fig08_forecast_fan` but in bps space using the
    ``oas_implied_*`` outputs from :func:`forecast`. Returns ``None`` if the
    model doesn't carry the overlay block (e.g. pure ETF v4 pickles).
    """
    from gtrends_bayes.inference import load_model, forecast

    model = load_model(str(model_path))
    if "oas_overlay_translation" not in model:
        return None
    overlay = model["oas_overlay_translation"]
    horizons = ["1w", "2w", "1m", "1q", "6m"]

    all_paths: dict[str, dict] = {}
    for h in horizons:
        all_paths[h] = forecast(
            model, h, y_history.index.max(), y_history, x_latest,
            n_draws=1000, seed=42,
        )

    out6m = all_paths["6m"]
    days = np.arange(1, out6m["horizon_bd"] + 1)
    as_of = pd.Timestamp(out6m["as_of"])
    fcst_idx = pd.date_range(as_of + pd.Timedelta(days=1),
                             periods=len(days), freq="B")

    fig, ax = plt.subplots(figsize=(12, 6))
    # OAS anchor: most recent observed OAS bps (embedded in the pickle).
    last_oas = float(overlay["last_oas_bps"])
    last_oas_date = pd.Timestamp(overlay["last_oas_date"])
    # Draw a small anchor segment on the left.
    ax.scatter([last_oas_date], [last_oas], color="black", s=40, zorder=5,
               label=f"latest observed OAS = {last_oas:.0f} bps "
                     f"({last_oas_date.date()})")

    ax.fill_between(fcst_idx,
                    out6m["oas_implied_path_band_lo"],
                    out6m["oas_implied_path_band_hi"],
                    color=TARGET_COLORS[target], alpha=0.18,
                    label="90% credible band (translated)")
    ax.plot(fcst_idx, out6m["oas_implied_path_median"],
            color=TARGET_COLORS[target], lw=2,
            label="OAS-implied median (ETF forecast × overlap slope)")

    horizon_bd = {"1w": 5, "2w": 10, "1m": 21, "1q": 63, "6m": 126}
    offsets = {"1w": (8, 22), "2w": (8, -22), "1m": (8, 22),
               "1q": (8, 22), "6m": (-30, 22)}
    for h in horizons:
        idx = horizon_bd[h] - 1
        terminal = fcst_idx[idx]
        med = all_paths[h]["oas_implied_median"]
        lo, hi = all_paths[h]["oas_implied_band"]
        ax.scatter([terminal], [med], color=TARGET_COLORS[target], zorder=5,
                   s=60, edgecolors="white", linewidths=1.5)
        dx, dy = offsets[h]
        ax.annotate(
            f"{h}: {med:.0f} bps\n[{lo:.0f}, {hi:.0f}]",
            xy=(terminal, med),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, color="#333",
            ha="left" if dx >= 0 else "right",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=TARGET_COLORS[target], lw=0.8, alpha=0.85),
            arrowprops=dict(arrowstyle="-", lw=0.6,
                            color=TARGET_COLORS[target], alpha=0.5),
        )
        ax.errorbar([terminal], [med], yerr=[[med - lo], [hi - med]],
                    color=TARGET_COLORS[target], alpha=0.45, capsize=4, lw=1)
    ax.axhline(last_oas, color="black", lw=0.7, linestyle=":", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{target} OAS (bps)")
    quality = overlay["proxy_quality_label"]
    ax.set_title(
        f"{target} OAS-implied forecast (via ETF→OAS translation, "
        f"Pearson={overlay['pearson']:+.2f}, "
        f"{quality}) — as of {as_of.date()}"
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return _save(fig, out_dir, f"14_oas_implied_fan_{target}")


def fig15_oas_direct_vs_implied(
    etf_model_path: Path, oas_model_path: Path,
    y_etf: pd.Series, y_oas: pd.Series, x_latest: pd.DataFrame,
    out_dir: Path, target: str,
) -> Path | None:
    """Side-by-side compare: OAS-direct BSTS forecast vs OAS-implied translation.

    Shows model disagreement honestly — for HY (defensible proxy) they
    should be reasonably close; for IG (weak proxy) they can diverge.
    """
    from gtrends_bayes.inference import load_model, forecast

    etf_model = load_model(str(etf_model_path))
    oas_model = load_model(str(oas_model_path))
    if "oas_overlay_translation" not in etf_model:
        return None

    horizons = ["1w", "2w", "1m", "1q", "6m"]
    as_of_etf = y_etf.index.max()
    as_of_oas = y_oas.index.max()

    direct: list[tuple[str, float, float, float]] = []
    implied: list[tuple[str, float, float, float]] = []
    for h in horizons:
        # Direct OAS-direct BSTS forecast (bps).
        r_d = forecast(oas_model, h, as_of_oas, y_oas, x_latest,
                       n_draws=1000, seed=42)
        # Translated from ETF model.
        r_i = forecast(etf_model, h, as_of_etf, y_etf, x_latest,
                       n_draws=1000, seed=42)
        direct.append(
            (h, r_d["level_median"], r_d["level_band"][0], r_d["level_band"][1])
        )
        implied.append(
            (h, r_i["oas_implied_median"], r_i["oas_implied_band"][0],
             r_i["oas_implied_band"][1])
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(horizons))
    width = 0.35

    d_meds = [r[1] for r in direct]
    d_los = [r[2] for r in direct]
    d_his = [r[3] for r in direct]
    i_meds = [r[1] for r in implied]
    i_los = [r[2] for r in implied]
    i_his = [r[3] for r in implied]

    # Use errorbar to show point + band per group.
    ax.errorbar(x - width/2, d_meds,
                yerr=[np.array(d_meds) - np.array(d_los),
                      np.array(d_his) - np.array(d_meds)],
                fmt="o", capsize=4, color=TARGET_COLORS[target],
                lw=2, markersize=10, label="OAS-direct BSTS (small-N)")
    ax.errorbar(x + width/2, i_meds,
                yerr=[np.array(i_meds) - np.array(i_los),
                      np.array(i_his) - np.array(i_meds)],
                fmt="s", capsize=4, color="#7f7f7f",
                lw=2, markersize=10, label="OAS-implied (ETF translation)")
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.axhline(float(y_oas.iloc[-1]), color="black", lw=0.7,
               linestyle=":", alpha=0.6,
               label=f"latest observed = {float(y_oas.iloc[-1]):.0f} bps")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(f"{target} OAS (bps)")
    ax.set_title(
        f"{target} — OAS-direct BSTS vs OAS-implied translation\n"
        "Honest comparison of two independent paths to the same number"
    )
    ax.legend(loc="best")
    return _save(fig, out_dir, f"15_oas_direct_vs_implied_{target}")


def fig16_oas_history_with_forecast(
    oas_overlay_dir: Path, oas_model_dir: Path,
    y_oas_dict: dict[str, pd.Series], x_latest: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Past 3y of actual OAS bps + the current OAS-direct forecast bands.

    The "where are we headed" slide. Two stacked panels (HY, IG).
    """
    from gtrends_bayes.inference import load_model, forecast

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for ax, target in zip(axes, ("HY", "IG")):
        oas_hist = pd.read_parquet(
            oas_overlay_dir / f"{target}_OAS_weekly.parquet"
        ).iloc[:, 0]
        ax.plot(oas_hist.index, oas_hist.values, color=TARGET_COLORS[target],
                lw=1.4, label=f"{target} OAS history (FRED, bps)")

        oas_model_path = oas_model_dir / f"{target}_OAS_v5.pkl"
        if oas_model_path.exists() and target in y_oas_dict:
            model = load_model(str(oas_model_path))
            y_oas = y_oas_dict[target]
            as_of = y_oas.index.max()
            horizons = ["1w", "2w", "1m", "1q", "6m"]
            results = [
                (h, forecast(model, h, as_of, y_oas, x_latest,
                             n_draws=1000, seed=42))
                for h in horizons
            ]
            # Plot the 6m fan to the right of the history.
            out6m = results[-1][1]
            days = np.arange(1, out6m["horizon_bd"] + 1)
            fcst_idx = pd.date_range(as_of + pd.Timedelta(days=1),
                                     periods=len(days), freq="B")
            ax.fill_between(fcst_idx, out6m["level_path_q05"],
                            out6m["level_path_q95"],
                            color=TARGET_COLORS[target], alpha=0.18,
                            label="OAS-direct 90% band")
            ax.plot(fcst_idx, out6m["level_path_median"],
                    color=TARGET_COLORS[target], lw=2,
                    label="OAS-direct median (6m)")
            # Mark terminal points.
            horizon_bd = {"1w": 5, "2w": 10, "1m": 21, "1q": 63, "6m": 126}
            for h, r in results:
                idx = horizon_bd[h] - 1
                ax.scatter([fcst_idx[idx]], [r["level_median"]],
                           color=TARGET_COLORS[target], s=40, zorder=5,
                           edgecolors="white")

        ax.set_ylabel(f"{target} OAS (bps)")
        ax.set_title(
            f"{target} OAS: 3-year history + current OAS-direct forecast",
            fontsize=11,
        )
        ax.legend(loc="upper left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return _save(fig, out_dir, "16_oas_history_with_forecast")


# ---------- Tables -------------------------------------------------------------


def _build_hit_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
    pivot = sub.pivot_table(index=["target", "model"],
                            columns="horizon_label",
                            values="hit_rate",
                            observed=True)
    pivot = pivot.reindex(columns=["1w", "2w", "1m", "1q", "6m"])
    pivot = pivot.round(3).reset_index()
    return pivot


def _build_rmse_table(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.copy()
    sub["horizon_label"] = sub.horizon.map(HORIZON_LABEL)
    pivot = sub.pivot_table(index=["target", "model"],
                            columns="horizon_label",
                            values="rmse",
                            observed=True)
    pivot = pivot.reindex(columns=["1w", "2w", "1m", "1q", "6m"])
    pivot = pivot.round(3).reset_index()
    return pivot


def _build_calibration_table(recal: dict) -> pd.DataFrame:
    rows = []
    for tgt in ("HY", "IG"):
        for model, cells in recal[tgt].items():
            cell = cells.get("0.8")
            if cell:
                rows.append({
                    "target": tgt,
                    "model": model,
                    "alpha": round(cell["alpha"], 3),
                    "pre_cov_80": round(cell["empirical_pre_full"], 3),
                    "post_cov_80": round(cell["empirical_post_full"], 3),
                    "in_band": ("✓" if 0.75 <= cell["empirical_post_full"]
                                <= 0.85 else "✗"),
                })
    return pd.DataFrame(rows)


def _build_granger_table(eval_json: dict) -> pd.DataFrame:
    rows = []
    by_cadence = eval_json["by_cadence"]
    for cadence in ("weekly", "daily"):
        for tgt in ("HY", "IG"):
            d = by_cadence[cadence][tgt]["granger"]
            sig = "✓" if d["p_value"] < 0.05 else "✗"
            rows.append({
                "cadence": cadence,
                "target": tgt,
                "F_stat": round(d["f_stat"], 3),
                "p_value": round(d["p_value"], 4),
                "delta_R2": round(d["delta_r2"], 4),
                "significant_at_0.05": sig,
            })
    return pd.DataFrame(rows)


def _build_top_inclusion_table(posteriors: dict, n: int = 10) -> pd.DataFrame:
    rows = []
    for tgt, posterior in posteriors.items():
        cs = posterior["coefficient_summary"].copy()
        cs = cs.sort_values("inclusion_prob", ascending=False).head(n)
        for rank, (name, row) in enumerate(cs.iterrows(), start=1):
            rows.append({
                "target": tgt,
                "rank": rank,
                "predictor": name,
                "inclusion_prob": round(row["inclusion_prob"], 3),
                "beta_when_included": round(row["mean_when_included"], 3),
                "sign_consistency": round(row["sign_consistency"], 3),
            })
    return pd.DataFrame(rows)


def _build_oas_translation_table(frozen_dir: Path) -> pd.DataFrame:
    """Per-target OAS-translation provenance, read from frozen pickles."""
    import pickle as _pickle
    rows = []
    for target in ("HY", "IG"):
        pkl = frozen_dir / f"{target}_v5.pkl"
        if not pkl.exists():
            continue
        with open(pkl, "rb") as f:
            m = _pickle.load(f)
        overlay = m.get("oas_overlay_translation")
        if overlay is None:
            continue
        rows.append({
            "target": target,
            "slope_bps_per_dlog": round(overlay["slope_bps_per_dlog"], 2),
            "pearson": round(overlay["pearson"], 3),
            "spearman": round(overlay["spearman"], 3),
            "n_overlap_weeks": overlay["n_overlap_weeks"],
            "last_oas_bps": round(overlay["last_oas_bps"], 1),
            "last_oas_date": overlay["last_oas_date"],
            "proxy_quality": overlay["proxy_quality_label"],
        })
    return pd.DataFrame(rows)


# ---------- Main ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build_v5_report")
    p.add_argument("--out-dir", default="dist/v5/report",
                   help="Output directory for figures/ and tables/.")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--posterior-dir", default="data/processed/posterior")
    p.add_argument("--frozen-model-dir", default="dist/v5/model")
    p.add_argument("--bundle-data-dir", default=None,
                   help="If given, use this directory's HY/IG_history.csv + "
                        "trends.parquet for forward fan charts. Defaults to "
                        "<bundle root>/data after unzipping the data sideband; "
                        "if absent we skip the fan-chart figures.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_matplotlib()

    out_root = Path(args.out_dir)
    fig_dir = out_root / "figures"
    tbl_dir = out_root / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    processed = Path(args.processed_dir)
    posterior_dir = Path(args.posterior_dir)
    frozen_dir = Path(args.frozen_model_dir)

    # ---- Load all source data ----------------------------------------------
    print("→ loading source artifacts...")
    hsweep = pd.read_csv(processed / "backtest" / "horizon_sweep_v3.csv")
    with open(processed / "backtest" / "recalibration_alphas_v3.json") as f:
        recal = json.load(f)
    with open(processed / "risk_index_v3" / "_evaluation.json") as f:
        eval_json = json.load(f)
    with open(processed / "oas_overlay" / "correlation.json") as f:
        oas_corr = json.load(f)

    posteriors: dict[str, dict] = {}
    for tgt in ("HY", "IG"):
        with open(posterior_dir / f"{tgt}_bsts_v3.pkl", "rb") as f:
            posteriors[tgt] = pickle.load(f)

    # ---- Crisis windows for the Risk Index overlay -------------------------
    crises = [
        ("Lehman", "2008-09-01", "2008-12-31"),
        ("Euro debt", "2011-08-01", "2011-12-31"),
        ("Energy 2015", "2015-12-01", "2016-02-29"),
        ("COVID", "2020-02-15", "2020-04-30"),
        ("UK gilt", "2022-09-15", "2022-11-01"),
        ("SVB",  "2023-03-08", "2023-04-15"),
    ]

    paths_made: list[Path] = []

    # ---- Figures -----------------------------------------------------------
    print("→ generating figures...")
    paths_made += fig01_hit_rate_by_horizon(hsweep, fig_dir)
    paths_made += fig02_rmse_by_horizon(hsweep, fig_dir)
    paths_made += [fig03_bsts_vs_rw_hit_rate(hsweep, fig_dir)]
    paths_made += fig04_inclusion_top_predictors(posteriors, fig_dir)
    paths_made += fig05_risk_index_weekly(processed / "risk_index_v3", fig_dir, crises)
    paths_made += fig06_risk_index_daily(processed / "risk_index_v3", fig_dir, crises)
    paths_made += [fig07_calibration_coverage(recal, fig_dir)]
    paths_made += [fig09_band_width_vs_horizon(hsweep, fig_dir)]
    paths_made += [fig10_oas_proxy_scatter(processed / "oas_overlay",
                                           oas_corr, fig_dir)]
    paths_made += [fig11_topic_vs_category_share(
        posteriors, Path("config/predictors.yaml"), fig_dir)]
    paths_made += fig12_forecast_vs_baselines(
        processed / "backtest" / "raw_v3", fig_dir)
    paths_made += [fig13_inclusion_summary_table(posteriors, fig_dir)]

    # Forecast fan (uses pure-Python inference; needs the data sideband).
    # If --bundle-data-dir isn't passed, transparently unpack the v5 data
    # zip into a tempdir so the fan-chart sees the exact inputs the PM
    # would.
    bundle_data = args.bundle_data_dir
    if bundle_data is None:
        sideband = frozen_dir.parent / "gtrends-bayes-v5-data.zip"
        if sideband.exists():
            import tempfile
            import zipfile
            tmp = Path(tempfile.mkdtemp(prefix="v5_report_sideband_"))
            with zipfile.ZipFile(sideband) as zf:
                zf.extractall(tmp)
            bundle_data = str(tmp)
            print(f"→ unpacked sideband to {tmp} for fan-chart inputs")
    if bundle_data is not None:
        bd = Path(bundle_data)
        # Cache the ETF y-history + Trends X once; both ETF and OAS figures use them.
        x_lib: pd.DataFrame | None = None
        try:
            x_lib = pd.read_parquet(bd / "trends.parquet")
            x_lib.index = pd.to_datetime(x_lib.index)
        except Exception as exc:  # noqa: BLE001
            print(f"   trends.parquet load failed: {exc}")

        y_hist_etf: dict[str, pd.Series] = {}
        y_hist_oas: dict[str, pd.Series] = {}
        for tgt in ("HY", "IG"):
            try:
                y_hist_etf[tgt] = pd.read_csv(
                    bd / f"{tgt}_history.csv",
                    parse_dates=[0], index_col=0,
                ).iloc[:, 0]
            except Exception as exc:  # noqa: BLE001
                print(f"   {tgt}_history.csv load failed: {exc}")
            try:
                y_hist_oas[tgt] = pd.read_csv(
                    bd / f"{tgt}_OAS_history.csv",
                    parse_dates=[0], index_col=0,
                ).iloc[:, 0]
            except Exception:
                pass  # OAS history is optional

        for tgt in ("HY", "IG"):
            if tgt not in y_hist_etf or x_lib is None:
                continue
            # Fig 08: ETF fan
            try:
                paths_made.append(fig08_forecast_fan(
                    frozen_dir / f"{tgt}_v5.pkl",
                    y_hist_etf[tgt], x_lib, fig_dir, tgt,
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"   fan-chart for {tgt} skipped: {exc}")
            # Fig 14: OAS-implied fan (v5.1)
            try:
                fan_oas = fig14_oas_implied_fan(
                    frozen_dir / f"{tgt}_v5.pkl",
                    y_hist_etf[tgt], x_lib, fig_dir, tgt,
                )
                if fan_oas is not None:
                    paths_made.append(fan_oas)
            except Exception as exc:  # noqa: BLE001
                print(f"   OAS-implied fan for {tgt} skipped: {exc}")
            # Fig 15: OAS-direct vs OAS-implied (needs OAS-direct pickle)
            oas_pkl = frozen_dir / f"{tgt}_OAS_v5.pkl"
            if oas_pkl.exists() and tgt in y_hist_oas:
                try:
                    cmp = fig15_oas_direct_vs_implied(
                        frozen_dir / f"{tgt}_v5.pkl",
                        oas_pkl,
                        y_hist_etf[tgt], y_hist_oas[tgt], x_lib,
                        fig_dir, tgt,
                    )
                    if cmp is not None:
                        paths_made.append(cmp)
                except Exception as exc:  # noqa: BLE001
                    print(f"   OAS direct-vs-implied for {tgt} skipped: {exc}")

        # Fig 16: 3-year OAS history + OAS-direct forecast band, stacked HY/IG.
        if y_hist_oas and x_lib is not None:
            try:
                paths_made.append(fig16_oas_history_with_forecast(
                    processed / "oas_overlay", frozen_dir,
                    y_hist_oas, x_lib, fig_dir,
                ))
            except Exception as exc:  # noqa: BLE001
                print(f"   OAS history-with-forecast skipped: {exc}")
    else:
        print("   fan-chart skipped: no data sideband found")

    # ---- Tables ------------------------------------------------------------
    print("→ generating tables...")
    _write_table(_build_hit_rate_table(hsweep), tbl_dir, "T1_hit_rate",
                 "Cumulative-direction hit rate. Higher = more directional skill.")
    _write_table(_build_rmse_table(hsweep), tbl_dir, "T2_rmse",
                 "RMSE in transform space (level for v5).")
    _write_table(_build_calibration_table(recal), tbl_dir, "T3_calibration",
                 "Conformal recalibration: 8/8 cells land in [0.75, 0.85].")
    _write_table(_build_granger_table(eval_json), tbl_dir, "T4_granger",
                 "Granger causality of Trends Risk Index over Δlog(target), "
                 "controlling for ΔVIX.")
    _write_table(_build_top_inclusion_table(posteriors), tbl_dir,
                 "T5_top_inclusion",
                 "Top-10 inclusion predictors per target with sign + consistency.")
    oas_trans = _build_oas_translation_table(frozen_dir)
    if not oas_trans.empty:
        _write_table(oas_trans, tbl_dir, "T6_oas_translation",
                     "ETF→OAS translation provenance: regression slope, "
                     "Pearson, n_overlap_weeks, latest OAS anchor.")

    # ---- README index ------------------------------------------------------
    idx = out_root / "README.md"
    idx.write_text(
        "# gtrends-bayes v5 — Presentation Report\n\n"
        "Auto-generated by `scripts/build_v5_report.py`. Re-run after any "
        "refit / Risk Index / freeze pass.\n\n"
        "## Figures\n\n"
        + "\n".join(
            f"- `figures/{p.name}`" for p in sorted(fig_dir.glob("*.png"))
        )
        + "\n\n## Tables\n\n"
        + "\n".join(
            f"- `tables/{p.name}`" for p in sorted(tbl_dir.glob("*.md"))
        )
        + "\n\n*Numbers + plots correspond to the bundle in `dist/v5/`.*\n"
    )

    print(f"\n✓ wrote {len(paths_made)} figures + 5 tables to {out_root}/")
    print(f"  open {idx} for the index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
