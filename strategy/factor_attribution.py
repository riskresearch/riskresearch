"""
Factor attribution analysis
============================
Rolls the strategy's daily returns against four factors:
  1. Market beta     — ES=F daily returns (equity risk)
  2. Duration        — ZN=F daily returns (bond/duration risk)
  3. Momentum        — UMD factor from Ken French library
  4. Carry           — approximated from the IR signal vs ES

Uses rolling 252-day OLS windows to show how factor exposures
evolve over time, and a full-sample decomposition to show what
fraction of strategy return is explained by each factor.

Run from the project root:
    python strategy/factor_attribution.py

Output:
    strategy/outputs/factor/fig_rolling_betas.png
    strategy/outputs/factor/fig_factor_decomposition.png
    strategy/outputs/factor/fig_alpha_timeline.png
    strategy/outputs/factor/factor_stats.csv
"""

import sys
import urllib.request
import zipfile
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas_datareader.data as web
from datetime import datetime

# =========================================================
# Paths
# =========================================================
TABLES_DIR = Path(__file__).parent / "outputs" / "tables"
OUT_DIR    = Path(__file__).parent / "outputs" / "factor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS = {
    "market":   "#1a1a1a",
    "duration": "#2166ac",
    "momentum": "#d6604d",
    "carry":    "#4dac26",
    "alpha":    "#756bb1",
    "strategy": "#1a1a1a",
}

ROLL_WINDOW = 252   # rolling OLS window in trading days
MIN_PERIODS = 126   # minimum periods for rolling estimate


# =========================================================
# Download helpers
# =========================================================
def fred(series: str, start: datetime, end: datetime) -> pd.Series:
    return web.DataReader(series, "fred", start, end).squeeze()


def download_ff_momentum() -> pd.Series:
    """Download Ken French UMD (momentum) factor."""
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Momentum_Factor_CSV.zip"
    )
    print("  Downloading UMD momentum factor from Ken French...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        fname = [n for n in z.namelist()
                 if n.endswith(".CSV") or n.endswith(".csv")][0]
        with z.open(fname) as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()

    records = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) >= 2:
            try:
                val = int(parts[0])
                if len(parts[0]) == 6:
                    started = True
            except ValueError:
                if started:
                    break
                continue
        if started:
            try:
                date_int = int(parts[0])
                year     = date_int // 100
                month    = date_int % 100
                if 1 <= month <= 12 and 1900 <= year <= 2100:
                    val = float(parts[1]) / 100
                    records.append((pd.Timestamp(year=year, month=month, day=1), val))
            except (ValueError, TypeError):
                continue

    s = pd.Series(dict(records), name="UMD")
    s.index = pd.DatetimeIndex(s.index)
    return s


# =========================================================
# Load strategy data
# =========================================================
def load_strategy_data() -> dict:
    print("\nLoading strategy outputs...")

    # Equity curves
    eq_path = TABLES_DIR / "rebased_equity_curves.csv"
    if not eq_path.exists():
        raise FileNotFoundError(
            f"{eq_path} not found. Run strategy.py first."
        )
    curves = pd.read_csv(eq_path, index_col=0, parse_dates=True)

    # Find portfolio column
    port_col = None
    for c in curves.columns:
        if "portfolio_plot" in c or c == "Final portfolio":
            port_col = c
            break
    if port_col is None:
        port_col = curves.columns[0]

    equity = curves[port_col].dropna()
    strat_rets = equity.pct_change().dropna()
    print(f"  Strategy returns: {len(strat_rets)} days, "
          f"{strat_rets.index[0].date()} to {strat_rets.index[-1].date()}")

    # Asset contribution equity (individual futures returns)
    contrib_path = TABLES_DIR / "asset_contribution_equity.csv"
    asset_rets   = None
    if contrib_path.exists():
        contrib = pd.read_csv(contrib_path, index_col=0, parse_dates=True)
        asset_rets = contrib.pct_change().dropna()
        print(f"  Asset returns loaded: {asset_rets.shape}")

    # IR signal (carry proxy)
    ir_path = TABLES_DIR / "alt_ir_vs_es.csv"
    ir_signal = None
    if ir_path.exists():
        ir_df    = pd.read_csv(ir_path, index_col=0, parse_dates=True)
        ir_signal = ir_df
        print(f"  IR signal loaded: {ir_signal.shape}")

    # Weights
    wt_path = TABLES_DIR / "weights_signal.csv"
    weights = None
    if wt_path.exists():
        weights = pd.read_csv(wt_path, index_col=0, parse_dates=True)

    return {
        "strat_rets":  strat_rets,
        "equity":      equity,
        "asset_rets":  asset_rets,
        "ir_signal":   ir_signal,
        "weights":     weights,
    }


# =========================================================
# Build factor returns
# =========================================================
def build_factors(strat_rets: pd.Series) -> pd.DataFrame:
    start = strat_rets.index[0]
    end   = strat_rets.index[-1]

    print("\nBuilding factor returns...")

    factors = {}

    # 1. Market factor — use ES=F from asset contributions if available
    # Otherwise download SPY as proxy
    try:
        import yfinance as yf
        es_raw = yf.download("ES=F", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
        if not es_raw.empty:
            es_price = es_raw["Close"].squeeze()
            es_rets  = es_price.pct_change().dropna()
            es_rets.index = pd.DatetimeIndex([
                pd.Timestamp(year=d.year, month=d.month, day=d.day)
                for d in es_rets.index
            ])
            factors["Market (ES)"] = es_rets
            print(f"  Market (ES=F): {len(es_rets)} days")
    except Exception as e:
        print(f"  Warning: ES=F download failed: {e}")

    # 2. Duration factor — ZN=F (10-year Treasury futures)
    try:
        import yfinance as yf
        zn_raw = yf.download("ZN=F", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
        if not zn_raw.empty:
            zn_price = zn_raw["Close"].squeeze()
            zn_rets  = zn_price.pct_change().dropna()
            zn_rets.index = pd.DatetimeIndex([
                pd.Timestamp(year=d.year, month=d.month, day=d.day)
                for d in zn_rets.index
            ])
            factors["Duration (ZN)"] = zn_rets
            print(f"  Duration (ZN=F): {len(zn_rets)} days")
    except Exception as e:
        print(f"  Warning: ZN=F download failed: {e}")

    # 3. Momentum factor — UMD from Ken French (monthly -> daily approx)
    try:
        umd_monthly = download_ff_momentum()
        # Convert monthly to daily by forward-filling within month
        umd_daily = umd_monthly.reindex(
            strat_rets.index, method="ffill"
        ) / 21  # approximate daily from monthly
        factors["Momentum (UMD)"] = umd_daily
        print(f"  Momentum (UMD): {len(umd_daily)} days")
    except Exception as e:
        print(f"  Warning: UMD download failed: {e}")

    # 4. Gold factor — GC=F as real asset / inflation hedge proxy
    try:
        import yfinance as yf
        gc_raw = yf.download("GC=F", start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
        if not gc_raw.empty:
            gc_price = gc_raw["Close"].squeeze()
            gc_rets  = gc_price.pct_change().dropna()
            gc_rets.index = pd.DatetimeIndex([
                pd.Timestamp(year=d.year, month=d.month, day=d.day)
                for d in gc_rets.index
            ])
            factors["Gold (GC)"] = gc_rets
            print(f"  Gold (GC=F): {len(gc_rets)} days")
    except Exception as e:
        print(f"  Warning: GC=F download failed: {e}")

    factor_df = pd.DataFrame(factors)
    return factor_df


# =========================================================
# Full-sample OLS regression
# =========================================================
def full_sample_regression(
    strat_rets: pd.Series,
    factor_df: pd.DataFrame,
) -> dict:
    common = strat_rets.index.intersection(factor_df.dropna().index)
    y = strat_rets.loc[common].values
    X_raw = factor_df.loc[common].values

    # Remove rows with NaN in factors
    valid = ~np.isnan(X_raw).any(axis=1)
    y     = y[valid]
    X_raw = X_raw[valid]

    # Add constant
    X = np.column_stack([np.ones(len(X_raw)), X_raw])
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    alpha_daily = coeffs[0]
    betas       = coeffs[1:]
    fitted      = X @ coeffs
    resid       = y - fitted

    # R-squared
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Annualized alpha
    alpha_ann = alpha_daily * 252

    # T-statistics (approximate)
    n         = len(y)
    k         = X.shape[1]
    mse       = ss_res / (n - k)
    cov_beta  = mse * np.linalg.pinv(X.T @ X)
    se        = np.sqrt(np.diag(cov_beta))
    t_stats   = coeffs / se

    factor_names = list(factor_df.columns)

    return {
        "alpha_daily":  alpha_daily,
        "alpha_ann":    alpha_ann,
        "betas":        dict(zip(factor_names, betas)),
        "t_stats":      dict(zip(["alpha"] + factor_names,
                                  t_stats.tolist())),
        "r_squared":    r2,
        "n_obs":        n,
        "residuals":    pd.Series(resid, index=strat_rets.loc[common].index[valid]),
    }


# =========================================================
# Rolling OLS regression
# =========================================================
def rolling_regression(
    strat_rets: pd.Series,
    factor_df: pd.DataFrame,
    window: int = ROLL_WINDOW,
    min_periods: int = MIN_PERIODS,
) -> pd.DataFrame:
    common = strat_rets.index.intersection(factor_df.index)
    y      = strat_rets.loc[common]
    X      = factor_df.loc[common]

    factor_names = list(X.columns)
    results      = []

    for i in range(len(y)):
        if i < min_periods:
            results.append({
                "date":       y.index[i],
                "alpha_ann":  np.nan,
                **{f"beta_{f}": np.nan for f in factor_names},
                "r_squared":  np.nan,
            })
            continue

        start_i = max(0, i - window + 1)
        y_w     = y.iloc[start_i:i + 1].values
        X_w     = X.iloc[start_i:i + 1].values

        # Drop NaN rows
        valid = ~np.isnan(X_w).any(axis=1) & ~np.isnan(y_w)
        if valid.sum() < min_periods:
            results.append({
                "date":       y.index[i],
                "alpha_ann":  np.nan,
                **{f"beta_{f}": np.nan for f in factor_names},
                "r_squared":  np.nan,
            })
            continue

        y_v   = y_w[valid]
        X_v   = X_w[valid]
        X_c   = np.column_stack([np.ones(len(X_v)), X_v])
        coeffs, _, _, _ = np.linalg.lstsq(X_c, y_v, rcond=None)

        fitted  = X_c @ coeffs
        ss_tot  = np.sum((y_v - y_v.mean()) ** 2)
        ss_res  = np.sum((y_v - fitted) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        row = {
            "date":       y.index[i],
            "alpha_ann":  coeffs[0] * 252,
            "r_squared":  r2,
        }
        for j, fname in enumerate(factor_names):
            row[f"beta_{fname}"] = coeffs[j + 1]

        results.append(row)

    df_roll = pd.DataFrame(results).set_index("date")
    return df_roll


# =========================================================
# Chart 1 — Rolling factor betas
# =========================================================
def plot_rolling_betas(
    roll: pd.DataFrame,
    factor_names: list,
) -> None:
    n      = len(factor_names)
    colors = [COLORS["market"], COLORS["duration"],
              COLORS["momentum"], COLORS["carry"]]

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, fname in enumerate(factor_names):
        col = f"beta_{fname}"
        if col not in roll.columns:
            continue

        ax    = axes[i]
        color = colors[i % len(colors)]
        beta  = roll[col].dropna()

        ax.fill_between(beta.index, beta.values, 0,
                        where=(beta.values > 0),
                        alpha=0.2, color=color)
        ax.fill_between(beta.index, beta.values, 0,
                        where=(beta.values <= 0),
                        alpha=0.2, color="#d6604d")
        ax.plot(beta.index, beta.values,
                color=color, lw=1.4,
                label=f"Rolling {ROLL_WINDOW}d beta")
        ax.axhline(0, color="#999999", lw=0.6, linestyle="--")

        # Full-sample mean
        mean_beta = float(beta.mean())
        ax.axhline(mean_beta, color=color, lw=0.8,
                   linestyle=":", alpha=0.7,
                   label=f"Full-sample mean ({mean_beta:.2f})")

        ax.set_ylabel(f"Beta to\n{fname}", fontsize=8, labelpad=6)
        ax.legend(fontsize=7.5, framealpha=0.5, loc="upper right")

    axes[0].set_title(
        f"Rolling {ROLL_WINDOW}-day factor betas\n"
        f"Strategy returns regressed on four factors",
        pad=10,
    )
    axes[-1].set_xlabel("Date", labelpad=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_rolling_betas.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_rolling_betas.png")


# =========================================================
# Chart 2 — Full-sample factor decomposition
# =========================================================
def plot_factor_decomposition(
    reg_result: dict,
    factor_df: pd.DataFrame,
    strat_rets: pd.Series,
) -> None:
    factor_names = list(reg_result["betas"].keys())
    betas        = [reg_result["betas"][f] for f in factor_names]
    t_stats      = [reg_result["t_stats"].get(f, np.nan)
                    for f in factor_names]
    alpha_ann    = reg_result["alpha_ann"]
    r2           = reg_result["r_squared"]

    # Annualized return contribution of each factor
    common = strat_rets.index.intersection(factor_df.dropna().index)
    factor_ann_rets = {}
    for fname in factor_names:
        if fname in factor_df.columns:
            f_ret = factor_df.loc[common, fname].dropna()
            factor_ann_rets[fname] = float(f_ret.mean()) * 252

    contributions = {
        fname: betas[i] * factor_ann_rets.get(fname, 0)
        for i, fname in enumerate(factor_names)
    }
    contributions["Alpha"] = alpha_ann

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: beta bar chart with t-stat annotation
    ax   = axes[0]
    x    = np.arange(len(factor_names))
    clrs = [COLORS["market"], COLORS["duration"],
            COLORS["momentum"], COLORS["carry"]]

    bars = ax.bar(x, betas, color=clrs[:len(factor_names)],
                  width=0.5, edgecolor="white")

    for bar, beta, t in zip(bars, betas, t_stats):
        if np.isfinite(beta):
            ypos = beta + abs(beta) * 0.05 if beta >= 0 \
                   else beta - abs(beta) * 0.08
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"β={beta:.2f}\n(t={t:.1f})",
                ha="center",
                va="bottom" if beta >= 0 else "top",
                fontsize=8,
            )

    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f.replace(" ", "\n") for f in factor_names],
        fontsize=8,
    )
    ax.set_ylabel("Factor loading (β)", labelpad=8)
    ax.set_title(
        f"Full-sample factor loadings\n"
        f"R² = {r2:.1%}  |  "
        f"Annualized alpha = {alpha_ann:.1%}",
        pad=10,
    )

    # Right: return contribution waterfall
    ax2    = axes[1]
    labels = list(contributions.keys())
    vals   = list(contributions.values())
    colors_wf = [
        COLORS["market"], COLORS["duration"],
        COLORS["momentum"], COLORS["carry"],
        COLORS["alpha"],
    ][:len(labels)]

    x2   = np.arange(len(labels))
    bars2 = ax2.bar(x2, [v * 100 for v in vals],
                    color=colors_wf, width=0.5,
                    edgecolor="white")

    for bar, val in zip(bars2, vals):
        if np.isfinite(val):
            ypos = val * 100 + 0.3 if val >= 0 else val * 100 - 0.3
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:.1%}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8, fontweight="bold",
            )

    ax2.axhline(0, color="#999999", lw=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [l.replace(" ", "\n") for l in labels],
        fontsize=8,
    )
    ax2.set_ylabel("Annualized return contribution (%)", labelpad=8)
    ax2.set_title(
        "Return decomposition: factor contributions\n"
        "β × factor annual return",
        pad=10,
    )

    fig.suptitle(
        "Factor attribution — full sample\n"
        "Strategy daily returns regressed on Market, Duration, "
        "Momentum, and Gold factors",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_factor_decomposition.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_factor_decomposition.png")


# =========================================================
# Chart 3 — Rolling alpha timeline
# =========================================================
def plot_alpha_timeline(
    roll: pd.DataFrame,
    equity: pd.Series,
) -> None:
    alpha = roll["alpha_ann"].dropna()
    r2    = roll["r_squared"].dropna()

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel 1: equity curve
    ax = axes[0]
    eq_common = equity.reindex(alpha.index).ffill().dropna()
    ax.plot(eq_common.index, eq_common.values / eq_common.iloc[0],
            color=COLORS["strategy"], lw=1.4)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value\n(log scale)", fontsize=8, labelpad=6)
    ax.set_title(
        f"Rolling {ROLL_WINDOW}-day annualized alpha vs four-factor model",
        pad=8,
    )

    # Panel 2: rolling alpha
    ax2 = axes[1]
    ax2.fill_between(alpha.index, alpha.values * 100, 0,
                     where=(alpha.values > 0),
                     alpha=0.3, color=COLORS["alpha"])
    ax2.fill_between(alpha.index, alpha.values * 100, 0,
                     where=(alpha.values <= 0),
                     alpha=0.3, color=COLORS["momentum"])
    ax2.plot(alpha.index, alpha.values * 100,
             color=COLORS["alpha"], lw=1.4,
             label="Rolling annualized alpha (%)")
    ax2.axhline(0, color="#999999", lw=0.8, linestyle="--")

    mean_alpha = float(alpha.mean()) * 100
    ax2.axhline(mean_alpha, color=COLORS["alpha"], lw=0.8,
                linestyle=":", alpha=0.7,
                label=f"Mean alpha ({mean_alpha:.1f}%)")

    ax2.set_ylabel("Annualized alpha (%)", fontsize=8, labelpad=6)
    ax2.legend(fontsize=7.5, framealpha=0.5)

    # Panel 3: rolling R²
    ax3 = axes[2]
    ax3.fill_between(r2.index, r2.values * 100, 0,
                     alpha=0.25, color=COLORS["duration"])
    ax3.plot(r2.index, r2.values * 100,
             color=COLORS["duration"], lw=1.2,
             label="Rolling R² (%)")
    ax3.axhline(float(r2.mean()) * 100,
                color=COLORS["duration"], lw=0.8,
                linestyle=":", alpha=0.7,
                label=f"Mean R² ({r2.mean():.1%})")
    ax3.set_ylabel("Rolling R² (%)", fontsize=8, labelpad=6)
    ax3.set_xlabel("Date", labelpad=8)
    ax3.legend(fontsize=7.5, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_alpha_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_alpha_timeline.png")


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 60)
    print("RR Strategy — Factor Attribution Analysis")
    print("=" * 60)

    # Load strategy data
    data = load_strategy_data()
    strat_rets = data["strat_rets"]
    equity     = data["equity"]

    # Build factor returns
    factor_df = build_factors(strat_rets)

    if factor_df.empty:
        print("ERROR: No factor data available.")
        return

    print(f"\nFactor data range: "
          f"{factor_df.dropna().index[0].date()} to "
          f"{factor_df.dropna().index[-1].date()}")
    print(f"Factor columns: {factor_df.columns.tolist()}")

    # Align strategy returns and factors
    common     = strat_rets.index.intersection(factor_df.index)
    strat_align = strat_rets.loc[common]
    factor_align = factor_df.loc[common]
    print(f"\nAligned sample: {len(common)} days")

    # Full-sample regression
    print("\nRunning full-sample OLS regression...")
    reg = full_sample_regression(strat_align, factor_align)

    print("\n" + "=" * 60)
    print("FULL-SAMPLE FACTOR ATTRIBUTION")
    print("=" * 60)
    print(f"Sample:           {len(common)} days")
    print(f"R-squared:        {reg['r_squared']:.1%}")
    print(f"Annualized alpha: {reg['alpha_ann']:.1%} "
          f"(t={reg['t_stats'].get('alpha', np.nan):.2f})")
    print(f"\nFactor loadings:")
    for fname, beta in reg["betas"].items():
        t = reg["t_stats"].get(fname, np.nan)
        print(f"  {fname:20s}: β={beta:+.3f}  (t={t:.2f})")

    # Rolling regression
    print(f"\nRunning rolling {ROLL_WINDOW}-day regression...")
    roll = rolling_regression(strat_align, factor_align)
    print(f"Rolling estimates computed: {len(roll)} rows")

    # Summary of rolling alpha
    alpha_series = roll["alpha_ann"].dropna()
    print(f"\nRolling alpha summary:")
    print(f"  Mean:           {alpha_series.mean():.1%}")
    print(f"  Median:         {alpha_series.median():.1%}")
    print(f"  Std:            {alpha_series.std():.1%}")
    print(f"  % positive:     {(alpha_series > 0).mean():.0%}")
    print(f"  Min:            {alpha_series.min():.1%}")
    print(f"  Max:            {alpha_series.max():.1%}")

    # Save results
    stats_dict = {
        "metric":  ["R-squared", "Alpha (ann.)", "Alpha t-stat"] +
                   [f"Beta: {f}" for f in reg["betas"]] +
                   [f"T-stat: {f}" for f in reg["betas"]],
        "value":   [reg["r_squared"], reg["alpha_ann"],
                    reg["t_stats"].get("alpha", np.nan)] +
                   list(reg["betas"].values()) +
                   [reg["t_stats"].get(f, np.nan)
                    for f in reg["betas"]],
    }
    stats_df = pd.DataFrame(stats_dict).set_index("metric")
    stats_df.to_csv(OUT_DIR / "factor_stats.csv")
    roll.to_csv(OUT_DIR / "rolling_estimates.csv")
    print(f"\nStats saved to: {OUT_DIR / 'factor_stats.csv'}")

    # Build charts
    print("\nBuilding charts...")
    factor_names = list(factor_df.columns)
    plot_rolling_betas(roll, factor_names)
    plot_factor_decomposition(reg, factor_align, strat_align)
    plot_alpha_timeline(roll, equity)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()