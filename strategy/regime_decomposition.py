"""
Regime decomposition analysis
==============================
Defines four macro regimes using FRED data, maps each trading day
in the backtest to a regime, and decomposes strategy performance
by regime.

Regimes:
  1. Equity Bull   — S&P 500 12-month return > +15%
  2. Equity Bear   — S&P 500 12-month return < -10% OR VIX > 30
  3. Inflation     — CPI YoY > 4% and rising (3-month change positive)
  4. Neutral       — all other periods

Run from the project root:
    python strategy/regime_decomposition.py

Output:
    strategy/outputs/regime/fig_regime_timeline.png
    strategy/outputs/regime/fig_regime_performance.png
    strategy/outputs/regime/fig_regime_allocation.png
    strategy/outputs/regime/regime_stats.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas_datareader.data as web
from datetime import datetime

# =========================================================
# Paths
# =========================================================
TABLES_DIR = Path(__file__).parent / "outputs" / "tables"
OUT_DIR    = Path(__file__).parent / "outputs" / "regime"
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

REGIME_COLORS = {
    "Equity Bull":  "#2166ac",
    "Equity Bear":  "#d6604d",
    "Inflation":    "#e6a817",
    "Neutral":      "#888888",
}

# =========================================================
# Regime thresholds
# =========================================================
BULL_THRESHOLD    =  0.15   # 12m S&P 500 return > 15%
BEAR_THRESHOLD    = -0.10   # 12m S&P 500 return < -10%
VIX_THRESHOLD     =  30.0   # VIX > 30 => Bear/Risk-off
CPI_THRESHOLD     =   4.0   # CPI YoY > 4%


# =========================================================
# Download macro data
# =========================================================
def download_macro(start: str, end: str) -> pd.DataFrame:
    print("  Downloading macro data from FRED...")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    macro = {}

    # S&P 500 price index
    try:
        sp500 = web.DataReader("SP500", "fred", start_dt, end_dt).squeeze()
        macro["SP500"] = sp500
        print(f"    SP500: {len(sp500)} rows")
    except Exception as e:
        print(f"    Warning: SP500 failed: {e}")

    # VIX
    try:
        vix = web.DataReader("VIXCLS", "fred", start_dt, end_dt).squeeze()
        macro["VIX"] = vix
        print(f"    VIX: {len(vix)} rows")
    except Exception as e:
        print(f"    Warning: VIX failed: {e}")

    # CPI
    try:
        cpi = web.DataReader("CPIAUCSL", "fred", start_dt, end_dt).squeeze()
        macro["CPI"] = cpi
        print(f"    CPI: {len(cpi)} rows")
    except Exception as e:
        print(f"    Warning: CPI failed: {e}")

    return pd.DataFrame(macro)


# =========================================================
# Build regime series
# =========================================================
def build_regimes(macro: pd.DataFrame, equity_index: pd.DatetimeIndex) -> pd.Series:
    """
    Assign each date in equity_index to one of four regimes.
    Priority: Equity Bear > Inflation > Equity Bull > Neutral
    """
    # Resample all macro to business day frequency
    macro_daily = macro.reindex(equity_index, method="ffill")

    # S&P 500 12-month rolling return
    if "SP500" in macro_daily.columns:
        sp500 = macro_daily["SP500"].ffill()
        ret_12m = sp500.pct_change(252)
    else:
        ret_12m = pd.Series(np.nan, index=equity_index)

    # VIX daily
    vix = macro_daily["VIX"].ffill() if "VIX" in macro_daily.columns \
        else pd.Series(np.nan, index=equity_index)

    # CPI YoY and 3-month momentum
    if "CPI" in macro_daily.columns:
        cpi     = macro_daily["CPI"].ffill()
        cpi_yoy = cpi.pct_change(252) * 100
        cpi_mom = cpi.pct_change(63)        # 3-month change
    else:
        cpi_yoy = pd.Series(np.nan, index=equity_index)
        cpi_mom = pd.Series(np.nan, index=equity_index)

    # Assign regimes — priority order:
    # 1. Bear/Risk-off: large equity drawdown OR high VIX
    # 2. Inflation: high and rising CPI
    # 3. Bull: strong equity trend
    # 4. Neutral: everything else
    regime = pd.Series("Neutral", index=equity_index)

    bull_mask = ret_12m > BULL_THRESHOLD
    bear_mask = (ret_12m < BEAR_THRESHOLD) | (vix > VIX_THRESHOLD)
    infl_mask = (cpi_yoy > CPI_THRESHOLD) & (cpi_mom > 0)

    regime[bull_mask] = "Equity Bull"
    regime[infl_mask] = "Inflation"    # Inflation overrides bull
    regime[bear_mask] = "Equity Bear"  # Bear overrides everything

    return regime, ret_12m, vix, cpi_yoy


# =========================================================
# Compute regime statistics
# =========================================================
def regime_stats(
    equity: pd.Series,
    weights: pd.DataFrame,
    regime: pd.Series,
) -> pd.DataFrame:
    rets   = equity.pct_change().dropna()

    # Align everything to the intersection of equity and regime indices
    common = rets.index.intersection(regime.index)
    rets   = rets.loc[common]
    reg    = regime.loc[common]
    eq_aligned = equity.reindex(common).ffill().dropna()

    records = []
    for name in ["Equity Bull", "Equity Bear", "Inflation", "Neutral"]:
        mask = reg == name
        if mask.sum() < 20:
            continue

        r       = rets[mask]
        eq_reg  = eq_aligned[mask]
        n_days  = int(mask.sum())
        n_years = n_days / 252

        cagr    = float((1 + r).prod() ** (1 / n_years) - 1) \
                  if n_years > 0 else np.nan
        ann_vol = float(r.std(ddof=1) * np.sqrt(252))
        sharpe  = cagr / ann_vol if ann_vol > 0 else np.nan

        # Drawdown computed on the equity sub-series for that regime
        roll_max = eq_reg.cummax()
        max_dd   = float(((eq_reg - roll_max) / roll_max).min())

        win_rate = float((r > 0).mean())

        # Average ES weight during regime
        if weights is not None and "ES=F" in weights.columns:
            wt_common = weights.index.intersection(mask[mask].index)
            avg_es_wt = float(weights.loc[wt_common, "ES=F"].mean()) \
                        if len(wt_common) > 0 else np.nan
        else:
            avg_es_wt = np.nan

        records.append({
            "Regime":        name,
            "Days":          n_days,
            "Pct of sample": n_days / len(rets) * 100,
            "CAGR":          cagr,
            "Ann. vol":      ann_vol,
            "Sharpe":        sharpe,
            "Max drawdown":  max_dd,
            "Win rate":      win_rate,
            "Avg ES weight": avg_es_wt,
        })

    return pd.DataFrame(records).set_index("Regime")


# =========================================================
# Chart 1 — Regime timeline
# =========================================================
def plot_regime_timeline(
    equity: pd.Series,
    regime: pd.Series,
    ret_12m: pd.Series,
    vix: pd.Series,
    cpi_yoy: pd.Series,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    # Panel 1: equity curve with regime shading
    ax = axes[0]
    ax.plot(equity.index, equity.values / equity.iloc[0],
            color="#1a1a1a", lw=1.4, zorder=5)

    _shade_regimes(ax, regime, equity.index)
    ax.set_yscale("log")
    ax.set_ylabel("Portfolio value\n(log scale)", labelpad=6, fontsize=8)
    ax.set_title("Strategy equity curve with macro regime overlay", pad=8)

    # Panel 2: S&P 500 12-month return
    ax2 = axes[1]
    ax2.plot(ret_12m.index, ret_12m.values * 100,
             color="#2166ac", lw=1.2)
    ax2.axhline(BULL_THRESHOLD * 100, color="#2166ac", lw=0.8,
                linestyle="--", alpha=0.6,
                label=f"Bull threshold ({BULL_THRESHOLD*100:.0f}%)")
    ax2.axhline(BEAR_THRESHOLD * 100, color="#d6604d", lw=0.8,
                linestyle="--", alpha=0.6,
                label=f"Bear threshold ({BEAR_THRESHOLD*100:.0f}%)")
    ax2.axhline(0, color="#999999", lw=0.6)
    _shade_regimes(ax2, regime, ret_12m.index)
    ax2.set_ylabel("S&P 500\n12m return (%)", labelpad=6, fontsize=8)
    ax2.legend(fontsize=7, framealpha=0.5, loc="upper left")

    # Panel 3: VIX
    ax3 = axes[2]
    ax3.plot(vix.index, vix.values,
             color="#756bb1", lw=1.0)
    ax3.axhline(VIX_THRESHOLD, color="#d6604d", lw=0.8,
                linestyle="--", alpha=0.6,
                label=f"Bear threshold (VIX={VIX_THRESHOLD})")
    _shade_regimes(ax3, regime, vix.index)
    ax3.set_ylabel("VIX", labelpad=6, fontsize=8)
    ax3.legend(fontsize=7, framealpha=0.5, loc="upper right")

    # Panel 4: CPI YoY
    ax4 = axes[3]
    ax4.plot(cpi_yoy.index, cpi_yoy.values,
             color="#e6a817", lw=1.2)
    ax4.axhline(CPI_THRESHOLD, color="#e6a817", lw=0.8,
                linestyle="--", alpha=0.6,
                label=f"Inflation threshold ({CPI_THRESHOLD}%)")
    ax4.axhline(0, color="#999999", lw=0.6)
    _shade_regimes(ax4, regime, cpi_yoy.index)
    ax4.set_ylabel("CPI YoY (%)", labelpad=6, fontsize=8)
    ax4.legend(fontsize=7, framealpha=0.5)

    # Regime legend
    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.25, label=r)
        for r in REGIME_COLORS
    ]
    fig.legend(
        handles=patches, loc="lower center",
        ncol=4, fontsize=8, framealpha=0.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Macro regime classification, 2003–2026\n"
        "Bear priority > Inflation > Bull > Neutral",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_regime_timeline.png")


def _shade_regimes(ax, regime, index):
    """Shade background by regime for a given axis."""
    common = regime.index.intersection(index)
    reg    = regime.loc[common]

    current_regime = None
    start_date     = None

    for date, r in reg.items():
        if r != current_regime:
            if current_regime is not None and start_date is not None:
                ax.axvspan(start_date, date,
                           color=REGIME_COLORS.get(current_regime, "#cccccc"),
                           alpha=0.12, zorder=0)
            current_regime = r
            start_date     = date

    if current_regime is not None and start_date is not None:
        ax.axvspan(start_date, common[-1],
                   color=REGIME_COLORS.get(current_regime, "#cccccc"),
                   alpha=0.12, zorder=0)


# =========================================================
# Chart 2 — Regime performance bar charts
# =========================================================
def plot_regime_performance(stats: pd.DataFrame) -> None:
    metrics = ["CAGR", "Sharpe", "Max drawdown", "Win rate"]
    labels  = {
        "CAGR":         "Annualized CAGR",
        "Sharpe":       "Annualized Sharpe ratio",
        "Max drawdown": "Maximum drawdown",
        "Win rate":     "Daily win rate",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes_flat = axes.flatten()

    regimes = stats.index.tolist()
    colors  = [REGIME_COLORS.get(r, "#888888") for r in regimes]
    x       = np.arange(len(regimes))

    for ax, metric in zip(axes_flat, metrics):
        vals = stats[metric].values.astype(float)

        if metric in ("CAGR", "Max drawdown", "Win rate"):
            display_vals = vals * 100
            fmt          = "{:.1f}%"
        else:
            display_vals = vals
            fmt          = "{:.2f}"

        bars = ax.bar(x, display_vals, color=colors,
                      width=0.55, edgecolor="white")

        for bar, val in zip(bars, display_vals):
            if np.isfinite(val):
                ypos = val + abs(val) * 0.03 if val >= 0 \
                       else val - abs(val) * 0.06
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ypos, fmt.format(val),
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold",
                )

        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(regimes, fontsize=8, rotation=15, ha="right")
        ax.set_title(labels[metric], pad=8, fontsize=9)

    # Add day count as subtitle on each bar
    ax_cagr = axes_flat[0]
    for i, (regime, row) in enumerate(stats.iterrows()):
        ax_cagr.text(
            i, ax_cagr.get_ylim()[0] * 0.98,
            f"n={int(row['Days'])}d\n({row['Pct of sample']:.0f}%)",
            ha="center", va="top", fontsize=7, color="#555555",
        )

    fig.suptitle(
        "Strategy performance by macro regime\n"
        "Common analysis period: 2003-08-18 to 2026",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_regime_performance.png")


# =========================================================
# Chart 3 — Regime allocation heatmap
# =========================================================
def plot_regime_allocation(
    weights: pd.DataFrame,
    regime: pd.Series,
) -> None:
    if weights is None:
        print("  Skipping allocation chart — weights not available.")
        return

    symbols = weights.columns.tolist()
    common  = weights.index.intersection(regime.index)
    wt      = weights.loc[common]
    reg     = regime.loc[common]

    records = []
    for r in ["Equity Bull", "Equity Bear", "Inflation", "Neutral"]:
        mask = reg == r
        if mask.sum() < 10:
            continue
        row = {"Regime": r}
        for sym in symbols:
            row[sym] = float(wt.loc[mask, sym].mean())
        records.append(row)

    df_alloc = pd.DataFrame(records).set_index("Regime")

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(
        df_alloc.values.astype(float),
        cmap="RdYlGn", aspect="auto",
        vmin=0, vmax=df_alloc.values.max(),
    )

    ax.set_xticks(range(len(symbols)))
    ax.set_xticklabels(
        [s.replace("=F", "") for s in symbols],
        fontsize=9,
    )
    ax.set_yticks(range(len(df_alloc)))
    ax.set_yticklabels(df_alloc.index, fontsize=9)

    for i in range(len(df_alloc)):
        for j in range(len(symbols)):
            val = df_alloc.values[i, j]
            ax.text(j, i, f"{val:.1%}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color="black" if val < 0.4 else "white")

    plt.colorbar(im, ax=ax, label="Average weight")
    ax.set_title(
        "Average asset allocation by macro regime\n"
        "Signal weights — strategy baseline",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_allocation.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_regime_allocation.png")


# =========================================================
# Chart 4 — Cumulative return by regime
# =========================================================
def plot_cumulative_by_regime(
    equity: pd.Series,
    regime: pd.Series,
) -> None:
    rets   = equity.pct_change().dropna()
    common = rets.index.intersection(regime.index)
    rets   = rets.loc[common]
    reg    = regime.loc[common]

    fig, ax = plt.subplots(figsize=(11, 5))

    for r_name, color in REGIME_COLORS.items():
        mask         = reg == r_name
        regime_rets  = rets.copy()
        regime_rets[~mask] = 0.0
        cum          = (1 + regime_rets).cumprod()
        ax.plot(cum.index, cum.values,
                color=color, lw=1.6,
                label=f"{r_name} ({mask.sum()}d)")

    ax.axhline(1, color="#cccccc", lw=0.8, linestyle="--")
    ax.set_ylabel("Cumulative return contribution (start = 1)", labelpad=8)
    ax.set_title(
        "Cumulative return contribution by macro regime\n"
        "Each line shows compounded return earned only during that regime",
        pad=10,
    )
    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_cumulative.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_regime_cumulative.png")


# =========================================================
# Main
# =========================================================
def main():
    print("=" * 60)
    print("RR Strategy — Regime Decomposition Analysis")
    print("=" * 60)

    # Load strategy outputs
    print("\nLoading strategy outputs...")

    equity_path  = TABLES_DIR / "rebased_equity_curves.csv"
    weights_path = TABLES_DIR / "weights_signal.csv"

    if not equity_path.exists():
        print(f"ERROR: {equity_path} not found.")
        print("Run strategy.py first to generate outputs.")
        return

    curves_df = pd.read_csv(equity_path, index_col=0, parse_dates=True)
    print(f"  Equity curves loaded: {curves_df.shape}")
    print(f"  Columns: {curves_df.columns.tolist()}")

    # Find the portfolio column
    port_col = None
    for candidate in ["Final portfolio", "portfolio", "Portfolio"]:
        if candidate in curves_df.columns:
            port_col = candidate
            break
    if port_col is None:
        # Take first column that is not a known benchmark
        for col in curves_df.columns:
            if "benchmark" not in col.lower() and "es=f" not in col.lower():
                port_col = col
                break
    if port_col is None:
        port_col = curves_df.columns[0]

    equity = curves_df[port_col].dropna()
    print(f"  Using equity column: '{port_col}'")
    print(f"  Equity range: {equity.index[0].date()} to {equity.index[-1].date()}")

    # Load weights
    weights = None
    if weights_path.exists():
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        print(f"  Weights loaded: {weights.shape}")
    else:
        print("  Warning: weights_signal.csv not found")

    # Download macro data
    start = equity.index[0].strftime("%Y-%m-%d")
    end   = equity.index[-1].strftime("%Y-%m-%d")
    macro = download_macro(start, end)

    # Build regime series
    print("\nBuilding regime classifications...")
    regime, ret_12m, vix, cpi_yoy = build_regimes(macro, equity.index)

    # Regime summary
    print("\nRegime distribution:")
    for r_name in ["Equity Bull", "Equity Bear", "Inflation", "Neutral"]:
        n    = (regime == r_name).sum()
        pct  = n / len(regime) * 100
        print(f"  {r_name:15s}: {n:5d} days ({pct:.1f}%)")

    # Compute statistics
    print("\nComputing regime statistics...")
    stats = regime_stats(equity, weights, regime)

    print("\n" + "=" * 60)
    print("REGIME PERFORMANCE SUMMARY")
    print("=" * 60)
    print(stats[[
        "Days", "Pct of sample", "CAGR",
        "Sharpe", "Max drawdown", "Win rate", "Avg ES weight",
    ]].to_string(
        float_format=lambda x: f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    ))

    # Save CSV
    stats.to_csv(OUT_DIR / "regime_stats.csv")
    print(f"\nStats saved to: {OUT_DIR / 'regime_stats.csv'}")

    # Build charts
    print("\nBuilding charts...")
    plot_regime_timeline(equity, regime, ret_12m, vix, cpi_yoy)
    plot_regime_performance(stats)
    plot_regime_allocation(weights, regime)
    plot_cumulative_by_regime(equity, regime)

    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()