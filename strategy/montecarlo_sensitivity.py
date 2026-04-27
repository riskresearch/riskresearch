"""
Monte Carlo parameter sensitivity — Latin Hypercube sampling
=============================================================
Samples 100 combinations from the 6-parameter design space.
All combinations are evaluated over the SAME analysis period
(clipped to the common start date of the slowest combination)
to ensure a fair apples-to-apples comparison.

Run from the project root:
    python strategy/montecarlo_sensitivity.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import strategy as st

# =========================================================
# Output directory
# =========================================================
OUT_DIR = Path(__file__).parent / "outputs" / "montecarlo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Parameter grids (5 levels each)
# =========================================================
PARAM_GRIDS = {
    "VOL_WINDOW":          [11, 16, 21, 26, 32],
    "SORTINO_WINDOW":      [126, 189, 252, 315, 378],
    "IR_WINDOW":           [126, 189, 252, 315, 378],
    "TACTICAL_MAX_WEIGHT": [0.25, 0.375, 0.50, 0.625, 0.75],
    "ALLOC_SMOOTH_SPAN":   [5, 8, 10, 13, 15],
    "NO_TRADE_BAND":       [0.05, 0.075, 0.10, 0.125, 0.15],
}

PARAM_DISPLAY = {
    "VOL_WINDOW":          "Vol window",
    "SORTINO_WINDOW":      "Sortino window",
    "IR_WINDOW":           "IR window",
    "TACTICAL_MAX_WEIGHT": "Tactical cap",
    "ALLOC_SMOOTH_SPAN":   "EMA span",
    "NO_TRADE_BAND":       "No-trade band",
}

BASELINE = {
    "VOL_WINDOW":          21,
    "SORTINO_WINDOW":      252,
    "IR_WINDOW":           252,
    "TACTICAL_MAX_WEIGHT": 0.50,
    "ALLOC_SMOOTH_SPAN":   10,
    "NO_TRADE_BAND":       0.10,
}

# Slowest combination — determines common start date
SLOWEST = {
    "VOL_WINDOW":          32,
    "SORTINO_WINDOW":      378,
    "IR_WINDOW":           378,
    "TACTICAL_MAX_WEIGHT": 0.50,
    "ALLOC_SMOOTH_SPAN":   15,
    "NO_TRADE_BAND":       0.10,
}

N_SAMPLES        = 50
RANDOM_SEED      = 42
SHARPE_THRESHOLD = 0.70
BASELINE_SHARPE  = 0.807


# =========================================================
# Latin Hypercube sampler
# =========================================================
def latin_hypercube_sample(param_grids, n_samples, seed):
    rng        = np.random.default_rng(seed)
    param_keys = list(param_grids.keys())
    n_params   = len(param_keys)

    lhs_indices = np.zeros((n_samples, n_params), dtype=int)
    for j in range(n_params):
        n_levels = len(param_grids[param_keys[j]])
        repeats  = n_samples // n_levels
        extra    = n_samples % n_levels
        base     = list(range(n_levels)) * repeats + list(range(extra))
        lhs_indices[:, j] = rng.permutation(base)

    samples = []
    for i in range(n_samples):
        combo = {
            key: param_grids[key][int(lhs_indices[i, j])]
            for j, key in enumerate(param_keys)
        }
        samples.append(combo)
    return samples


# =========================================================
# Apply / restore parameters
# =========================================================
def apply_params(combo):
    st.VOL_WINDOW       = int(combo["VOL_WINDOW"])
    st.SORTINO_WINDOW   = int(combo["SORTINO_WINDOW"])
    st.SORTINO_Z_WINDOW = int(combo["SORTINO_WINDOW"])
    st.IR_WINDOW        = int(combo["IR_WINDOW"])
    st.TACTICAL_MAX_WEIGHT                  = float(combo["TACTICAL_MAX_WEIGHT"])
    st.ALLOC_SMOOTH_SPAN                    = int(combo["ALLOC_SMOOTH_SPAN"])
    st.ALLOC_SMOOTH_WINDOW                  = int(combo["ALLOC_SMOOTH_SPAN"])
    st.MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY = float(combo["NO_TRADE_BAND"])


def restore_baseline():
    st.VOL_WINDOW       = BASELINE["VOL_WINDOW"]
    st.SORTINO_WINDOW   = BASELINE["SORTINO_WINDOW"]
    st.SORTINO_Z_WINDOW = BASELINE["SORTINO_WINDOW"]
    st.IR_WINDOW        = BASELINE["IR_WINDOW"]
    st.TACTICAL_MAX_WEIGHT                  = BASELINE["TACTICAL_MAX_WEIGHT"]
    st.ALLOC_SMOOTH_SPAN                    = BASELINE["ALLOC_SMOOTH_SPAN"]
    st.ALLOC_SMOOTH_WINDOW                  = BASELINE["ALLOC_SMOOTH_SPAN"]
    st.MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY = BASELINE["NO_TRADE_BAND"]


# =========================================================
# Calibrate common start date
# =========================================================
def calibrate_common_start(data):
    """
    Run the slowest combination to find the latest possible
    strategy_start_ts. All combinations will be clipped to
    this date before computing statistics.
    """
    print("  Running slowest combination to calibrate common start date...")
    apply_params(SLOWEST)
    try:
        signals     = st.build_signals(data)
        asset_layer = st.build_asset_leverage(data, signals)
        allocations = st.build_allocations(data, signals, asset_layer)
        sim_results = st.run_simulation(data, signals, asset_layer, allocations)
        analysis    = st.build_analysis(
            data, signals, asset_layer, allocations, sim_results
        )
        common_start = analysis["strategy_start_ts"]
        print(f"  Common analysis start date: {common_start.date()}")
        return common_start
    finally:
        restore_baseline()


# =========================================================
# Compute stats from equity curve clipped to common start
# =========================================================
def compute_stats(eq: pd.Series, common_start: pd.Timestamp) -> dict:
    eq = eq.loc[eq.index >= common_start].dropna()
    if len(eq) < 50:
        return {k: np.nan for k in
                ["cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar"]}

    rets    = eq.pct_change().dropna()
    n_years = len(eq) / 252
    cagr    = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(252))
    sharpe  = cagr / ann_vol if ann_vol > 0 else np.nan

    downside = rets[rets < 0]
    down_vol = (float(downside.std(ddof=1) * np.sqrt(252))
                if len(downside) > 1 else np.nan)
    sortino  = cagr / down_vol if down_vol and down_vol > 0 else np.nan

    roll_max = eq.cummax()
    max_dd   = float(((eq - roll_max) / roll_max).min())
    calmar   = cagr / abs(max_dd) if max_dd and max_dd != 0 else np.nan

    return {"cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe,
            "sortino": sortino, "max_dd": max_dd, "calmar": calmar}


# =========================================================
# Run one combination
# =========================================================
def run_combo(data, combo, run_id, common_start):
    apply_params(combo)
    try:
        signals     = st.build_signals(data)
        asset_layer = st.build_asset_leverage(data, signals)
        allocations = st.build_allocations(data, signals, asset_layer)
        sim_results = st.run_simulation(data, signals, asset_layer, allocations)
        analysis    = st.build_analysis(
            data, signals, asset_layer, allocations, sim_results
        )

        eq_raw  = analysis["portfolio_plot"]
        stats   = compute_stats(eq_raw, common_start)

        # Normalised equity curve clipped to common start
        eq_clip = eq_raw.loc[eq_raw.index >= common_start].dropna()
        eq_norm = eq_clip / eq_clip.iloc[0] if len(eq_clip) > 0 else None

        result = {
            "run_id":   run_id,
            **stats,
            "status":   "ok",
            "eq_curve": eq_norm,
        }

    except Exception as e:
        result = {
            "run_id":   run_id,
            "cagr": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
            "sortino": np.nan, "max_dd": np.nan, "calmar": np.nan,
            "status":   f"error: {e}",
            "eq_curve": None,
        }
    finally:
        restore_baseline()

    return {**combo, **result}


# =========================================================
# Charts
# =========================================================
def setup_style():
    plt.rcParams.update({
        "figure.dpi": 150, "font.family": "serif", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })


def plot_sharpe_histogram(ok, common_start):
    sharpes = ok["sharpe"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.5))

    counts, edges, patches = ax.hist(
        sharpes, bins=20, color="#2166ac",
        edgecolor="white", alpha=0.85,
    )
    for patch, left in zip(patches, edges[:-1]):
        if left < SHARPE_THRESHOLD:
            patch.set_facecolor("#d6604d")

    ax.axvline(BASELINE_SHARPE, color="#1a1a1a", lw=1.8,
               linestyle="--", label=f"Baseline ({BASELINE_SHARPE:.3f})")
    ax.axvline(SHARPE_THRESHOLD, color="#d6604d", lw=1.2,
               linestyle=":", label=f"Threshold ({SHARPE_THRESHOLD:.2f})")

    pct_above = (sharpes >= SHARPE_THRESHOLD).mean() * 100
    ax.text(0.97, 0.92,
            f"{pct_above:.0f}% of combinations\nabove Sharpe {SHARPE_THRESHOLD}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))

    ax.set_xlabel("Annualized Sharpe ratio", labelpad=8)
    ax.set_ylabel("Number of combinations", labelpad=8)
    ax.set_title(
        f"Distribution of Sharpe ratios across {len(ok)} combinations\n"
        f"Common analysis start: {common_start.date()} | "
        f"Mean={sharpes.mean():.3f}  Median={sharpes.median():.3f}  "
        f"Std={sharpes.std():.3f}  Min={sharpes.min():.3f}  "
        f"Max={sharpes.max():.3f}",
        pad=12, fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sharpe_histogram.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_sharpe_histogram.png")


def plot_sharpe_vs_drawdown(ok):
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(
        ok["max_dd"] * 100, ok["sharpe"],
        c=ok["calmar"], cmap="RdYlGn",
        s=35, alpha=0.75, edgecolors="white", linewidths=0.5,
    )
    baseline_row = ok[
        (ok["SORTINO_WINDOW"] == 252) &
        (ok["IR_WINDOW"] == 252) &
        (ok["TACTICAL_MAX_WEIGHT"] == 0.5) &
        (ok["VOL_WINDOW"] == 21)
    ]
    if not baseline_row.empty:
        ax.scatter(
            baseline_row["max_dd"].values[0] * 100,
            baseline_row["sharpe"].values[0],
            s=120, color="#1a1a1a", zorder=6,
            marker="D", label="Baseline",
        )
    plt.colorbar(sc, ax=ax, label="Calmar ratio")
    ax.axhline(SHARPE_THRESHOLD, color="#d6604d", lw=0.8,
               linestyle=":", label=f"Sharpe threshold ({SHARPE_THRESHOLD})")
    ax.set_xlabel("Maximum drawdown (%)", labelpad=8)
    ax.set_ylabel("Annualized Sharpe ratio", labelpad=8)
    ax.set_title(
        "Sharpe vs maximum drawdown — common analysis period\n"
        "Color = Calmar ratio. Diamond = baseline.",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sharpe_vs_drawdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_sharpe_vs_drawdown.png")


def plot_best_worst(ok):
    param_cols   = list(PARAM_GRIDS.keys())
    display_cols = [PARAM_DISPLAY[p] for p in param_cols]
    fig, axes    = plt.subplots(2, 1, figsize=(13, 8))

    for ax_idx, (subset, title, color) in enumerate([
        (ok.nlargest(10, "sharpe"),
         "Top 10 combinations by Sharpe ratio",    "#2166ac"),
        (ok.nsmallest(10, "sharpe"),
         "Bottom 10 combinations by Sharpe ratio", "#d6604d"),
    ]):
        ax = axes[ax_idx]
        ax.axis("off")
        rows = [
            [str(row[p]) for p in param_cols] + [
                f"{row['sharpe']:.3f}", f"{row['cagr']:.1%}",
                f"{row['max_dd']:.1%}", f"{row['calmar']:.3f}",
            ]
            for _, row in subset.iterrows()
        ]
        col_labels = display_cols + ["Sharpe", "CAGR", "Max DD", "Calmar"]
        col_widths = [0.10] * len(param_cols) + [0.08, 0.07, 0.08, 0.08]

        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       colWidths=col_widths)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.8)

        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor(color)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("white")

        bg = "#f0f4ff" if ax_idx == 0 else "#fff0f0"
        for i in range(1, len(rows) + 1):
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(bg if i % 2 == 0 else "white")
                tbl[i, j].set_edgecolor("#eeeeee")

        ax.set_title(title, fontsize=9, pad=10, loc="left", color=color)

    fig.suptitle("Best and worst parameter combinations\n"
                 "Monte Carlo Latin Hypercube — common analysis period",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_best_worst.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  Saved: fig_best_worst.png")


def plot_parameter_boxplots(ok):
    param_keys = list(PARAM_GRIDS.keys())
    fig, axes  = plt.subplots(2, 3, figsize=(13, 7))
    axes_flat  = axes.flatten()

    for idx, param in enumerate(param_keys):
        ax     = axes_flat[idx]
        levels = sorted(ok[param].unique())
        data_by_level = [
            ok[ok[param] == lvl]["sharpe"].dropna().values
            for lvl in levels
        ]
        bp = ax.boxplot(
            data_by_level, patch_artist=True,
            medianprops=dict(color="#1a1a1a", lw=1.5),
            whiskerprops=dict(color="#666666"),
            capprops=dict(color="#666666"),
            flierprops=dict(marker="o", markerfacecolor="#d6604d",
                            markersize=4, alpha=0.6),
        )
        box_colors = ["#d6604d", "#f4a582", "#ffffff", "#92c5de", "#2166ac"]
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        bl_val = BASELINE[param]
        if bl_val in levels:
            ax.axvline(levels.index(bl_val) + 1,
                       color="#888888", lw=0.8,
                       linestyle="--", alpha=0.6)
        ax.axhline(BASELINE_SHARPE, color="#888888",
                   lw=0.6, linestyle=":", alpha=0.5)
        ax.set_xticks(range(1, len(levels) + 1))
        ax.set_xticklabels([str(v) for v in levels],
                           fontsize=7.5, rotation=20, ha="right")
        ax.set_title(PARAM_DISPLAY[param], fontsize=9, pad=6)
        ax.set_ylabel("Sharpe ratio", fontsize=7.5, labelpad=4)

    fig.suptitle(
        "Sharpe ratio distribution by parameter level — common analysis period\n"
        "Each box shows distribution when that parameter takes the given value "
        "(all others vary freely).\n"
        "Dashed vertical = baseline level. Dotted horizontal = baseline Sharpe.",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_parameter_boxplots.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_parameter_boxplots.png")


def plot_equity_paths(records, common_start):
    curves  = []
    sharpes = []
    is_bl   = []

    for r in records:
        if r["status"] != "ok" or r["eq_curve"] is None:
            continue
        curves.append(r["eq_curve"])
        sharpes.append(r["sharpe"])
        combo = {k: r[k] for k in PARAM_GRIDS}
        is_bl.append(all(combo[k] == BASELINE[k] for k in PARAM_GRIDS))

    if not curves:
        print("  No curves to plot.")
        return

    sharpe_arr = np.array(sharpes)
    vmin, vmax = sharpe_arr.min(), sharpe_arr.max()
    cmap       = plt.get_cmap("RdYlGn")
    norm       = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    ax1, ax2  = axes

    order = sorted(range(len(curves)), key=lambda i: 1 if is_bl[i] else 0)

    for i in order:
        eq    = curves[i]
        dd    = (eq / eq.cummax() - 1) * 100
        color = "#1a1a1a" if is_bl[i] else cmap(norm(sharpes[i]))
        alpha = 1.0 if is_bl[i] else 0.15
        lw    = 2.2 if is_bl[i] else 0.7
        zord  = 10  if is_bl[i] else 1

        ax1.plot(eq.index, eq.values,
                 color=color, alpha=alpha, lw=lw, zorder=zord)
        ax2.plot(eq.index, dd.values,
                 color=color, alpha=alpha, lw=lw, zorder=zord)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, pad=0.01).set_label("Sharpe ratio", fontsize=8)
    plt.colorbar(sm, ax=ax2, pad=0.01).set_label("Sharpe ratio", fontsize=8)

    # Percentile band
    ref_index   = curves[0].index
    all_eq_vals = np.array([
        c.reindex(ref_index, method="nearest").values
        for c in curves
    ])
    p5  = np.nanpercentile(all_eq_vals, 5,  axis=0)
    p95 = np.nanpercentile(all_eq_vals, 95, axis=0)
    ax1.fill_between(ref_index, p5, p95, alpha=0.08, color="#2166ac",
                     label="5th–95th percentile band")

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio value (start = 1, log scale)", labelpad=8)
    ax1.set_title(
        f"Monte Carlo equity paths — {len(curves)} combinations\n"
        f"Common start: {common_start.date()} | "
        "Black = baseline. Color = Sharpe (red = low, green = high).",
        pad=10,
    )
    ax1.legend(fontsize=8, framealpha=0.5, loc="upper left")

    ax2.axhline(0, color="#999999", lw=0.8, linestyle="--")
    ax2.set_ylabel("Drawdown from peak (%)", labelpad=8)
    ax2.set_title("Drawdown paths — common analysis period", pad=10)

    fig.suptitle(
        "Monte Carlo simulation — Latin Hypercube parameter sampling\n"
        "6 parameters × 5 levels — 100 combinations — common analysis period",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_equity_paths.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_equity_paths.png")


# =========================================================
# Main
# =========================================================
def main():
    t0 = time.time()
    setup_style()

    print("=" * 60)
    print("RR Strategy — Monte Carlo Sensitivity Analysis")
    print("Corrected: common analysis start date for all combinations")
    print(f"Latin Hypercube: {N_SAMPLES} combinations, 6 params × 5 levels")
    print("=" * 60)

    samples = latin_hypercube_sample(PARAM_GRIDS, N_SAMPLES, RANDOM_SEED)
    if BASELINE not in samples:
        samples[0] = BASELINE
        print("Note: baseline inserted at position 0.")

    print(f"\nSample coverage check:")
    for param, grid in PARAM_GRIDS.items():
        counts = {v: sum(1 for s in samples if s[param] == v) for v in grid}
        print(f"  {PARAM_DISPLAY[param]:22s}: "
              + "  ".join(f"{v}×{c}" for v, c in counts.items()))

    print(f"\nDownloading data (once)...")
    data = st.load_data()
    print(f"Data loaded: {len(data['close_df'])} rows, "
          f"symbols: {data['all_symbols']}")

    print(f"\nCalibrating common analysis start date...")
    common_start = calibrate_common_start(data)

    print(f"\nRunning {N_SAMPLES} combinations...")
    print(
        f"{'Run':>4}  {'Vol':>4}  {'Srt':>4}  {'IR':>4}  "
        f"{'Cap':>6}  {'EMA':>4}  {'Band':>6}  "
        f"{'Sharpe':>7}  {'CAGR':>7}  {'MaxDD':>8}  {'Time':>6}"
    )
    print("-" * 75)

    records = []

    for i, combo in enumerate(samples):
        t_run  = time.time()
        result = run_combo(data, combo, run_id=i + 1,
                           common_start=common_start)
        elapsed = time.time() - t_run

        if result["status"] == "ok":
            line = (f"Sharpe={result['sharpe']:.3f}  "
                    f"CAGR={result['cagr']:.1%}  "
                    f"MaxDD={result['max_dd']:.1%}")
        else:
            line = f"FAILED: {result['status']}"

        print(
            f"{i+1:>4}  "
            f"{combo['VOL_WINDOW']:>4}  "
            f"{combo['SORTINO_WINDOW']:>4}  "
            f"{combo['IR_WINDOW']:>4}  "
            f"{combo['TACTICAL_MAX_WEIGHT']:>6.3f}  "
            f"{combo['ALLOC_SMOOTH_SPAN']:>4}  "
            f"{combo['NO_TRADE_BAND']:>6.3f}  "
            f"  {line}  ({elapsed:.1f}s)"
        )

        records.append(result)

        if (i + 1) % 10 == 0:
            df_p = pd.DataFrame([
                {k: v for k, v in r.items() if k != "eq_curve"}
                for r in records
            ])
            df_p.to_csv(OUT_DIR / "montecarlo_results_partial.csv",
                        index=False)
            elapsed_total = time.time() - t0
            remaining     = (elapsed_total / (i + 1)) * (N_SAMPLES - i - 1)
            print(f"  --- {i+1}/{N_SAMPLES} | "
                  f"elapsed {elapsed_total/60:.1f}min | "
                  f"est. remaining {remaining/60:.1f}min ---")

    # Save CSV
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "eq_curve"}
        for r in records
    ])
    csv_path = OUT_DIR / "montecarlo_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    ok      = df[df["status"] == "ok"]
    sharpes = ok["sharpe"].dropna()

    print("\n" + "=" * 60)
    print("MONTE CARLO SUMMARY — COMMON ANALYSIS PERIOD")
    print(f"Common start: {common_start.date()}")
    print("=" * 60)
    print(f"Successful runs:      {len(ok)} / {len(df)}")
    print(f"Sharpe mean:          {sharpes.mean():.3f}")
    print(f"Sharpe median:        {sharpes.median():.3f}")
    print(f"Sharpe std:           {sharpes.std():.3f}")
    print(f"Sharpe min:           {sharpes.min():.3f}")
    print(f"Sharpe max:           {sharpes.max():.3f}")
    print(f"5th percentile:       {sharpes.quantile(0.05):.3f}")
    print(f"95th percentile:      {sharpes.quantile(0.95):.3f}")
    print(f"% above {SHARPE_THRESHOLD}:        "
          f"{(sharpes >= SHARPE_THRESHOLD).mean()*100:.0f}%")
    print(f"Baseline percentile:  "
          f"{(sharpes < BASELINE_SHARPE).mean()*100:.0f}th")

    param_keys = list(PARAM_GRIDS.keys())
    print(f"\nCorrelation with Sharpe:")
    for p in param_keys:
        print(f"  {PARAM_DISPLAY[p]:22s}: {ok[p].corr(ok['sharpe']):+.3f}")

    print(f"\nCorrelation with Max Drawdown:")
    for p in param_keys:
        print(f"  {PARAM_DISPLAY[p]:22s}: {ok[p].corr(ok['max_dd']):+.3f}")

    top5_cols = param_keys + ["sharpe", "cagr", "max_dd", "calmar"]
    print(f"\nTop 5 combinations:")
    print(ok.nlargest(5, "sharpe")[top5_cols].to_string(index=False))
    print(f"\nBottom 5 combinations:")
    print(ok.nsmallest(5, "sharpe")[top5_cols].to_string(index=False))

    print(f"\nSortino × IR window interaction (mean Sharpe):")
    pivot = ok.groupby(
        ["SORTINO_WINDOW", "IR_WINDOW"]
    )["sharpe"].mean().unstack()
    print(pivot.round(3).to_string())

    print(f"\nBuilding charts...")
    ok_records = [r for r in records if r["status"] == "ok"]
    plot_sharpe_histogram(ok, common_start)
    plot_sharpe_vs_drawdown(ok)
    plot_best_worst(ok)
    plot_parameter_boxplots(ok)
    plot_equity_paths(ok_records, common_start)

    total_time = time.time() - t0
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"All outputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()