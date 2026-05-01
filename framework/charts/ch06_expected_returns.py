import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import pandas_datareader.data as web
from datetime import datetime

# =========================================================
# Paths
# =========================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "ch06"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHILLER_FILE = DATA_DIR / "shiller_ie_data.xls"

# =========================================================
# Style
# =========================================================
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

COLORS = {
    "yield":      "#2166ac",
    "growth":     "#4dac26",
    "inflation":  "#d6604d",
    "multiple":   "#756bb1",
    "total":      "#1a1a1a",
    "equity":     "#2166ac",
    "bond":       "#4dac26",
    "real_rate":  "#d6604d",
    "neutral":    "#888888",
    "cape":       "#d6604d",
    "margin":     "#2166ac",
}


# =========================================================
# Load Shiller data
# =========================================================
def load_shiller() -> pd.DataFrame:
    df = pd.read_excel(SHILLER_FILE, sheet_name="Data", header=7)
    df = df.iloc[:, :7].copy()
    df.columns = ["date_raw", "price", "dividend", "earnings",
                  "cpi", "col5", "cape"]
    df = df[pd.to_numeric(df["date_raw"], errors="coerce").notna()].copy()
    df["date_raw"] = df["date_raw"].astype(float).round(2)
    year_int  = df["date_raw"].astype(int)
    month_int = ((df["date_raw"] - year_int) * 100).round(0).astype(int).clip(1, 12)
    df["date"] = pd.to_datetime(dict(year=year_int, month=month_int, day=1))
    df = df.set_index("date").sort_index()
    for col in ["price", "dividend", "earnings", "cpi", "cape"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["price", "dividend", "cpi"])
    df = df[~df.index.duplicated(keep="first")]
    return df


# =========================================================
# Download helper
# =========================================================
def fred(series: str, start: datetime, end: datetime) -> pd.Series:
    return web.DataReader(series, "fred", start, end).squeeze()


# =========================================================
# Chart 1 — Four-regime return decomposition
# =========================================================
def chart_regime_decomposition(df: pd.DataFrame) -> None:
    """
    Decompose S&P 500 returns into yield, real EPS growth,
    inflation, and multiple change across four regimes.
    Also show realized inflation and starting CAPE for each regime.
    """
    regimes = [
        ("1966–1981", "1966-01-01", "1981-12-01"),
        ("1982–1999", "1982-01-01", "1999-12-01"),
        ("2000–2009", "2000-01-01", "2009-12-01"),
        ("2010–2021", "2010-01-01", "2021-12-01"),
    ]

    rows = []
    for name, start, end in regimes:
        sub = df.loc[start:end].dropna(
            subset=["price", "dividend", "earnings", "cpi", "cape"]
        )
        if len(sub) < 24:
            continue

        n_years  = len(sub) / 12
        p0       = float(sub["price"].iloc[0])
        p1       = float(sub["price"].iloc[-1])
        d0       = float(sub["dividend"].iloc[0])
        e0       = float(sub["earnings"].iloc[0])
        e1       = float(sub["earnings"].iloc[-1])
        cpi0     = float(sub["cpi"].iloc[0])
        cpi1     = float(sub["cpi"].iloc[-1])
        cape0    = float(sub["cape"].iloc[0])

        div_yield   = (d0 / p0) * 100
        inflation   = ((cpi1 / cpi0) ** (1 / n_years) - 1) * 100
        real_e0     = e0 / cpi0
        real_e1     = e1 / cpi1
        if real_e0 > 0 and real_e1 > 0:
            real_eps = ((real_e1 / real_e0) ** (1 / n_years) - 1) * 100
        else:
            real_eps = 0.0

        total_price = ((p1 / p0) ** (1 / n_years) - 1) * 100
        total_nom   = total_price + div_yield
        multiple    = total_nom - div_yield - real_eps - inflation

        rows.append({
            "regime":          name,
            "Dividend yield":  div_yield,
            "Real EPS growth": real_eps,
            "Inflation":       inflation,
            "Multiple change": multiple,
            "Total":           total_nom,
            "Starting CAPE":   cape0,
            "Realized inflation": inflation,
        })

    decomp = pd.DataFrame(rows).set_index("regime")

    components  = ["Dividend yield", "Real EPS growth",
                   "Inflation", "Multiple change"]
    comp_colors = [COLORS["yield"], COLORS["growth"],
                   COLORS["inflation"], COLORS["multiple"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: stacked bar decomposition
    ax  = axes[0]
    x   = np.arange(len(decomp))
    bot = np.zeros(len(decomp))

    for comp, color in zip(components, comp_colors):
        vals = decomp[comp].values.astype(float)
        ax.bar(x, vals, bottom=bot, color=color,
               label=comp, width=0.55,
               edgecolor="white", linewidth=0.5)
        bot = bot + vals

    ax.plot(x, decomp["Total"].values.astype(float),
            "D", color=COLORS["total"], markersize=8,
            zorder=10, label="Total nominal return",
            markeredgecolor="white", markeredgewidth=1.2)

    # -------------------------------------------------------
    # Label annotation — same spatial-overlap logic as ch01.
    # Cap each positive segment's visible top by any subsequent
    # negative bar that spatially overlaps it, so labels land
    # inside the actually-visible colour band.
    # -------------------------------------------------------
    vals_matrix = np.array(
        [decomp[comp].values.astype(float) for comp in components]
    )
    n_comp = len(components)
    n_bars = len(decomp)

    cum_bot2 = np.zeros((n_comp + 1, n_bars))
    for ci in range(n_comp):
        cum_bot2[ci + 1] = cum_bot2[ci] + vals_matrix[ci]

    bottom2 = np.zeros(n_bars)
    for ci, (comp, color) in enumerate(zip(components, comp_colors)):
        vals = vals_matrix[ci]
        for i, (val, bot_i) in enumerate(zip(vals, bottom2)):
            if abs(val) <= 0.5:
                bottom2[i] += val
                continue

            x_center = x[i]

            if val >= 0:
                seg_bot_v = bot_i
                seg_top_v = bot_i + val
                for cj in range(ci + 1, n_comp):
                    v_j = vals_matrix[cj, i]
                    if v_j < 0:
                        neg_top = cum_bot2[cj,     i]
                        neg_bot = cum_bot2[cj + 1, i]
                        if neg_bot < seg_top_v and neg_top > seg_bot_v:
                            seg_top_v = max(seg_bot_v, min(seg_top_v, neg_bot))
                seg_mid = (seg_bot_v + seg_top_v) / 2
                seg_h   = seg_top_v - seg_bot_v
            else:
                seg_top_v = bot_i
                seg_bot_v = bot_i + val
                seg_mid   = (seg_bot_v + seg_top_v) / 2
                seg_h     = seg_top_v - seg_bot_v

            if seg_h >= 0.8:
                ax.text(x_center, seg_mid, f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=7.5, color="white",
                        fontweight="bold", zorder=10)
            else:
                offset_dir = 1 if val >= 0 else -1
                ax.text(x_center,
                        seg_mid + offset_dir * (seg_h * 0.5 + 0.25),
                        f"{val:.1f}%",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=7.0, color="#333333",
                        fontweight="bold", zorder=10)

        bottom2 = bottom2 + vals
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(decomp.index, fontsize=9)
    ax.set_ylabel("Annualized contribution (pp)", labelpad=8)
    ax.set_title("Return decomposition by regime\nS&P 500", pad=10)
    ax.legend(fontsize=7.5, framealpha=0.5, loc="upper right")

    # Right: starting CAPE and realized inflation table
    ax2 = axes[1]
    ax2.axis("off")

    table_data = []
    for regime in decomp.index:
        table_data.append([
            regime,
            f"{decomp.loc[regime, 'Starting CAPE']:.0f}x",
            f"{decomp.loc[regime, 'Realized inflation']:.1f}%",
            f"{decomp.loc[regime, 'Total']:.1f}%",
        ])

    col_labels = ["Regime", "Starting CAPE",
                  "Realized inflation\n(annualized)", "Total nominal\nreturn (ann.)"]
    tbl = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.28, 0.22, 0.28, 0.22],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.5)

    for j in range(4):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    for i in range(1, len(table_data) + 1):
        color = "#f7f7f7" if i % 2 == 0 else "white"
        for j in range(4):
            tbl[i, j].set_facecolor(color)
            tbl[i, j].set_edgecolor("#eeeeee")

    ax2.set_title("Regime summary statistics", pad=10, fontsize=9)

    fig.suptitle(
        "Four-regime decomposition of U.S. equity returns\n"
        "S&P 500, annualized. Source: Shiller (2023)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_regime_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_regime_decomposition.png")


# =========================================================
# Chart 2 — Starting bond yield vs subsequent return
# =========================================================
def chart_bond_yield_vs_return() -> None:
    """
    10-year Treasury yield at start of year vs
    realized 10-year annualized total return.
    Approximate bond total return from yield and price change.
    """
    start = datetime(1960, 1, 1)
    end   = datetime(2023, 12, 31)

    print("  Downloading Treasury yield from FRED...")
    try:
        gs10 = fred("GS10", start, end)
    except Exception as e:
        print(f"  Warning: FRED download failed: {e}")
        return

    gs10_m = gs10.resample("MS").last().dropna()

    # Approximate monthly bond return
    bond_ret = (-7.0 * gs10_m.diff() / 100 + gs10_m / 1200).dropna()

    # Compute 10-year forward annualized total return
    horizon = 120
    records = []
    for i in range(len(bond_ret) - horizon):
        start_y = float(gs10_m.iloc[i])
        rets_10y = bond_ret.iloc[i:i + horizon].values
        cum_ret  = float(np.prod(1 + rets_10y) ** (1 / 10) - 1)
        records.append({
            "date":      bond_ret.index[i],
            "yield_pct": start_y,
            "fwd_ret":   cum_ret * 100,
        })

    df_sc = pd.DataFrame(records).dropna()

    x      = df_sc["yield_pct"].values
    y      = df_sc["fwd_ret"].values
    coeffs = np.polyfit(x, y, 1)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    y_fit  = np.polyval(coeffs, x_fit)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    sc = ax.scatter(
        df_sc["yield_pct"], df_sc["fwd_ret"],
        c=df_sc["date"].apply(lambda d: d.year),
        cmap="viridis", s=10, alpha=0.6,
    )
    ax.plot(x_fit, y_fit, color=COLORS["real_rate"], lw=1.8,
            label=f"OLS fit (slope={coeffs[0]:.2f})")
    ax.axline((0, 0), slope=1, color="#cccccc", lw=0.8,
              linestyle=":", label="45-degree line")

    plt.colorbar(sc, ax=ax, label="Year")
    ax.set_xlabel("10-year Treasury yield at start (%)", labelpad=8)
    ax.set_ylabel("Subsequent 10-year annualized total return (%)", labelpad=8)
    ax.set_title("Starting yield and 10-year bond return\n1960–2015", pad=10)
    ax.legend(fontsize=8, framealpha=0.5)

    # Right: yield history with annotation
    ax2 = axes[1]
    ax2.fill_between(gs10_m.index, gs10_m.values,
                     color=COLORS["bond"], alpha=0.3)
    ax2.plot(gs10_m.index, gs10_m.values,
             color=COLORS["bond"], lw=1.4)

    # Annotate key periods
    for label, date_str, offset in [
        ("Peak\n~16%",    "1981-09-01", (0,  1.5)),
        ("Post-QE\ntrough\n~0.5%", "2020-08-01", (0, 1.0)),
    ]:
        dt  = pd.Timestamp(date_str)
        idx = gs10_m.index.get_indexer([dt], method="nearest")[0]
        if idx >= 0:
            val = float(gs10_m.iloc[idx])
            ax2.annotate(
                label,
                xy=(gs10_m.index[idx], val),
                xytext=(gs10_m.index[idx] - pd.DateOffset(years=4), val + offset[1]),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                color="#333333",
            )

    ax2.set_ylabel("10-year Treasury yield (%)", labelpad=8)
    ax2.set_title("10-year Treasury yield history\n1960–2023", pad=10)

    fig.suptitle(
        "Starting bond yield as a forward return anchor\n"
        "10-year U.S. Treasury. Source: FRED (GS10)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_bond_yield_vs_return.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_bond_yield_vs_return.png")


# =========================================================
# Chart 3 — Corporate profit margins BEA
# =========================================================
def chart_corporate_margins() -> None:
    """
    U.S. corporate profit margins from BEA national accounts.
    Series: A446RC1Q027SBEA — Corporate profits after tax as
    share of GNP. Quarterly, FRED.
    """
    start = datetime(1947, 1, 1)
    end   = datetime(2023, 12, 31)

    print("  Downloading corporate profit margins from FRED...")
    try:
        # Corporate profits after tax / GDP approximation
        profits = fred("CP",     start, end)   # Corporate profits after tax
        gdp     = fred("GDP",    start, end)   # Nominal GDP
    except Exception as e:
        print(f"  Warning: FRED download failed: {e}")
        return

    # Align and compute margin
    common  = profits.index.intersection(gdp.index)
    profits = profits.loc[common]
    gdp     = gdp.loc[common]
    margin  = (profits / gdp * 100).dropna()

    # Long-run average
    long_avg = float(margin.mean())

    # Rolling 4-quarter average
    roll_avg = margin.rolling(4, min_periods=3).mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.fill_between(margin.index, margin.values,
                    color=COLORS["margin"], alpha=0.2)
    ax.plot(margin.index, roll_avg.values,
            color=COLORS["margin"], lw=1.8,
            label="4-quarter rolling average")
    ax.axhline(long_avg, color=COLORS["real_rate"], lw=1.2,
               linestyle="--",
               label=f"Long-run average ({long_avg:.1f}%)")

    # Shade NBER recessions approximately
    recessions = [
        ("1973-11-01", "1975-03-01"),
        ("1980-01-01", "1980-07-01"),
        ("1981-07-01", "1982-11-01"),
        ("1990-07-01", "1991-03-01"),
        ("2001-03-01", "2001-11-01"),
        ("2007-12-01", "2009-06-01"),
        ("2020-02-01", "2020-04-01"),
    ]
    for r_start, r_end in recessions:
        ax.axvspan(pd.Timestamp(r_start), pd.Timestamp(r_end),
                   color="#eeeeee", alpha=0.7, zorder=0)

    # Annotate key episodes
    for label, date_str, offset in [
        ("Post-2008\nmargin expansion", "2012-01-01", (0,  1.5)),
        ("2022\ncompression",           "2022-06-01", (0, -1.8)),
    ]:
        dt  = pd.Timestamp(date_str)
        idx = roll_avg.index.get_indexer([dt], method="nearest")[0]
        if idx >= 0 and not pd.isna(roll_avg.iloc[idx]):
            val = float(roll_avg.iloc[idx])
            ax.annotate(
                label,
                xy=(roll_avg.index[idx], val),
                xytext=(roll_avg.index[idx] + pd.DateOffset(years=2),
                        val + offset[1]),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                color="#333333",
            )

    ax.set_ylabel("Corporate profits after tax / GDP (%)", labelpad=8)
    ax.set_title(
        "U.S. corporate profit margins, 1947–2023\n"
        "Corporate profits after tax as share of GDP. "
        "Grey shading = NBER recessions.",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_corporate_margins.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_corporate_margins.png")


# =========================================================
# Chart 4 — Scenario return table
# =========================================================
def chart_scenario_table() -> None:
    """
    Base, favorable, and adverse expected return scenarios
    for U.S. equities and investment-grade bonds.
    Hardcoded from current market conditions (early 2024).
    """
    # Current market inputs (approximate, early 2024)
    # Equity: S&P 500 dividend yield ~1.4%, CAPE ~31x
    # Bond: 10-yr Treasury ~4.3%

    scenarios = {
        "Asset": [
            "U.S. Equities", "U.S. Equities", "U.S. Equities",
            "IG Bonds (10yr)", "IG Bonds (10yr)", "IG Bonds (10yr)",
        ],
        "Scenario": [
            "Base", "Favorable", "Adverse",
            "Base", "Favorable", "Adverse",
        ],
        "Key assumption": [
            "Multiple stable, trend earnings growth, inflation ~2.5%",
            "Multiple expands, above-trend growth, inflation subsides",
            "Multiple contracts, recession, inflation stays elevated",
            "Yields stable, hold to maturity",
            "Yields fall 50bp, soft landing",
            "Yields rise 100bp, re-acceleration of inflation",
        ],
        "Yield / carry (%)": [
            "1.4", "1.4", "1.4",
            "4.3", "4.3", "4.3",
        ],
        "Est. real return\n(ann., 10yr, %)": [
            "4–5", "7–9", "-1–1",
            "1.5–2.0", "2.5–3.5", "-1.0–0.5",
        ],
        "Main driver": [
            "Earnings growth",
            "Multiple expansion",
            "Multiple contraction",
            "Carry",
            "Carry + price gain",
            "Duration loss",
        ],
    }

    df = pd.DataFrame(scenarios)

    fig, ax = plt.subplots(figsize=(14, 4.0), facecolor="white")
    ax.axis("off")

    col_widths = [0.11, 0.09, 0.32, 0.13, 0.16, 0.19]
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.1)

    # Header
    for j in range(len(df.columns)):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    # Row colors by scenario
    scenario_colors = {
        "Base":      "#e8f4f8",
        "Favorable": "#e8f5e9",
        "Adverse":   "#fce8e8",
    }
    for i in range(1, len(df) + 1):
        scenario = df.iloc[i - 1]["Scenario"]
        color    = scenario_colors.get(scenario, "white")
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_facecolor(color)
            cell.set_edgecolor("#eeeeee")

    ax.set_title(
        "Expected return scenarios for U.S. equities and investment-grade bonds\n"
        "Approximate inputs as of early 2024. Real return estimates are 10-year annualized.",
        fontsize=9, pad=15, loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scenario_table.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  Saved: fig_scenario_table.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Loading Shiller data...")
    shiller = load_shiller()
    print(f"  Loaded {len(shiller)} monthly observations")

    print("\nBuilding Chart 1 — Four-regime return decomposition...")
    chart_regime_decomposition(shiller)

    print("\nBuilding Chart 2 — Starting bond yield vs forward return...")
    chart_bond_yield_vs_return()

    print("\nBuilding Chart 3 — Corporate profit margins...")
    chart_corporate_margins()

    print("\nBuilding Chart 4 — Scenario return table...")
    chart_scenario_table()

    print("\nAll charts saved to:", OUT_DIR.resolve())