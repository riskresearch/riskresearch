import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

# =========================================================
# Paths
# =========================================================
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "outputs" / "ch01"
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
    "equity":    "#2166ac",
    "growth":    "#4dac26",
    "inflation": "#d6604d",
    "multiple":  "#756bb1",
    "total":     "#1a1a1a",
    "real_rate": "#d6604d",
    "neutral":   "#888888",
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
# Chart 1 — Return decomposition
# =========================================================
def chart_return_decomposition(df: pd.DataFrame) -> None:
    regimes = [
        ("1926–1965", "1926-01-01", "1965-12-01"),
        ("1966–1981", "1966-01-01", "1981-12-01"),
        ("1982–1999", "1982-01-01", "1999-12-01"),
        ("2000–2021", "2000-01-01", "2021-12-01"),
    ]

    rows = []
    for name, start, end in regimes:
        sub = df.loc[start:end].dropna(
            subset=["price", "dividend", "earnings", "cpi", "cape"]
        )
        if len(sub) < 24:
            continue

        n_years = len(sub) / 12
        p0  = float(sub["price"].iloc[0])
        p1  = float(sub["price"].iloc[-1])
        d0  = float(sub["dividend"].iloc[0])
        e0  = float(sub["earnings"].iloc[0])
        e1  = float(sub["earnings"].iloc[-1])
        cpi0 = float(sub["cpi"].iloc[0])
        cpi1 = float(sub["cpi"].iloc[-1])

        div_yield  = (d0 / p0) * 100
        inflation  = ((cpi1 / cpi0) ** (1 / n_years) - 1) * 100
        real_e0    = e0 / cpi0
        real_e1    = e1 / cpi1
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
        })

    decomp = pd.DataFrame(rows).set_index("regime")
    components  = ["Dividend yield", "Real EPS growth",
                   "Inflation", "Multiple change"]
    comp_colors = [
        COLORS["equity"],
        COLORS["growth"],
        COLORS["inflation"],
        COLORS["multiple"],
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(decomp))
    bottom = np.zeros(len(decomp))

    # Draw bars
    for comp, color in zip(components, comp_colors):
        vals = decomp[comp].values.astype(float)
        ax.bar(
            x, vals, bottom=bottom, color=color,
            label=comp, width=0.55,
            edgecolor="white", linewidth=0.5,
        )
        bottom = bottom + vals

    # Draw total marker
    ax.plot(
        x, decomp["Total"].values.astype(float),
        "D", color=COLORS["total"],
        markersize=7, zorder=5, label="Total nominal return",
    )

    # -------------------------------------------------------
    # Annotate each segment with its value
    # Use a fresh pass with correct cumulative bottom tracking
    # -------------------------------------------------------
    bottom2 = np.zeros(len(decomp))
    for comp, color in zip(components, comp_colors):
        vals = decomp[comp].values.astype(float)
        for i, (val, bot) in enumerate(zip(vals, bottom2)):
            if abs(val) < 0.4:
                bottom2[i] += val
                continue

            x_center = x[i]
            # Visual top and bottom of this segment
            seg_top = bot + max(val, 0)
            seg_bot = bot + min(val, 0)
            seg_mid = (seg_top + seg_bot) / 2
            seg_h   = seg_top - seg_bot

            if seg_h >= 0.8:
                # Enough room — label inside in white
                ax.text(
                    x_center, seg_mid,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white",
                    fontweight="bold", zorder=10,
                )
            else:
                # Too narrow — label just outside in dark
                offset_dir = 1 if val >= 0 else -1
                ax.text(
                    x_center,
                    seg_mid + offset_dir * (seg_h * 0.5 + 0.25),
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=7.0, color="#333333",
                    fontweight="bold", zorder=10,
                )

        bottom2 = bottom2 + vals

    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(decomp.index, fontsize=9)
    ax.set_ylabel("Annualized contribution (percentage points)", labelpad=8)
    ax.set_title(
        "Decomposition of U.S. equity returns by regime\n"
        "S&P 500, annualized nominal return",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_return_decomposition.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_return_decomposition.png")


# =========================================================
# Chart 2 — CAPE vs subsequent 10-year returns
# =========================================================
def chart_cape_vs_returns(df: pd.DataFrame) -> None:
    horizon = 120  # months

    records = []
    dates   = df.index.tolist()
    for i in range(len(dates) - horizon):
        cape_val = float(df["cape"].iloc[i])
        if np.isnan(cape_val) or cape_val <= 0:
            continue

        price_start = float(df["price"].iloc[i])
        price_end   = float(df["price"].iloc[i + horizon])
        div_start   = float(df["dividend"].iloc[i])
        cpi_start   = float(df["cpi"].iloc[i])
        cpi_end     = float(df["cpi"].iloc[i + horizon])

        if price_start <= 0 or cpi_start <= 0:
            continue

        # Approximate real total return
        price_ret  = (price_end / price_start) ** (1 / 10) - 1
        div_yield  = div_start / price_start
        inflation  = (cpi_end / cpi_start) ** (1 / 10) - 1
        real_ret   = (price_ret + div_yield - inflation) * 100

        records.append({
            "date": dates[i],
            "cape": cape_val,
            "ret10": real_ret,
        })

    scatter = pd.DataFrame(records).dropna()

    x      = scatter["cape"].values
    y      = scatter["ret10"].values
    coeffs = np.polyfit(x, y, 1)
    x_fit  = np.linspace(x.min(), x.max(), 200)
    y_fit  = np.polyval(coeffs, x_fit)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        x, y, s=4, alpha=0.4,
        color=COLORS["equity"], label="Monthly observations",
    )
    ax.plot(
        x_fit, y_fit,
        color=COLORS["inflation"], lw=1.8,
        label=f"OLS fit (slope = {coeffs[0]:.2f})",
    )
    ax.axhline(0, color="#999999", lw=0.8, linestyle="--")

    for label, date_str, offset in [
        ("Jan 2009\n(CAPE≈15)", "2009-01-01", (-3,  3)),
        ("Jan 2000\n(CAPE≈44)", "2000-01-01", ( 1, -6)),
        ("Jan 1982\n(CAPE≈8)",  "1982-01-01", ( 1,  3)),
    ]:
        row = scatter[scatter["date"] == pd.Timestamp(date_str)]
        if not row.empty:
            ax.annotate(
                label,
                xy=(float(row["cape"].iloc[0]),
                    float(row["ret10"].iloc[0])),
                xytext=(
                    float(row["cape"].iloc[0]) + offset[0],
                    float(row["ret10"].iloc[0]) + offset[1],
                ),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                color="#333333",
            )

    ax.set_xlabel("Shiller CAPE at start of period", labelpad=8)
    ax.set_ylabel(
        "Subsequent 10-year annualized real total return (%)",
        labelpad=8,
    )
    ax.set_title(
        "Starting valuation and long-horizon equity returns\n"
        "S&P 500, monthly observations 1881–2015",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cape_vs_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_cape_vs_returns.png")


# =========================================================
# Chart 3 — 2022 repricing
# =========================================================
def chart_2022_repricing() -> None:
    start = "2021-01-01"
    end   = "2023-01-31"

    qqq = yf.download("QQQ", start=start, end=end,
                      progress=False, auto_adjust=True)
    spy = yf.download("SPY", start=start, end=end,
                      progress=False, auto_adjust=True)

    if qqq.empty or spy.empty:
        print("  Warning: equity download failed.")
        return

    qqq_close = qqq["Close"].squeeze()
    spy_close = spy["Close"].squeeze()

    try:
        real_yield = web.DataReader(
            "DFII10", "fred",
            datetime(2021, 1, 1),
            datetime(2023, 1, 31),
        ).squeeze()
    except Exception as e:
        print(f"  Warning: TIPS yield download failed: {e}")
        return

    # Rebase to 100
    qqq_idx = qqq_close / float(qqq_close.iloc[0]) * 100
    spy_idx = spy_close / float(spy_close.iloc[0]) * 100

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(qqq_idx.index, qqq_idx.values,
             color=COLORS["equity"], lw=1.8,
             label="QQQ (Nasdaq-100)")
    ax1.plot(spy_idx.index, spy_idx.values,
             color=COLORS["neutral"], lw=1.4,
             linestyle="--", label="SPY (S&P 500)")

    ax2.plot(real_yield.index, real_yield.values,
             color=COLORS["real_rate"], lw=1.6,
             label="10-yr TIPS real yield (%)")

    ax1.set_ylabel(
        "Equity index (Jan 2021 = 100)",
        color=COLORS["equity"], labelpad=8,
    )
    ax2.set_ylabel(
        "10-year TIPS real yield (%)",
        color=COLORS["real_rate"], labelpad=8,
    )
    ax1.tick_params(axis="y", labelcolor=COLORS["equity"])
    ax2.tick_params(axis="y", labelcolor=COLORS["real_rate"])

    ax1.axvline(pd.Timestamp("2022-01-01"),
                color="#cccccc", lw=0.8, linestyle=":")
    ax1.text(pd.Timestamp("2022-01-15"), ax1.get_ylim()[0] * 1.02,
             "2022 begins", fontsize=7.5, color="#888888")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=8, framealpha=0.5, loc="upper right",
    )

    ax1.set_title(
        "Rising real yields and the repricing of long-duration equities\n"
        "January 2021 – December 2022",
        pad=12,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_2022_repricing.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_2022_repricing.png")


# =========================================================
# Chart 4 — Valuation trap
# =========================================================
def chart_valuation_trap() -> None:
    start = "2011-01-01"
    end   = "2017-01-01"

    tickers = {
        "EUFN": "European Banks (EUFN)",
        "SPY":  "S&P 500 (SPY)",
    }
    prices = {}

    for t, label in tickers.items():
        raw = yf.download(t, start=start, end=end,
                          progress=False, auto_adjust=True)
        if raw.empty:
            print(f"  Warning: no data for {t}")
            continue
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        prices[label] = close

    if len(prices) < 2:
        print("  Skipping valuation trap chart — download failed.")
        return

    df     = pd.DataFrame(prices).dropna()
    df_idx = df / df.iloc[0] * 100

    eufn_col = "European Banks (EUFN)"
    spy_col  = "S&P 500 (SPY)"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_idx.index, df_idx[eufn_col].values,
            color=COLORS["inflation"], lw=1.8,
            label=eufn_col)
    ax.plot(df_idx.index, df_idx[spy_col].values,
            color=COLORS["neutral"], lw=1.6,
            linestyle="--", label=spy_col)

    # Annotate
    mid_date  = pd.Timestamp("2014-06-01")
    nearest   = df_idx.index[df_idx.index.get_indexer(
        [mid_date], method="nearest"
    )[0]]
    y_val = float(df_idx.loc[nearest, eufn_col])
    ax.annotate(
        "European banks remain below\n2011 levels through 2016\n"
        "despite low P/B ratios",
        xy=(nearest, y_val),
        xytext=(pd.Timestamp("2013-01-01"), 60),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        color="#333333",
    )

    ax.set_ylabel("Total return index (Jan 2011 = 100)", labelpad=8)
    ax.set_title(
        "Valuation trap: European banks vs S&P 500, 2011–2016\n"
        "Low P/B ratios did not prevent sustained underperformance",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_valuation_trap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_valuation_trap.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Loading Shiller data...")
    shiller = load_shiller()
    print(f"  Loaded {len(shiller)} monthly observations")

    print("\nBuilding Chart 1 — Return decomposition...")
    chart_return_decomposition(shiller)

    print("\nBuilding Chart 2 — CAPE vs 10-year returns...")
    chart_cape_vs_returns(shiller)

    print("\nBuilding Chart 3 — 2022 repricing...")
    chart_2022_repricing()

    print("\nBuilding Chart 4 — Valuation trap...")
    chart_valuation_trap()

    print("\nAll charts saved to:", OUT_DIR.resolve())