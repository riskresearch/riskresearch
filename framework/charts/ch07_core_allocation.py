import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import pandas_datareader.data as web
from datetime import datetime
import yfinance as yf


# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(__file__).parent.parent / "outputs" / "ch07"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "passive":   "#2166ac",
    "factor":    "#4dac26",
    "active":    "#756bb1",
    "drawdown":  "#d6604d",
    "neutral":   "#888888",
    "value":     "#d6604d",
    "cost_low":  "#2166ac",
    "cost_mid":  "#4dac26",
    "cost_high": "#d6604d",
}


# =========================================================
# Download helper
# =========================================================
def fred(series: str, start: datetime, end: datetime) -> pd.Series:
    return web.DataReader(series, "fred", start, end).squeeze()


# =========================================================
# Chart 1 — Fee drag over 20 years
# =========================================================
def chart_fee_drag() -> None:
    """
    Illustrate the compounding cost of fees over 20 years.
    Three scenarios:
      A: Passive core at 5bps
      B: Factor/smart beta at 20bps
      C: Active fund at 80bps
      D: Active fund at 150bps (plus alpha assumption)
    Assume 7% gross annual return for all.
    """
    years      = np.arange(0, 21)
    gross_ret  = 0.07
    initial    = 100.0

    scenarios = [
        ("Passive index\n(5bps)",          0.0005, COLORS["passive"],   "-"),
        ("Smart beta / factor\n(20bps)",   0.0020, COLORS["factor"],    "--"),
        ("Active equity\n(80bps)",         0.0080, COLORS["active"],    "-."),
        ("Hedge fund\n(150bps + fees)",    0.0150, COLORS["drawdown"],  ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: cumulative wealth
    ax = axes[0]
    terminal_vals = []
    for label, fee, color, ls in scenarios:
        net_ret = gross_ret - fee
        wealth  = initial * (1 + net_ret) ** years
        lw = 2.4 if fee == 0.0005 else 1.6
        ax.plot(years, wealth, color=color, lw=lw,
                linestyle=ls, label=label)
        terminal_vals.append((wealth[-1], color, label))

    # Right-edge terminal value labels (extend xlim to make room)
    ax.set_xlim(0, 24)
    for i, (val, color, label) in enumerate(terminal_vals):
        short = label.split("\n")[0]   # first line only, e.g. "Passive index"
        ax.text(20.4, val, f"{val:.0f}", color=color,
                fontsize=7, va="center", fontweight="bold")

    ax.set_xlabel("Years", labelpad=8)
    ax.set_ylabel("Portfolio value (start = 100)", labelpad=8)
    ax.set_title("Cumulative wealth after fees\n7% gross annual return", pad=10)
    ax.legend(fontsize=8, framealpha=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

    # Right: fee drag in dollar terms
    ax2 = axes[1]
    passive_label, passive_fee, passive_color, passive_ls = scenarios[0]
    passive_wealth = initial * (1 + gross_ret - passive_fee) ** years

    # Passive baseline explicitly at zero drag
    ax2.plot(years, np.zeros_like(years, dtype=float),
             color=passive_color, lw=2.4, linestyle=passive_ls,
             label=passive_label)

    for label, fee, color, ls in scenarios[1:]:
        net_ret = gross_ret - fee
        wealth  = initial * (1 + net_ret) ** years
        drag    = passive_wealth - wealth
        ax2.plot(years, drag, color=color, lw=1.6,
                 linestyle=ls, label=label)

    ax2.set_xlabel("Years", labelpad=8)
    ax2.set_ylabel("Cumulative fee drag vs passive (£/$ per 100 invested)",
                   labelpad=8)
    ax2.set_title("Fee drag relative to passive baseline\n"
                  "Cumulative cost per 100 invested", pad=10)
    ax2.legend(fontsize=8, framealpha=0.5)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(5))

    fig.suptitle(
        "The compounding cost of fees over 20 years\n"
        "7% gross annual return assumed for all strategies",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_fee_drag.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_fee_drag.png")


# =========================================================
# Chart 2 — Value factor underperformance and flows
# =========================================================
def chart_value_underperformance() -> None:
    start = "2013-01-01"
    end   = "2022-12-31"

    tickers = {"IVE": "S&P 500 Value (IVE)",
               "IVW": "S&P 500 Growth (IVW)",
               "SPY": "S&P 500 (SPY)"}
    prices  = {}

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
        print("  Skipping value underperformance chart.")
        return

    df     = pd.DataFrame(prices).dropna()
    df_idx = df / df.iloc[0] * 100

    value_col  = "S&P 500 Value (IVE)"
    growth_col = "S&P 500 Growth (IVW)"
    sp500_col  = "S&P 500 (SPY)"

    rel = (df_idx[value_col] / df_idx[growth_col] * 100)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(df_idx.index, df_idx[value_col],
            color=COLORS["value"], lw=1.8,
            label="S&P 500 Value (IVE)")
    ax.plot(df_idx.index, df_idx[growth_col],
            color=COLORS["passive"], lw=1.8,
            label="S&P 500 Growth (IVW)")
    ax.plot(df_idx.index, df_idx[sp500_col],
            color=COLORS["neutral"], lw=1.2,
            linestyle=":", label="S&P 500 (SPY)")
    ax.set_ylabel("Total return index (Jan 2013 = 100)", labelpad=8)
    ax.set_title(
        "Value vs Growth vs S&P 500: 2013–2022\n"
        "Total return, rebased to 100 in January 2013",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.5)

    ax2 = axes[1]
    ax2.plot(rel.index, rel.values,
             color=COLORS["value"], lw=1.8,
             label="Value / Growth ratio")
    ax2.axhline(100, color="#cccccc", lw=0.8, linestyle="--")
    ax2.fill_between(rel.index, rel.values, 100,
                     where=(rel.values < 100),
                     alpha=0.2, color=COLORS["value"],
                     label="Value underperformance")
    ax2.fill_between(rel.index, rel.values, 100,
                     where=(rel.values >= 100),
                     alpha=0.2, color=COLORS["passive"],
                     label="Value outperformance")

    trough_date = rel.idxmin()
    trough_val  = float(rel.min())
    ax2.annotate(
        f"Value trough\n{trough_date.strftime('%b %Y')}\n"
        f"Ratio = {trough_val:.0f}",
        xy=(trough_date, trough_val),
        xytext=(trough_date - pd.DateOffset(years=2),
                trough_val + 5),
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
        color="#333333",
    )

    ax2.set_ylabel(
        "Value / Growth relative index (Jan 2013 = 100)",
        labelpad=8,
    )
    ax2.legend(fontsize=8, framealpha=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_value_underperformance.png",
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_value_underperformance.png")


# =========================================================
# Chart 3 — Portfolio architecture diagram
# =========================================================
def chart_portfolio_architecture() -> None:
    """
    Illustrative layered portfolio architecture diagram.
    Four layers with approximate risk budget and fee at each.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)

    layers = [
        {
            "y": 0.4,
            "height": 1.6,
            "color": "#d1e8f7",
            "edge":  "#2166ac",
            "label": "Layer 1: Passive market exposure",
            "detail": "Broad equity index + government bonds + diversified commodities\n"
                      "Risk budget: 50–60% | Fee: 3–10bps | Instruments: ETFs, futures",
        },
        {
            "y": 2.2,
            "height": 1.4,
            "color": "#d9f0d3",
            "edge":  "#4dac26",
            "label": "Layer 2: Systematic factor tilts",
            "detail": "Value, momentum, quality, carry — rules-based, diversified\n"
                      "Risk budget: 20–25% | Fee: 10–25bps | Instruments: factor ETFs, systematic overlays",
        },
        {
            "y": 3.8,
            "height": 1.4,
            "color": "#e8dff5",
            "edge":  "#756bb1",
            "label": "Layer 3: Diversifying alternative strategies",
            "detail": "Trend-following, relative value, macro — low equity correlation\n"
                      "Risk budget: 15–20% | Fee: 50–100bps | Instruments: managed futures, CTA",
        },
        {
            "y": 5.4,
            "height": 1.4,
            "color": "#fde8d8",
            "edge":  "#d6604d",
            "label": "Layer 4: Concentrated active positions",
            "detail": "High-conviction fundamental or systematic — must justify cost\n"
                      "Risk budget: 5–10% | Fee: 80–200bps | Instruments: active funds, direct",
        },
    ]

    for layer in layers:
        rect = plt.Rectangle(
            (0.5, layer["y"]), 9, layer["height"],
            linewidth=1.5,
            edgecolor=layer["edge"],
            facecolor=layer["color"],
        )
        ax.add_patch(rect)
        ax.text(
            1.0, layer["y"] + layer["height"] * 0.65,
            layer["label"],
            fontsize=9.5, fontweight="bold",
            color=layer["edge"], va="center",
        )
        ax.text(
            1.0, layer["y"] + layer["height"] * 0.25,
            layer["detail"],
            fontsize=7.5, color="#333333", va="center",
        )

    # Arrow indicating hierarchy
    ax.annotate(
        "", xy=(0.2, 5.8), xytext=(0.2, 0.8),
        arrowprops=dict(
            arrowstyle="->", color="#666666", lw=1.5,
        ),
    )
    ax.text(
        0.05, 3.3, "Increasing\ncost &\ncomplexity",
        fontsize=7.5, color="#666666",
        ha="center", va="center", rotation=90,
    )

    ax.set_title(
        "Layered portfolio architecture\n"
        "Core first — active additions require explicit justification at each layer",
        fontsize=10, pad=12,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_portfolio_architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_portfolio_architecture.png")


# =========================================================
# Chart 4 — Active sleeve justification table
# =========================================================
def chart_active_sleeve_table() -> None:
    """
    Table of four justifications for adding an active sleeve,
    with required evidence, typical cost, and failure mode.
    """
    data = {
        "Justification": [
            "Expected return\nimprovement",
            "Diversification",
            "Capital efficiency",
            "Downside protection",
        ],
        "What it requires": [
            "Identifiable return source not in core; net alpha after costs > 0",
            "Low correlation to core in stress periods, not just in calm markets",
            "Achieves same exposure with less capital; frees room for other bets",
            "Reduces core's dominant tail risk; may cost carry in calm regimes",
        ],
        "Typical cost": [
            "15–80bps management fee; turnover; tax drag",
            "50–150bps; operational complexity; monitoring burden",
            "Financing cost; margin; roll cost for futures overlays",
            "Premium / carry bleed; opportunity cost in trending markets",
        ],
        "Primary failure mode": [
            "Factor exposure misidentified as alpha; fees exceed gross edge",
            "Calm-period correlation used; strategy fails in stress alongside core",
            "Leverage hidden; financing conditions change; margin called",
            "Protection too expensive; hedge removes too much upside; timing error",
        ],
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 3.8), facecolor="white")
    ax.axis("off")

    col_widths = [0.14, 0.30, 0.26, 0.30]
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.8)

    for j in range(len(df.columns)):
        cell = tbl[0, j]
        cell.set_facecolor("#2166ac")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    row_colors = [
        "#d1e8f7", "#d9f0d3", "#e8dff5", "#fde8d8",
    ]
    for i in range(1, len(df) + 1):
        color = row_colors[(i - 1) % len(row_colors)]
        for j in range(len(df.columns)):
            cell = tbl[i, j]
            cell.set_facecolor(color)
            cell.set_edgecolor("#eeeeee")

    ax.set_title(
        "Four justifications for an active sleeve — required evidence and failure modes",
        fontsize=9, pad=15, loc="left",
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_active_sleeve_table.png",
                bbox_inches="tight", dpi=180)
    plt.close(fig)
    print("  Saved: fig_active_sleeve_table.png")


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    print("Building Chart 1 — Fee drag over 20 years...")
    chart_fee_drag()

    print("\nBuilding Chart 2 — Value factor underperformance 2013–2022...")
    chart_value_underperformance()

    print("\nBuilding Chart 3 — Portfolio architecture diagram...")
    chart_portfolio_architecture()

    print("\nBuilding Chart 4 — Active sleeve justification table...")
    chart_active_sleeve_table()

    print("\nAll charts saved to:", OUT_DIR.resolve())