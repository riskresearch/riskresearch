import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vectorbt as vbt


# =========================================================
# Configuration
# =========================================================
START = "2000-01-01"
END = "2026-03-23"

EQUITY_SYMBOL = "ES=F"
ALT_SYMBOLS = ["NQ=F", "RTY=F", "GC=F", "SI=F", "ZN=F"]

COMMISSION_PER_CONTRACT = 5.0
MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY = 0.10

CONTRACT_SPECS = {
    "ES=F": {"tick_size": 0.25, "tick_value": 12.5},
    "NQ=F": {"tick_size": 0.25, "tick_value": 5.0},
    "GC=F": {"tick_size": 0.1, "tick_value": 10.0},
    "SI=F": {"tick_size": 0.005, "tick_value": 25.0},
    "ZN=F": {"tick_size": 0.015625, "tick_value": 15.625},
    "RTY=F": {"tick_size": 0.1, "tick_value": 5.0},
}

ASSET_START_DATES = {
    "ES=F": "2000-01-01",
    "NQ=F": "2000-01-01",
    "RTY=F": "2000-01-01",
    "ZN=F": "2000-01-01",
    "ZT=F": "2000-01-01",
    "BTC=F": "2017-12-18",
    "GC=F": "2000-01-01",
    "SI=F": "2000-01-01",
    "YM=F": "2000-01-01",
    "ZF=F": "2000-01-01",
    "CL=F": "2000-01-01",
    "NG=F": "2000-01-01",
    "ZC=F": "2000-01-01",
    "ZS=F": "2000-01-01",
    "LE=F": "2000-01-01",
    "HE=F": "2000-01-01",
    "CC=F": "2000-01-01",
    "KC=F": "2000-01-01",
    "CT=F": "2000-01-01",
}

ASSET_COLORS = {
    "ES=F": "tab:blue",
    "NQ=F": "tab:orange",
    "RTY=F": "tab:green",
    "ZN=F": "tab:red",
    "ZT=F": "tab:cyan",
    "BTC=F": "tab:purple",
    "GC=F": "tab:brown",
    "SI=F": "tab:pink",
    "YM=F": "tab:olive",
    "ZF=F": "gold",
    "CL=F": "sienna",
    "NG=F": "deepskyblue",
    "ZC=F": "goldenrod",
    "ZS=F": "olive",
    "LE=F": "lightcoral",
    "HE=F": "hotpink",
    "CC=F": "chocolate",
    "KC=F": "peru",
    "CT=F": "slateblue",
}
PORTFOLIO_COLOR = "black"
ALT_BUCKET_COLOR = "gray"

TARGET_NEG_VOL_MIN = 0.10
TARGET_NEG_VOL_MAX = 0.20

VOL_WINDOW = 21
SORTINO_WINDOW = 252
SORTINO_Z_WINDOW = 252

REALIZED_VOL_FLOOR = 0.01
MAR = 0.0

MAX_LEVERAGE = 3.0
MIN_LEVERAGE = 0.0
FLAT_EPS = 0.0

IR_WINDOW = 252

TACTICAL_MAX_WEIGHT = 0.50
TACTICAL_MIN_WEIGHT = 0.00

IR_FLOOR = 0.0
IR_CAP = 1.0

SMOOTH_ALLOCATIONS = True
ALLOC_SMOOTH_METHOD = "ema"
ALLOC_SMOOTH_SPAN = 10
ALLOC_SMOOTH_WINDOW = 10

PAIRWISE_CORR_WINDOW = 63
PAIRWISE_CORR_LOW = 0.75
PAIRWISE_CORR_HIGH = 0.92
PAIRWISE_PENALTY_STRENGTH = 0.40

ROLLING_AB_WINDOW = 252
TRADING_DAYS = 252
MAR_EQ_RET = 0.0

INIT_CASH = 1_000_000.0

OUTPUTS_DIR = Path("outputs")
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"
SNAPSHOTS_DIR = OUTPUTS_DIR / "snapshots"


# =========================================================
# Helpers
# =========================================================
def ensure_output_dirs() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def make_index_tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def drawdown_from_equity(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    return eq / peak - 1.0


def neg_annualized_vol_from_returns(rets: pd.Series, mar: float = 0.0, freq: int = 252) -> float:
    rets = rets.dropna()
    if len(rets) == 0:
        return float("nan")
    d = np.minimum(rets.to_numpy(dtype=float) - float(mar), 0.0)
    return float(np.sqrt(np.mean(d * d)) * np.sqrt(freq))


def perf_stats_from_equity(eq: pd.Series, freq: int = 252, mar_eq_ret: float = 0.0) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {
            "total_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "neg_ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_dd": np.nan,
        }

    rets = eq.pct_change().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = (len(eq) - 1) / freq
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else np.nan

    ann_vol = float(rets.std(ddof=0) * np.sqrt(freq)) if len(rets) > 1 else np.nan
    neg_ann_vol = neg_annualized_vol_from_returns(rets, mar=mar_eq_ret, freq=freq)

    ann_ret = float(rets.mean() * freq) if len(rets) > 0 else np.nan
    sharpe = float(ann_ret / ann_vol) if ann_vol and ann_vol != 0 else np.nan
    sortino = float(ann_ret / neg_ann_vol) if neg_ann_vol and neg_ann_vol != 0 else np.nan
    max_dd = float(drawdown_from_equity(eq).min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "neg_ann_vol": neg_ann_vol,
        "sharpe": np.nan if ann_vol == 0 or pd.isna(ann_vol) else sharpe,
        "sortino": np.nan if neg_ann_vol == 0 or pd.isna(neg_ann_vol) else sortino,
        "max_dd": max_dd,
    }


def downside_realized_vol_from_returns(
    rets: pd.Series,
    window: int,
    mar: float = 0.0,
    floor: float = 0.0,
    ann_factor: float = 252.0,
) -> pd.Series:
    d = np.minimum(rets.astype(float) - mar, 0.0)
    rv_down = d.rolling(window, min_periods=window).apply(
        lambda x: np.sqrt(np.mean(np.square(x))), raw=True
    ) * np.sqrt(ann_factor)
    if floor is not None and floor > 0:
        rv_down = rv_down.clip(lower=floor)
    return rv_down


def rolling_sortino_from_returns(
    rets: pd.Series,
    window: int = 252,
    mar: float = 0.0,
    ann_factor: int = 252,
) -> pd.Series:
    rets = rets.astype(float)
    mu_ann = rets.rolling(window, min_periods=window).mean() * ann_factor
    downside = np.minimum(rets - mar, 0.0)
    down_vol_ann = downside.rolling(window, min_periods=window).apply(
        lambda x: np.sqrt(np.mean(np.square(x))), raw=True
    ) * np.sqrt(ann_factor)
    return mu_ann / down_vol_ann.replace(0.0, np.nan)


def rolling_bounded_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    return np.tanh(z)


def map_zscore_to_target_vol_linear(z: pd.Series, vol_low: float, vol_high: float) -> pd.Series:
    score = (z + 1.0) / 2.0
    score = score.clip(lower=0.0, upper=1.0)
    return vol_low + (vol_high - vol_low) * score


def smooth_series(s: pd.Series, method: str = "ema", span: int = 10, window: int = 10) -> pd.Series:
    if method.lower() == "ema":
        return s.ewm(span=span, adjust=False, min_periods=1).mean()
    if method.lower() == "sma":
        return s.rolling(window=window, min_periods=1).mean()
    raise ValueError(f"Unknown smoothing method: {method}")


def rolling_information_ratio(active_ret: pd.Series, window: int = 252) -> pd.Series:
    mu = active_ret.rolling(window, min_periods=window).mean()
    sd = active_ret.rolling(window, min_periods=window).std(ddof=0)
    ir = mu / sd.replace(0.0, np.nan)
    return ir * np.sqrt(TRADING_DAYS)


def rolling_alpha_beta(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    window: int = 252,
    ann_factor: int = 252,
):
    df_ab = pd.concat([strat_ret.rename("S"), bench_ret.rename("B")], axis=1).dropna()
    if df_ab.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    s = df_ab["S"]
    b = df_ab["B"]

    mean_s = s.rolling(window, min_periods=window).mean()
    mean_b = b.rolling(window, min_periods=window).mean()
    cov_sb = s.rolling(window, min_periods=window).cov(b)
    var_b = b.rolling(window, min_periods=window).var()

    beta = cov_sb / var_b.replace(0.0, np.nan)
    alpha_ann = (mean_s - beta * mean_b) * ann_factor

    return beta.reindex(strat_ret.index), alpha_ann.reindex(strat_ret.index)


def corr_penalty_from_corr(
    corr_value: float,
    corr_low: float,
    corr_high: float,
    penalty_strength: float,
) -> float:
    if pd.isna(corr_value):
        return 1.0
    if corr_value <= corr_low:
        return 1.0
    if corr_value >= corr_high:
        return 1.0 - penalty_strength

    x = (corr_value - corr_low) / (corr_high - corr_low)
    x = float(np.clip(x, 0.0, 1.0))
    s = 3.0 * x**2 - 2.0 * x**3
    return 1.0 - penalty_strength * s


def contribution_summary_from_returns(contrib_ret_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in contrib_ret_df.columns:
        eq = INIT_CASH * (1.0 + contrib_ret_df[col].fillna(0.0)).cumprod()
        stats = perf_stats_from_equity(eq, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)
        total_contrib = float(contrib_ret_df[col].sum())
        rows.append(
            {
                "asset": col,
                "total_contribution_sum_ret": total_contrib,
                "total_return": stats["total_return"],
                "cagr": stats["cagr"],
                "ann_vol": stats["ann_vol"],
                "sortino": stats["sortino"],
                "max_dd": stats["max_dd"],
            }
        )
    return pd.DataFrame(rows).set_index("asset").sort_values(
        "total_contribution_sum_ret", ascending=False
    )


def normalize_positive_shares(s: pd.Series) -> pd.Series:
    s = s.clip(lower=0.0)
    denom = s.sum()
    if denom <= 0:
        return pd.Series(0.0, index=s.index)
    return s / denom


def get_contract_point_value(symbol: str) -> float:
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Missing CONTRACT_SPECS for {symbol}")
    spec = CONTRACT_SPECS[symbol]
    return float(spec["tick_value"]) / float(spec["tick_size"])


def compute_asset_leverage_series(
    raw_ret_s: pd.Series,
    target_vol_s: pd.Series,
    live_mask: pd.Series,
    realized_vol_window: int,
    mar: float,
    realized_vol_floor: float,
    ann_factor: int,
    min_leverage: float,
    max_leverage: float,
    flat_eps: float,
):
    raw_ret_s = raw_ret_s.astype(float).fillna(0.0)
    target_vol_s = target_vol_s.astype(float)
    live_mask = live_mask.fillna(False)

    rv_down = downside_realized_vol_from_returns(
        rets=raw_ret_s,
        window=realized_vol_window,
        mar=mar,
        floor=realized_vol_floor,
        ann_factor=ann_factor,
    )
    rv1 = rv_down.shift(1)

    l_raw = target_vol_s / rv1.replace(0.0, np.nan)
    l_raw = l_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    l_star = l_raw.clip(lower=min_leverage, upper=max_leverage)

    if flat_eps > 0:
        l_star = l_star.where(l_star >= flat_eps, 0.0)

    l_raw = l_raw.where(live_mask, 0.0)
    l_star = l_star.where(live_mask, 0.0)
    rv_down = rv_down.where(live_mask, np.nan)

    return {
        "rv_down": rv_down,
        "L_star_raw": l_raw,
        "L_star": l_star,
    }


def simulate_master_portfolio_direct(
    close_df: pd.DataFrame,
    asset_l_star_df: pd.DataFrame,
    weights_t1: pd.DataFrame,
    asset_live_mask: dict,
    symbols: list[str],
    init_cash: float,
    commission_per_contract: float,
    min_rebalance_notional_pct_of_equity: float,
):
    idx = close_df.index

    portfolio_equity = pd.Series(index=idx, dtype=float)
    portfolio_ret = pd.Series(0.0, index=idx, dtype=float)

    asset_contracts = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_trade_contracts = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_commission_paid = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_target_notional = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_desired_contracts_raw = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_rebalance_notional_frac = pd.DataFrame(0.0, index=idx, columns=symbols)
    asset_contribution_ret = pd.DataFrame(0.0, index=idx, columns=symbols)

    prev_contracts = {sym: 0 for sym in symbols}
    portfolio_equity.iloc[0] = init_cash

    for i, dt in enumerate(idx):
        eq_prev = init_cash if i == 0 else float(portfolio_equity.iloc[i - 1])
        total_pnl_net = 0.0

        for sym in symbols:
            px = float(close_df.loc[dt, sym]) if pd.notna(close_df.loc[dt, sym]) else np.nan
            weight = float(weights_t1.loc[dt, sym]) if pd.notna(weights_t1.loc[dt, sym]) else 0.0
            l_star = (
                float(asset_l_star_df.loc[dt, sym])
                if pd.notna(asset_l_star_df.loc[dt, sym])
                else 0.0
            )
            live_now = bool(asset_live_mask[sym].reindex(idx).fillna(False).loc[dt])

            point_value = get_contract_point_value(sym)

            if (
                (not live_now)
                or (not np.isfinite(px))
                or px <= 0.0
                or eq_prev <= 0.0
                or weight <= 0.0
                or l_star <= 0.0
            ):
                desired_contracts = 0
            else:
                desired_notional = eq_prev * weight * l_star
                contract_notional = px * point_value
                if contract_notional <= 0.0 or not np.isfinite(contract_notional):
                    desired_contracts = 0
                else:
                    desired_contracts = int(np.floor(desired_notional / contract_notional))

            asset_desired_contracts_raw.loc[dt, sym] = desired_contracts

            prev_c = prev_contracts[sym]
            contract_change = desired_contracts - prev_c
            contract_notional_now = px * point_value if np.isfinite(px) else np.nan

            if eq_prev > 0 and pd.notna(contract_notional_now) and np.isfinite(contract_notional_now):
                change_notional = abs(contract_change) * contract_notional_now
                rebalance_frac = change_notional / eq_prev
            else:
                rebalance_frac = 0.0

            asset_rebalance_notional_frac.loc[dt, sym] = rebalance_frac

            if abs(contract_change) == 0:
                executed_contracts = prev_c
            elif rebalance_frac < min_rebalance_notional_pct_of_equity:
                executed_contracts = prev_c
            else:
                executed_contracts = desired_contracts

            traded = abs(executed_contracts - prev_c)
            comm = traded * commission_per_contract

            asset_contracts.loc[dt, sym] = executed_contracts
            asset_trade_contracts.loc[dt, sym] = traded
            asset_commission_paid.loc[dt, sym] = comm
            asset_target_notional.loc[dt, sym] = (
                executed_contracts * px * point_value if np.isfinite(px) else 0.0
            )

            if i == 0:
                pnl_gross = 0.0
            else:
                px_prev = (
                    float(close_df.iloc[i - 1][sym])
                    if pd.notna(close_df.iloc[i - 1][sym])
                    else np.nan
                )
                price_change = (px - px_prev) if np.isfinite(px) and np.isfinite(px_prev) else 0.0
                pnl_gross = executed_contracts * point_value * price_change

            pnl_net = pnl_gross - comm
            total_pnl_net += pnl_net
            asset_contribution_ret.loc[dt, sym] = pnl_net / eq_prev if eq_prev > 0 else 0.0

            prev_contracts[sym] = executed_contracts

        portfolio_equity.loc[dt] = max(eq_prev + total_pnl_net, 0.0)
        portfolio_ret.loc[dt] = total_pnl_net / eq_prev if eq_prev > 0 else 0.0

    asset_contribution_equity = pd.DataFrame(index=idx, columns=symbols, dtype=float)
    for sym in symbols:
        asset_contribution_equity[sym] = INIT_CASH * (
            1.0 + asset_contribution_ret[sym].fillna(0.0)
        ).cumprod()

    return {
        "portfolio_equity": portfolio_equity,
        "portfolio_ret": portfolio_ret,
        "asset_contracts": asset_contracts,
        "asset_trade_contracts": asset_trade_contracts,
        "asset_commission_paid": asset_commission_paid,
        "asset_target_notional": asset_target_notional,
        "asset_desired_contracts_raw": asset_desired_contracts_raw,
        "asset_rebalance_notional_frac": asset_rebalance_notional_frac,
        "asset_contribution_ret": asset_contribution_ret,
        "asset_contribution_equity": asset_contribution_equity,
    }


# =========================================================
# Data loading
# =========================================================
def load_data():
    all_symbols = list(dict.fromkeys([EQUITY_SYMBOL] + ALT_SYMBOLS))

    for sym in all_symbols:
        if not isinstance(sym, str) or not sym.strip():
            raise ValueError(f"Invalid symbol in all_symbols: {repr(sym)}")
        if sym not in CONTRACT_SPECS:
            raise ValueError(f"Missing CONTRACT_SPECS for {sym}")

    requested_start_ts = pd.to_datetime(START)
    requested_end_ts = pd.to_datetime(END)

    max_warmup = max(
        VOL_WINDOW,
        SORTINO_WINDOW,
        SORTINO_Z_WINDOW,
        IR_WINDOW,
        ROLLING_AB_WINDOW,
        PAIRWISE_CORR_WINDOW,
    )
    if SMOOTH_ALLOCATIONS:
        max_warmup += max(ALLOC_SMOOTH_SPAN, ALLOC_SMOOTH_WINDOW)

    warmup_buffer_days = max_warmup + 10
    warmup_start_ts = requested_start_ts - pd.offsets.BDay(warmup_buffer_days)

    price_dict = {}
    failed_symbols = []

    print("Downloading symbols:")
    for sym in all_symbols:
        print("  ", sym)
        try:
            data = vbt.YFData.download(
                sym,
                start=warmup_start_ts.strftime("%Y-%m-%d"),
                end=END,
            )
            df_sym = data.get()
            df_sym = df_sym.rename(columns={c: c.capitalize() for c in df_sym.columns})

            if df_sym is None or df_sym.empty:
                raise ValueError(f"No data returned for {sym}")

            if "Close" not in df_sym.columns:
                raise ValueError(f"YF data missing Close for {sym}. Columns: {df_sym.columns.tolist()}")

            df_sym = df_sym[["Close"]].copy()
            df_sym.index = pd.to_datetime(df_sym.index)
            df_sym = make_index_tz_naive(df_sym)
            df_sym.index = df_sym.index.normalize()

            if df_sym["Close"].dropna().empty:
                raise ValueError(f"Close series empty for {sym}")

            price_dict[sym] = df_sym["Close"].astype(float)

        except Exception as e:
            print(f"[FAILED] {sym}: {type(e).__name__}: {e}")
            failed_symbols.append(sym)

    if failed_symbols:
        print("\nFailed symbols:")
        for sym in failed_symbols:
            print("  ", sym)

    if EQUITY_SYMBOL in failed_symbols:
        raise ValueError(f"EQUITY_SYMBOL failed to download: {EQUITY_SYMBOL}")

    usable_alt_symbols = [sym for sym in ALT_SYMBOLS if sym in price_dict]
    if len(usable_alt_symbols) == 0:
        raise ValueError("No ALT_SYMBOLS left after download failures.")

    all_symbols = list(dict.fromkeys([EQUITY_SYMBOL] + usable_alt_symbols))
    print("\nUsable symbols:", all_symbols)
    print("Usable ALT_SYMBOLS:", usable_alt_symbols)

    close_df = pd.concat({sym: price_dict[sym] for sym in all_symbols}, axis=1).sort_index()
    close_df = close_df.loc[warmup_start_ts:requested_end_ts].copy()
    close_df = close_df.loc[close_df.index.dayofweek < 5].copy()

    dates = close_df.index

    if len(close_df) < max_warmup + 3:
        raise ValueError(
            f"Not enough total rows after warmup. Need at least {max_warmup + 3} rows, got {len(close_df)}."
        )

    return {
        "close_df": close_df,
        "dates": dates,
        "all_symbols": all_symbols,
        "alt_symbols": usable_alt_symbols,
        "requested_start_ts": requested_start_ts,
        "requested_end_ts": requested_end_ts,
        "warmup_start_ts": warmup_start_ts,
        "max_warmup": max_warmup,
    }


# =========================================================
# Signals
# =========================================================
def build_signals(data: dict):
    close_df = data["close_df"]
    dates = data["dates"]
    all_symbols = data["all_symbols"]
    alt_symbols = data["alt_symbols"]

    raw_returns_df = close_df.pct_change().fillna(0.0)
    raw_benchmark_ret = raw_returns_df[EQUITY_SYMBOL].copy()

    raw_sortino = {}
    raw_sortino_z = {}
    for sym in all_symbols:
        srt = rolling_sortino_from_returns(
            rets=raw_returns_df[sym],
            window=SORTINO_WINDOW,
            mar=MAR,
            ann_factor=TRADING_DAYS,
        )
        raw_sortino[sym] = srt
        raw_sortino_z[sym] = rolling_bounded_zscore(srt, SORTINO_Z_WINDOW)

    raw_sortino_df = pd.DataFrame(raw_sortino).reindex(dates)
    raw_sortino_z_df = pd.DataFrame(raw_sortino_z).reindex(dates)

    alt_ir_vs_es = {}
    for alt in alt_symbols:
        alt_ir_vs_es[alt] = rolling_information_ratio(
            raw_returns_df[alt] - raw_benchmark_ret,
            window=IR_WINDOW,
        )

    alt_ir_vs_es_df = pd.DataFrame(alt_ir_vs_es).reindex(dates)

    return {
        "raw_returns_df": raw_returns_df,
        "raw_benchmark_ret": raw_benchmark_ret,
        "raw_sortino_df": raw_sortino_df,
        "raw_sortino_z_df": raw_sortino_z_df,
        "alt_ir_vs_es_df": alt_ir_vs_es_df,
    }


# =========================================================
# Asset leverage layer
# =========================================================
def build_asset_leverage(data: dict, signals: dict):
    close_df = data["close_df"]
    dates = data["dates"]
    all_symbols = data["all_symbols"]
    raw_returns_df = signals["raw_returns_df"]
    raw_sortino_z_df = signals["raw_sortino_z_df"]

    asset_target_vol = {}
    asset_l_star = {}
    asset_l_star_raw = {}
    asset_live_mask = {}
    asset_rv_down = {}

    for sym in all_symbols:
        close_s = close_df[sym].astype(float)
        start_dt = pd.to_datetime(ASSET_START_DATES.get(sym, START))
        asset_ret = raw_returns_df[sym].copy()

        target_vol = map_zscore_to_target_vol_linear(
            z=raw_sortino_z_df[sym],
            vol_low=TARGET_NEG_VOL_MIN,
            vol_high=TARGET_NEG_VOL_MAX,
        )
        asset_target_vol[sym] = target_vol

        live_mask = (close_s.index >= start_dt) & target_vol.notna() & close_s.notna()
        asset_live_mask[sym] = live_mask

        lev = compute_asset_leverage_series(
            raw_ret_s=asset_ret,
            target_vol_s=target_vol,
            live_mask=live_mask,
            realized_vol_window=VOL_WINDOW,
            mar=MAR,
            realized_vol_floor=REALIZED_VOL_FLOOR,
            ann_factor=TRADING_DAYS,
            min_leverage=MIN_LEVERAGE,
            max_leverage=MAX_LEVERAGE,
            flat_eps=FLAT_EPS,
        )
        asset_rv_down[sym] = lev["rv_down"]
        asset_l_star_raw[sym] = lev["L_star_raw"]
        asset_l_star[sym] = lev["L_star"]

    asset_target_vol_df = pd.DataFrame(asset_target_vol).reindex(dates)
    asset_l_star_raw_df = pd.DataFrame(asset_l_star_raw).reindex(dates).fillna(0.0)
    asset_l_star_df = pd.DataFrame(asset_l_star).reindex(dates).fillna(0.0)
    asset_rv_down_df = pd.DataFrame(asset_rv_down).reindex(dates)

    return {
        "asset_target_vol_df": asset_target_vol_df,
        "asset_l_star_raw_df": asset_l_star_raw_df,
        "asset_l_star_df": asset_l_star_df,
        "asset_live_mask": asset_live_mask,
        "asset_rv_down_df": asset_rv_down_df,
    }


# =========================================================
# Allocation layer
# =========================================================
def build_allocations(data: dict, signals: dict, asset_layer: dict):
    dates = data["dates"]
    alt_symbols = data["alt_symbols"]
    all_symbols = data["all_symbols"]

    alt_ir_vs_es_df = signals["alt_ir_vs_es_df"]
    raw_returns_df = signals["raw_returns_df"]
    asset_live_mask = asset_layer["asset_live_mask"]

    positive_alt_ir_df = alt_ir_vs_es_df.clip(lower=0.0)
    best_alt_ir = positive_alt_ir_df.max(axis=1).fillna(0.0)

    if IR_CAP <= IR_FLOOR:
        raise ValueError("IR_CAP must be strictly greater than IR_FLOOR for linear tactical sizing.")

    tactical_score = ((best_alt_ir - IR_FLOOR) / (IR_CAP - IR_FLOOR)).clip(lower=0.0, upper=1.0)

    tactical_weight_used = (
        TACTICAL_MIN_WEIGHT
        + (TACTICAL_MAX_WEIGHT - TACTICAL_MIN_WEIGHT) * tactical_score
    ).fillna(TACTICAL_MIN_WEIGHT)

    base_score_df = positive_alt_ir_df.copy()
    for alt in alt_symbols:
        live_and_signal = asset_live_mask[alt] & alt_ir_vs_es_df[alt].notna()
        base_score_df.loc[~live_and_signal, alt] = 0.0

    alt_returns_df = raw_returns_df[alt_symbols].copy()

    pair_corr = {}
    for i in alt_symbols:
        for j in alt_symbols:
            if i == j:
                continue
            pair_corr[(i, j)] = alt_returns_df[i].rolling(
                PAIRWISE_CORR_WINDOW,
                min_periods=PAIRWISE_CORR_WINDOW,
            ).corr(alt_returns_df[j])

    pairwise_max_stronger_corr_df = pd.DataFrame(np.nan, index=dates, columns=alt_symbols)
    pairwise_penalty_df = pd.DataFrame(1.0, index=dates, columns=alt_symbols)

    for dt in dates:
        eligible = [alt for alt in alt_symbols if base_score_df.loc[dt, alt] > 0.0]

        if len(eligible) <= 1:
            for alt in eligible:
                pairwise_penalty_df.loc[dt, alt] = 1.0
            continue

        for alt in eligible:
            score_i = float(base_score_df.loc[dt, alt])
            stronger_peers = [peer for peer in eligible if base_score_df.loc[dt, peer] > score_i]

            if len(stronger_peers) == 0:
                pairwise_penalty_df.loc[dt, alt] = 1.0
                pairwise_max_stronger_corr_df.loc[dt, alt] = np.nan
                continue

            corr_candidates = []
            for peer in stronger_peers:
                c = pair_corr[(alt, peer)].loc[dt]
                if pd.notna(c):
                    corr_candidates.append(max(float(c), 0.0))

            if len(corr_candidates) == 0:
                pairwise_penalty_df.loc[dt, alt] = 1.0
                pairwise_max_stronger_corr_df.loc[dt, alt] = np.nan
                continue

            max_corr_to_stronger = float(np.max(corr_candidates))
            pairwise_max_stronger_corr_df.loc[dt, alt] = max_corr_to_stronger
            pairwise_penalty_df.loc[dt, alt] = corr_penalty_from_corr(
                corr_value=max_corr_to_stronger,
                corr_low=PAIRWISE_CORR_LOW,
                corr_high=PAIRWISE_CORR_HIGH,
                penalty_strength=PAIRWISE_PENALTY_STRENGTH,
            )

    alloc_score_df = base_score_df * pairwise_penalty_df

    for alt in alt_symbols:
        live_and_signal = asset_live_mask[alt] & alt_ir_vs_es_df[alt].notna()
        alloc_score_df.loc[~live_and_signal, alt] = 0.0
        pairwise_penalty_df.loc[~live_and_signal, alt] = 1.0
        pairwise_max_stronger_corr_df.loc[~live_and_signal, alt] = np.nan

    score_sum = alloc_score_df.sum(axis=1)
    alt_weights = pd.DataFrame(0.0, index=dates, columns=alt_symbols)

    has_positive_score = score_sum > 0.0
    for alt in alt_symbols:
        alt_weights.loc[has_positive_score, alt] = (
            tactical_weight_used.loc[has_positive_score]
            * alloc_score_df.loc[has_positive_score, alt]
            / score_sum.loc[has_positive_score]
        )

    alt_weights = alt_weights.fillna(0.0)

    weights_df = pd.concat(
        [
            (1.0 - alt_weights.sum(axis=1)).rename(EQUITY_SYMBOL),
            alt_weights,
        ],
        axis=1,
    ).fillna(0.0)

    if SMOOTH_ALLOCATIONS:
        for col in weights_df.columns:
            weights_df[col] = smooth_series(
                weights_df[col],
                method=ALLOC_SMOOTH_METHOD,
                span=ALLOC_SMOOTH_SPAN,
                window=ALLOC_SMOOTH_WINDOW,
            )
        weights_sum = weights_df.sum(axis=1).replace(0.0, np.nan)
        weights_df = weights_df.div(weights_sum, axis=0).fillna(0.0)

    weights_t1 = weights_df.shift(1).fillna(0.0)

    return {
        "positive_alt_ir_df": positive_alt_ir_df,
        "best_alt_ir": best_alt_ir,
        "tactical_score": tactical_score,
        "tactical_weight_used": tactical_weight_used,
        "base_score_df": base_score_df,
        "pairwise_penalty_df": pairwise_penalty_df,
        "pairwise_max_stronger_corr_df": pairwise_max_stronger_corr_df,
        "alloc_score_df": alloc_score_df,
        "weights_df": weights_df[all_symbols],
        "weights_t1": weights_t1[all_symbols],
    }


# =========================================================
# Simulation layer
# =========================================================
def run_simulation(data: dict, signals: dict, asset_layer: dict, allocations: dict):
    close_df = data["close_df"]
    all_symbols = data["all_symbols"]
    alt_symbols = data["alt_symbols"]
    dates = data["dates"]

    asset_l_star_df = asset_layer["asset_l_star_df"]
    asset_live_mask = asset_layer["asset_live_mask"]
    raw_benchmark_ret = signals["raw_benchmark_ret"]
    weights_t1 = allocations["weights_t1"]

    sim = simulate_master_portfolio_direct(
        close_df=close_df[all_symbols],
        asset_l_star_df=asset_l_star_df[all_symbols],
        weights_t1=weights_t1[all_symbols],
        asset_live_mask=asset_live_mask,
        symbols=all_symbols,
        init_cash=INIT_CASH,
        commission_per_contract=COMMISSION_PER_CONTRACT,
        min_rebalance_notional_pct_of_equity=MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY,
    )

    portfolio_equity = sim["portfolio_equity"]
    portfolio_ret = sim["portfolio_ret"]

    asset_contracts_df = sim["asset_contracts"]
    asset_trade_contracts_df = sim["asset_trade_contracts"]
    asset_commission_paid_df = sim["asset_commission_paid"]
    asset_target_notional_df = sim["asset_target_notional"]
    asset_desired_contracts_raw_df = sim["asset_desired_contracts_raw"]
    asset_rebalance_notional_frac_df = sim["asset_rebalance_notional_frac"]
    weighted_contrib_ret_df = sim["asset_contribution_ret"]
    asset_contribution_equity_df = sim["asset_contribution_equity"]

    benchmark_equity = INIT_CASH * (1.0 + raw_benchmark_ret).cumprod()

    es_weighted_ret = weighted_contrib_ret_df[EQUITY_SYMBOL].fillna(0.0)
    alts_weighted_ret = weighted_contrib_ret_df[alt_symbols].sum(axis=1).fillna(0.0)

    es_weighted_equity = INIT_CASH * (1.0 + es_weighted_ret).cumprod()
    alts_weighted_equity = INIT_CASH * (1.0 + alts_weighted_ret).cumprod()

    es_only_weights_df = pd.DataFrame(0.0, index=dates, columns=all_symbols)
    es_only_weights_df[EQUITY_SYMBOL] = 1.0
    es_only_weights_t1 = es_only_weights_df.shift(1).fillna(0.0)

    es_only_sim = simulate_master_portfolio_direct(
        close_df=close_df[all_symbols],
        asset_l_star_df=asset_l_star_df[all_symbols],
        weights_t1=es_only_weights_t1[all_symbols],
        asset_live_mask=asset_live_mask,
        symbols=all_symbols,
        init_cash=INIT_CASH,
        commission_per_contract=COMMISSION_PER_CONTRACT,
        min_rebalance_notional_pct_of_equity=MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY,
    )
    es_only_equity = es_only_sim["portfolio_equity"]

    return {
        "portfolio_equity": portfolio_equity,
        "portfolio_ret": portfolio_ret,
        "benchmark_equity": benchmark_equity,
        "es_only_equity": es_only_equity,
        "es_weighted_equity": es_weighted_equity,
        "alts_weighted_equity": alts_weighted_equity,
        "asset_contracts_df": asset_contracts_df,
        "asset_trade_contracts_df": asset_trade_contracts_df,
        "asset_commission_paid_df": asset_commission_paid_df,
        "asset_target_notional_df": asset_target_notional_df,
        "asset_desired_contracts_raw_df": asset_desired_contracts_raw_df,
        "asset_rebalance_notional_frac_df": asset_rebalance_notional_frac_df,
        "weighted_contrib_ret_df": weighted_contrib_ret_df,
        "asset_contribution_equity_df": asset_contribution_equity_df,
    }


# =========================================================
# Analysis layer
# =========================================================
def build_analysis(data: dict, signals: dict, asset_layer: dict, allocations: dict, sim_results: dict):
    dates = data["dates"]
    requested_start_ts = data["requested_start_ts"]
    alt_symbols = data["alt_symbols"]

    asset_live_mask = asset_layer["asset_live_mask"]
    alt_ir_vs_es_df = signals["alt_ir_vs_es_df"]

    alt_live_any = pd.Series(False, index=dates)
    for alt in alt_symbols:
        alt_has_signal = asset_live_mask[alt] & alt_ir_vs_es_df[alt].notna()
        alt_live_any = alt_live_any | alt_has_signal

    strategy_live_mask = asset_live_mask[EQUITY_SYMBOL] & alt_live_any

    if not strategy_live_mask.any():
        raise ValueError("Strategy never becomes live with the current windows/settings.")

    strategy_start_ts = strategy_live_mask[strategy_live_mask].index[0]
    print(f"\nStrategy live start date: {strategy_start_ts.date()}")

    analysis_mask = dates >= max(requested_start_ts, strategy_start_ts)

    portfolio_equity_analysis = sim_results["portfolio_equity"].loc[analysis_mask].dropna()
    es_only_analysis = sim_results["es_only_equity"].loc[analysis_mask].dropna()
    benchmark_analysis = sim_results["benchmark_equity"].loc[analysis_mask].dropna()
    es_weighted_analysis = sim_results["es_weighted_equity"].loc[analysis_mask].dropna()
    alts_weighted_analysis = sim_results["alts_weighted_equity"].loc[analysis_mask].dropna()

    common_idx = portfolio_equity_analysis.index
    for s in [es_only_analysis, benchmark_analysis, es_weighted_analysis, alts_weighted_analysis]:
        common_idx = common_idx.intersection(s.index)

    portfolio_equity_analysis = portfolio_equity_analysis.loc[common_idx]
    es_only_analysis = es_only_analysis.loc[common_idx]
    benchmark_analysis = benchmark_analysis.loc[common_idx]
    es_weighted_analysis = es_weighted_analysis.loc[common_idx]
    alts_weighted_analysis = alts_weighted_analysis.loc[common_idx]

    if len(portfolio_equity_analysis) < 2:
        raise ValueError("No valid analysis data in the requested window.")

    first_plot_date = common_idx[0]
    portfolio_plot = INIT_CASH * (portfolio_equity_analysis / portfolio_equity_analysis.loc[first_plot_date])
    benchmark_plot = INIT_CASH * (benchmark_analysis / benchmark_analysis.loc[first_plot_date])
    es_only_plot = INIT_CASH * (es_only_analysis / es_only_analysis.loc[first_plot_date])
    es_weighted_plot = INIT_CASH * (es_weighted_analysis / es_weighted_analysis.loc[first_plot_date])
    alts_weighted_plot = INIT_CASH * (alts_weighted_analysis / alts_weighted_analysis.loc[first_plot_date])

    weights_analysis = allocations["weights_df"].loc[common_idx]
    weights_used_analysis = allocations["weights_t1"].loc[common_idx]
    weighted_contrib_ret_analysis = sim_results["weighted_contrib_ret_df"].loc[common_idx]

    raw_sortino_analysis = signals["raw_sortino_df"].loc[common_idx]
    raw_sortino_z_analysis = signals["raw_sortino_z_df"].loc[common_idx]
    target_vol_analysis = asset_layer["asset_target_vol_df"].loc[common_idx]
    alt_ir_vs_es_analysis = signals["alt_ir_vs_es_df"].loc[common_idx]
    tactical_weight_used_analysis = allocations["tactical_weight_used"].loc[common_idx]
    base_score_analysis = allocations["base_score_df"].loc[common_idx]
    alloc_score_analysis = allocations["alloc_score_df"].loc[common_idx]
    pairwise_penalty_analysis = allocations["pairwise_penalty_df"].loc[common_idx]
    pairwise_max_stronger_corr_analysis = allocations["pairwise_max_stronger_corr_df"].loc[common_idx]
    asset_l_star_analysis = asset_layer["asset_l_star_df"].loc[common_idx]

    asset_contracts_analysis = sim_results["asset_contracts_df"].loc[common_idx]
    asset_trade_contracts_analysis = sim_results["asset_trade_contracts_df"].loc[common_idx]
    asset_commission_analysis = sim_results["asset_commission_paid_df"].loc[common_idx]
    asset_desired_contracts_raw_analysis = sim_results["asset_desired_contracts_raw_df"].loc[common_idx]
    asset_rebalance_notional_frac_analysis = sim_results["asset_rebalance_notional_frac_df"].loc[common_idx]
    asset_target_notional_analysis = sim_results["asset_target_notional_df"].loc[common_idx]
    asset_contribution_equity_analysis = sim_results["asset_contribution_equity_df"].loc[common_idx]

    strat_rets = portfolio_plot.pct_change().dropna()
    bench_rets = benchmark_plot.pct_change().dropna()
    ab_idx = strat_rets.index.intersection(bench_rets.index)

    beta, alpha_ann = rolling_alpha_beta(
        strat_ret=strat_rets.loc[ab_idx],
        bench_ret=bench_rets.loc[ab_idx],
        window=ROLLING_AB_WINDOW,
        ann_factor=TRADING_DAYS,
    )

    beta_analysis = beta.loc[common_idx.intersection(beta.index)]
    alpha_ann_analysis = alpha_ann.loc[common_idx.intersection(alpha_ann.index)]

    asset_contribution_summary = contribution_summary_from_returns(weighted_contrib_ret_analysis)
    asset_total_contrib = weighted_contrib_ret_analysis.sum().sort_values(ascending=False)
    asset_positive_share = normalize_positive_shares(asset_total_contrib)

    latest_idx = common_idx[-1]
    snapshot = {
        "date": str(latest_idx.date()),
        "portfolio_equity": float(sim_results["portfolio_equity"].loc[latest_idx]),
        "tactical_weight_used_signal": float(tactical_weight_used_analysis.loc[latest_idx]),
        f"weight_signal_{EQUITY_SYMBOL}": float(weights_analysis.loc[latest_idx, EQUITY_SYMBOL]),
        "weight_total_alts_signal": float(weights_analysis.loc[latest_idx, alt_symbols].sum()),
        f"weight_used_{EQUITY_SYMBOL}": float(weights_used_analysis.loc[latest_idx, EQUITY_SYMBOL]),
        "weight_used_total_alts": float(weights_used_analysis.loc[latest_idx, alt_symbols].sum()),
        f"contracts_{EQUITY_SYMBOL}": float(asset_contracts_analysis.loc[latest_idx, EQUITY_SYMBOL]),
        f"desired_contracts_raw_{EQUITY_SYMBOL}": float(
            asset_desired_contracts_raw_analysis.loc[latest_idx, EQUITY_SYMBOL]
        ),
        f"rebalance_notional_frac_{EQUITY_SYMBOL}": float(
            asset_rebalance_notional_frac_analysis.loc[latest_idx, EQUITY_SYMBOL]
        ),
        f"target_notional_{EQUITY_SYMBOL}": float(asset_target_notional_analysis.loc[latest_idx, EQUITY_SYMBOL]),
        f"L_star_{EQUITY_SYMBOL}": float(asset_l_star_analysis.loc[latest_idx, EQUITY_SYMBOL]),
    }

    for alt in alt_symbols:
        snapshot[f"weight_signal_{alt}"] = float(weights_analysis.loc[latest_idx, alt])
        snapshot[f"weight_used_{alt}"] = float(weights_used_analysis.loc[latest_idx, alt])
        snapshot[f"contracts_{alt}"] = float(asset_contracts_analysis.loc[latest_idx, alt])
        snapshot[f"desired_contracts_raw_{alt}"] = float(asset_desired_contracts_raw_analysis.loc[latest_idx, alt])
        snapshot[f"rebalance_notional_frac_{alt}"] = float(
            asset_rebalance_notional_frac_analysis.loc[latest_idx, alt]
        )
        snapshot[f"target_notional_{alt}"] = float(asset_target_notional_analysis.loc[latest_idx, alt])
        snapshot[f"L_star_{alt}"] = float(asset_l_star_analysis.loc[latest_idx, alt])
        snapshot[f"ir_vs_es_raw_{alt}"] = (
            float(alt_ir_vs_es_analysis.loc[latest_idx, alt])
            if pd.notna(alt_ir_vs_es_analysis.loc[latest_idx, alt])
            else np.nan
        )
        snapshot[f"sortino_raw_{alt}"] = (
            float(raw_sortino_analysis.loc[latest_idx, alt])
            if pd.notna(raw_sortino_analysis.loc[latest_idx, alt])
            else np.nan
        )
        snapshot[f"sortino_z_raw_{alt}"] = (
            float(raw_sortino_z_analysis.loc[latest_idx, alt])
            if pd.notna(raw_sortino_z_analysis.loc[latest_idx, alt])
            else np.nan
        )
        snapshot[f"target_vol_{alt}"] = (
            float(target_vol_analysis.loc[latest_idx, alt])
            if pd.notna(target_vol_analysis.loc[latest_idx, alt])
            else np.nan
        )
        snapshot[f"penalty_{alt}"] = (
            float(pairwise_penalty_analysis.loc[latest_idx, alt])
            if pd.notna(pairwise_penalty_analysis.loc[latest_idx, alt])
            else np.nan
        )
        snapshot[f"max_corr_to_stronger_{alt}"] = (
            float(pairwise_max_stronger_corr_analysis.loc[latest_idx, alt])
            if pd.notna(pairwise_max_stronger_corr_analysis.loc[latest_idx, alt])
            else np.nan
        )

    portfolio_stats = perf_stats_from_equity(portfolio_plot, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)
    benchmark_stats = perf_stats_from_equity(benchmark_plot, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)
    es_only_stats = perf_stats_from_equity(es_only_plot, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)
    es_contrib_stats = perf_stats_from_equity(es_weighted_plot, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)
    alts_contrib_stats = perf_stats_from_equity(alts_weighted_plot, freq=TRADING_DAYS, mar_eq_ret=MAR_EQ_RET)

    return {
        "strategy_start_ts": strategy_start_ts,
        "common_idx": common_idx,
        "portfolio_plot": portfolio_plot,
        "benchmark_plot": benchmark_plot,
        "es_only_plot": es_only_plot,
        "es_weighted_plot": es_weighted_plot,
        "alts_weighted_plot": alts_weighted_plot,
        "weights_analysis": weights_analysis,
        "weights_used_analysis": weights_used_analysis,
        "weighted_contrib_ret_analysis": weighted_contrib_ret_analysis,
        "raw_sortino_analysis": raw_sortino_analysis,
        "raw_sortino_z_analysis": raw_sortino_z_analysis,
        "target_vol_analysis": target_vol_analysis,
        "alt_ir_vs_es_analysis": alt_ir_vs_es_analysis,
        "tactical_weight_used_analysis": tactical_weight_used_analysis,
        "base_score_analysis": base_score_analysis,
        "alloc_score_analysis": alloc_score_analysis,
        "pairwise_penalty_analysis": pairwise_penalty_analysis,
        "pairwise_max_stronger_corr_analysis": pairwise_max_stronger_corr_analysis,
        "asset_l_star_analysis": asset_l_star_analysis,
        "asset_contracts_analysis": asset_contracts_analysis,
        "asset_trade_contracts_analysis": asset_trade_contracts_analysis,
        "asset_commission_analysis": asset_commission_analysis,
        "asset_desired_contracts_raw_analysis": asset_desired_contracts_raw_analysis,
        "asset_rebalance_notional_frac_analysis": asset_rebalance_notional_frac_analysis,
        "asset_target_notional_analysis": asset_target_notional_analysis,
        "asset_contribution_equity_analysis": asset_contribution_equity_analysis,
        "beta_analysis": beta_analysis,
        "alpha_ann_analysis": alpha_ann_analysis,
        "asset_contribution_summary": asset_contribution_summary,
        "asset_total_contrib": asset_total_contrib,
        "asset_positive_share": asset_positive_share,
        "portfolio_stats": portfolio_stats,
        "benchmark_stats": benchmark_stats,
        "es_only_stats": es_only_stats,
        "es_contrib_stats": es_contrib_stats,
        "alts_contrib_stats": alts_contrib_stats,
        "snapshot": snapshot,
    }


# =========================================================
# Saving tables / snapshot
# =========================================================
def save_outputs(data: dict, signals: dict, allocations: dict, sim_results: dict, analysis: dict):
    perf_df = pd.DataFrame(
        {
            "final_portfolio": analysis["portfolio_stats"],
            f"{EQUITY_SYMBOL}_benchmark_buy_hold": analysis["benchmark_stats"],
            f"{EQUITY_SYMBOL}_only_option_b": analysis["es_only_stats"],
            f"{EQUITY_SYMBOL}_actual_contribution": analysis["es_contrib_stats"],
            "alts_actual_contribution": analysis["alts_contrib_stats"],
        }
    ).T

    perf_df.to_csv(TABLES_DIR / "performance_stats.csv")
    analysis["asset_contribution_summary"].to_csv(TABLES_DIR / "asset_contribution_summary.csv")
    analysis["asset_total_contrib"].rename("total_contribution").to_csv(TABLES_DIR / "asset_total_contribution.csv")
    analysis["asset_positive_share"].rename("positive_contribution_share").to_csv(
        TABLES_DIR / "asset_positive_contribution_share.csv"
    )
    analysis["asset_commission_analysis"].sum().sort_values(ascending=False).rename("total_commissions").to_csv(
        TABLES_DIR / "total_commissions_by_asset.csv"
    )
    analysis["asset_trade_contracts_analysis"].sum().sort_values(ascending=False).rename("total_contracts_traded").to_csv(
        TABLES_DIR / "total_contracts_traded_by_asset.csv"
    )

    analysis["weights_analysis"].to_csv(TABLES_DIR / "weights_signal.csv")
    analysis["weights_used_analysis"].to_csv(TABLES_DIR / "weights_used_t1.csv")
    analysis["alt_ir_vs_es_analysis"].to_csv(TABLES_DIR / "alt_ir_vs_es.csv")
    analysis["raw_sortino_analysis"].to_csv(TABLES_DIR / "raw_sortino.csv")
    analysis["raw_sortino_z_analysis"].to_csv(TABLES_DIR / "raw_sortino_z.csv")
    analysis["target_vol_analysis"].to_csv(TABLES_DIR / "target_vol.csv")
    analysis["base_score_analysis"].to_csv(TABLES_DIR / "base_score.csv")
    analysis["alloc_score_analysis"].to_csv(TABLES_DIR / "alloc_score.csv")
    analysis["pairwise_penalty_analysis"].to_csv(TABLES_DIR / "pairwise_penalty.csv")
    analysis["pairwise_max_stronger_corr_analysis"].to_csv(TABLES_DIR / "pairwise_max_stronger_corr.csv")
    analysis["asset_l_star_analysis"].to_csv(TABLES_DIR / "asset_l_star.csv")
    analysis["asset_contracts_analysis"].to_csv(TABLES_DIR / "asset_contracts.csv")
    analysis["asset_trade_contracts_analysis"].to_csv(TABLES_DIR / "asset_trade_contracts.csv")
    analysis["asset_commission_analysis"].to_csv(TABLES_DIR / "asset_commissions.csv")
    analysis["asset_desired_contracts_raw_analysis"].to_csv(TABLES_DIR / "asset_desired_contracts_raw.csv")
    analysis["asset_rebalance_notional_frac_analysis"].to_csv(TABLES_DIR / "asset_rebalance_notional_frac.csv")
    analysis["asset_target_notional_analysis"].to_csv(TABLES_DIR / "asset_target_notional.csv")
    analysis["asset_contribution_equity_analysis"].to_csv(TABLES_DIR / "asset_contribution_equity.csv")
    analysis["beta_analysis"].rename("beta").to_csv(TABLES_DIR / "rolling_beta.csv")
    analysis["alpha_ann_analysis"].rename("annualized_alpha").to_csv(TABLES_DIR / "rolling_annualized_alpha.csv")

    curves_df = pd.DataFrame(
        {
            "portfolio_plot": analysis["portfolio_plot"],
            "benchmark_plot": analysis["benchmark_plot"],
            "es_only_plot": analysis["es_only_plot"],
            "es_weighted_plot": analysis["es_weighted_plot"],
            "alts_weighted_plot": analysis["alts_weighted_plot"],
        }
    )
    curves_df.to_csv(TABLES_DIR / "rebased_equity_curves.csv")

    with open(SNAPSHOTS_DIR / "latest_allocator_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(analysis["snapshot"], f, indent=2, allow_nan=True)


# =========================================================
# Plot helpers
# =========================================================
def save_fig(filename: str):
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_outputs(data: dict, analysis: dict):
    all_symbols = data["all_symbols"]
    alt_symbols = data["alt_symbols"]

    plt.figure(figsize=(14, 6))
    plt.plot(analysis["portfolio_plot"], label="Final portfolio", color=PORTFOLIO_COLOR, linewidth=2)
    plt.plot(
        analysis["benchmark_plot"],
        label=f"{EQUITY_SYMBOL} benchmark (buy & hold)",
        color=ASSET_COLORS.get(EQUITY_SYMBOL, "tab:blue"),
        alpha=0.9,
    )
    plt.plot(
        analysis["es_weighted_plot"],
        label=f"{EQUITY_SYMBOL} actual portfolio contribution",
        color=ASSET_COLORS.get(EQUITY_SYMBOL, "tab:blue"),
        linestyle="--",
    )
    plt.plot(
        analysis["alts_weighted_plot"],
        label="Alt sleeves actual portfolio contribution",
        color=ALT_BUCKET_COLOR,
        linestyle="-.",
    )
    plt.yscale("log")
    plt.title(f"Final portfolio vs {EQUITY_SYMBOL} benchmark vs actual portfolio components")
    plt.grid(True)
    plt.legend()
    save_fig("01_equity_curve.png")

    plt.figure(figsize=(14, 4))
    plt.plot(drawdown_from_equity(analysis["portfolio_plot"]), label="Final portfolio drawdown", color=PORTFOLIO_COLOR, linewidth=2)
    plt.plot(
        drawdown_from_equity(analysis["benchmark_plot"]),
        label=f"{EQUITY_SYMBOL} benchmark drawdown",
        color=ASSET_COLORS.get(EQUITY_SYMBOL, "tab:blue"),
        alpha=0.9,
    )
    plt.plot(
        drawdown_from_equity(analysis["es_weighted_plot"]),
        label=f"{EQUITY_SYMBOL} contribution drawdown",
        color=ASSET_COLORS.get(EQUITY_SYMBOL, "tab:blue"),
        linestyle="--",
    )
    plt.plot(
        drawdown_from_equity(analysis["alts_weighted_plot"]),
        label="Alt contribution drawdown",
        color=ALT_BUCKET_COLOR,
        linestyle="-.",
    )
    plt.title("Drawdowns")
    plt.grid(True)
    plt.legend()
    save_fig("02_drawdowns.png")

    plt.figure(figsize=(14, 4))
    for alt in alt_symbols:
        plt.plot(
            analysis["alt_ir_vs_es_analysis"][alt],
            label=f"IR(raw {alt} vs raw {EQUITY_SYMBOL})",
            color=ASSET_COLORS.get(alt, None),
        )
    plt.axhline(0.0, linestyle="--", alpha=0.7, color="black", label="IR = 0")
    plt.axhline(IR_CAP, linestyle="--", alpha=0.5, color="gray", label="IR cap")
    plt.title(f"Rolling {IR_WINDOW}d raw alt-vs-ES IR")
    plt.grid(True)
    plt.legend()
    save_fig("03_alt_ir_vs_es.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["raw_sortino_analysis"][sym],
            label=f"Sortino raw {sym}",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.axhline(0.0, linestyle="--", alpha=0.7, color="black", label="Sortino = 0")
    plt.title(f"Rolling {SORTINO_WINDOW}d raw Sortino by asset")
    plt.grid(True)
    plt.legend()
    save_fig("04_raw_sortino.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["raw_sortino_z_analysis"][sym],
            label=f"Bounded Sortino z-score raw {sym}",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.axhline(0.0, linestyle="--", alpha=0.7, color="black", label="z = 0")
    plt.axhline(1.0, linestyle="--", alpha=0.5, color="gray", label="z = 1")
    plt.axhline(-1.0, linestyle="--", alpha=0.5, color="gray", label="z = -1")
    plt.title("Rolling bounded raw Sortino z-score by asset")
    plt.grid(True)
    plt.legend()
    save_fig("05_raw_sortino_z.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["target_vol_analysis"][sym],
            label=f"Target neg vol {sym}",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.axhline(TARGET_NEG_VOL_MIN, linestyle="--", alpha=0.7, color="black", label="Target vol min")
    plt.axhline(TARGET_NEG_VOL_MAX, linestyle="--", alpha=0.7, color="gray", label="Target vol max")
    plt.title("Per-asset target downside vol from bounded raw Sortino z-score (LINEAR)")
    plt.grid(True)
    plt.legend()
    save_fig("06_target_vol.png")

    plt.figure(figsize=(14, 4))
    plt.plot(analysis["tactical_weight_used_analysis"], label="Tactical sleeve signal", color="black")
    plt.axhline(TACTICAL_MAX_WEIGHT, linestyle="--", alpha=0.5, color="gray", label="Tactical max")
    plt.axhline(TACTICAL_MIN_WEIGHT, linestyle="--", alpha=0.5, color="gray", label="Tactical min")
    plt.title("Tactical sleeve size signal from LINEAR best positive raw alt IR")
    plt.grid(True)
    plt.legend()
    save_fig("07_tactical_weight_signal.png")

    plt.figure(figsize=(14, 4))
    for alt in alt_symbols:
        plt.plot(
            analysis["base_score_analysis"][alt],
            label=f"Base IR score {alt}",
            color=ASSET_COLORS.get(alt, None),
        )
    plt.title("Base alt scores = positive raw IR only")
    plt.grid(True)
    plt.legend()
    save_fig("08_base_scores.png")

    plt.figure(figsize=(14, 4))
    for alt in alt_symbols:
        plt.plot(
            analysis["pairwise_penalty_analysis"][alt],
            label=f"Pairwise penalty {alt}",
            color=ASSET_COLORS.get(alt, None),
        )
    plt.title("Penalty applied only to weaker members of correlated pairs")
    plt.grid(True)
    plt.legend()
    save_fig("09_pairwise_penalty.png")

    plt.figure(figsize=(14, 4))
    for alt in alt_symbols:
        plt.plot(
            analysis["pairwise_max_stronger_corr_analysis"][alt],
            label=f"Max corr to stronger peer {alt}",
            color=ASSET_COLORS.get(alt, None),
        )
    plt.axhline(PAIRWISE_CORR_LOW, linestyle="--", alpha=0.7, color="black", label="Pair corr low")
    plt.axhline(PAIRWISE_CORR_HIGH, linestyle="--", alpha=0.7, color="gray", label="Pair corr high")
    plt.title("Maximum positive correlation to stronger eligible peer")
    plt.grid(True)
    plt.legend()
    save_fig("10_pairwise_max_corr.png")

    plt.figure(figsize=(14, 4))
    for alt in alt_symbols:
        plt.plot(
            analysis["alloc_score_analysis"][alt],
            label=f"Final alloc score {alt}",
            color=ASSET_COLORS.get(alt, None),
        )
    plt.title("Final alt allocation scores = positive raw IR * weaker-peer correlation penalty")
    plt.grid(True)
    plt.legend()
    save_fig("11_alloc_scores.png")

    plt.figure(figsize=(14, 4))
    for sym in analysis["weights_analysis"].columns:
        plt.plot(
            analysis["weights_analysis"][sym],
            label=f"Weight signal {sym}",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Top-level portfolio weights (signal)")
    plt.grid(True)
    plt.legend()
    save_fig("12_weights_signal.png")

    plt.figure(figsize=(14, 4))
    for sym in analysis["weights_used_analysis"].columns:
        plt.plot(
            analysis["weights_used_analysis"][sym],
            label=f"Weight used t-1 {sym}",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Top-level portfolio weights actually used in execution (lagged)")
    plt.grid(True)
    plt.legend()
    save_fig("13_weights_used_t1.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        s = analysis["asset_contribution_equity_analysis"][sym]
        plt.plot(s, label=f"{sym} contribution equity", color=ASSET_COLORS.get(sym, None))
    plt.yscale("log")
    plt.title("Per-asset cumulative contribution equities (portfolio-based sizing)")
    plt.grid(True)
    plt.legend()
    save_fig("14_asset_contribution_equity.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        s = analysis["asset_l_star_analysis"][sym]
        plt.plot(s, label=f"{sym} leverage target", color=ASSET_COLORS.get(sym, None))
    plt.axhline(MAX_LEVERAGE, linestyle="--", alpha=0.7, color="black", label="Max leverage")
    plt.title("Per-asset sleeve leverage targets")
    plt.grid(True)
    plt.legend()
    save_fig("15_asset_leverage_targets.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["asset_contracts_analysis"][sym],
            label=f"{sym} contracts",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Per-asset executed contract holdings")
    plt.grid(True)
    plt.legend()
    save_fig("16_asset_contracts.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["asset_desired_contracts_raw_analysis"][sym],
            label=f"{sym} raw desired contracts",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Per-asset raw desired contract holdings before no-trade band")
    plt.grid(True)
    plt.legend()
    save_fig("17_asset_desired_contracts_raw.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["asset_rebalance_notional_frac_analysis"][sym],
            label=f"{sym} rebalance notional / portfolio equity",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.axhline(
        MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY,
        linestyle="--",
        alpha=0.7,
        color="black",
        label="No-trade band",
    )
    plt.title("Per-asset rebalance notional as fraction of total portfolio equity")
    plt.grid(True)
    plt.legend()
    save_fig("18_rebalance_notional_frac.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["asset_trade_contracts_analysis"][sym],
            label=f"{sym} traded contracts",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Per-asset daily traded contracts")
    plt.grid(True)
    plt.legend()
    save_fig("19_asset_trade_contracts.png")

    plt.figure(figsize=(14, 4))
    for sym in all_symbols:
        plt.plot(
            analysis["asset_commission_analysis"][sym].cumsum(),
            label=f"{sym} cumulative commissions",
            color=ASSET_COLORS.get(sym, None),
        )
    plt.title("Cumulative commissions by asset")
    plt.grid(True)
    plt.legend()
    save_fig("20_cumulative_commissions.png")

    plt.figure(figsize=(14, 5))
    cum_contrib = analysis["weighted_contrib_ret_analysis"].cumsum()
    for sym in analysis["weighted_contrib_ret_analysis"].columns:
        plt.plot(cum_contrib[sym], label=f"Cumulative contrib {sym}", color=ASSET_COLORS.get(sym, None))
    plt.axhline(0.0, linestyle="--", color="black", alpha=0.7)
    plt.title("Cumulative portfolio contribution by asset")
    plt.grid(True)
    plt.legend()
    save_fig("21_cumulative_contribution.png")

    plt.figure(figsize=(14, 4))
    plt.plot(analysis["beta_analysis"], label=f"Rolling {ROLLING_AB_WINDOW}d beta", color="black")
    plt.axhline(1.0, linestyle="--", alpha=0.7, color="gray", label="Beta = 1")
    plt.axhline(0.0, linestyle="--", alpha=0.7, color="black", label="Beta = 0")
    plt.title(f"Rolling {ROLLING_AB_WINDOW}d Beta: Portfolio vs {EQUITY_SYMBOL} benchmark")
    plt.grid(True)
    plt.legend()
    save_fig("22_rolling_beta.png")

    plt.figure(figsize=(14, 4))
    plt.plot(analysis["alpha_ann_analysis"], label=f"Rolling {ROLLING_AB_WINDOW}d annualized alpha", color="black")
    plt.axhline(0.0, linestyle="--", alpha=0.7, color="gray", label="Alpha = 0")
    plt.title(f"Rolling {ROLLING_AB_WINDOW}d Annualized Alpha: Portfolio vs {EQUITY_SYMBOL} benchmark")
    plt.grid(True)
    plt.legend()
    save_fig("23_rolling_alpha.png")


# =========================================================
# Console summary
# =========================================================
def print_summary(data: dict, analysis: dict):
    print("\n=== Per-asset weighted contribution summary ===")
    print(analysis["asset_contribution_summary"])

    print("\n=== Total contribution by asset (sum of daily portfolio contribution returns over analysis window) ===")
    print(analysis["asset_total_contrib"])

    print("\n=== Positive contribution share by asset ===")
    print(analysis["asset_positive_share"])

    print("\n=== Equity Core + Tactical Sleeve Allocator (OPTION B: Direct Portfolio) ===")
    print("Signals from raw returns only:")
    print("  - Sortino(raw asset)")
    print("  - IR(raw alt vs raw ES)")
    print("Implementation layer:")
    print("  - Per-asset bounded raw Sortino z-score in [-1, 1] -> LINEAR target downside vol -> leverage")
    print("  - Correlation penalty computed from raw returns")
    print("  - Only the weaker member of correlated eligible pairs is penalized")
    print("  - Contract realism: integer contracts only, $5 commission / contract traded")
    print(f"  - Minimum rebalance band: {MIN_REBALANCE_NOTIONAL_PCT_OF_EQUITY:.2%} of total portfolio equity")
    print("  - Any unused tactical sleeve allocation goes back to ES")
    print("  - Tactical sleeve size uses LINEAR mapping from best positive alt IR")
    print("  - Position sizing uses ONE shared portfolio equity base")
    print("  - Target notional(asset) = portfolio_equity_prev * lagged_weight(asset) * current_leverage(asset)")
    print(f"Equity core symbol: {EQUITY_SYMBOL}")
    print(f"Alternative symbols: {data['alt_symbols']}")
    print(f"Requested analysis window: {data['requested_start_ts'].date()} -> {data['requested_end_ts'].date()}")
    print(f"Strategy live start date: {analysis['strategy_start_ts'].date()}")
    print(f"IR window: {IR_WINDOW}")
    print(f"Sortino window: {SORTINO_WINDOW}")
    print(f"Sortino z-score window: {SORTINO_Z_WINDOW}")
    print(f"Linear IR floor/cap: {IR_FLOOR:.2f} / {IR_CAP:.2f}")
    print(f"Tactical sleeve range: [{TACTICAL_MIN_WEIGHT:.2f}, {TACTICAL_MAX_WEIGHT:.2f}]")
    print(f"Target negative vol range: [{TARGET_NEG_VOL_MIN:.2f}, {TARGET_NEG_VOL_MAX:.2f}]")
    print(f"Correlation penalty low/high: {PAIRWISE_CORR_LOW:.2f} / {PAIRWISE_CORR_HIGH:.2f}")
    print(f"Correlation penalty strength: {PAIRWISE_PENALTY_STRENGTH:.2f}")

    print("\nFinal portfolio stats (rebased comparison window):")
    print(analysis["portfolio_stats"])

    print(f"\n{EQUITY_SYMBOL} benchmark buy-and-hold stats (rebased comparison window):")
    print(analysis["benchmark_stats"])

    print(f"\n{EQUITY_SYMBOL}-only portfolio stats under OPTION B (rebased comparison window):")
    print(analysis["es_only_stats"])

    print(f"\n{EQUITY_SYMBOL} actual portfolio contribution stats (rebased):")
    print(analysis["es_contrib_stats"])

    print("\nAlt sleeves actual portfolio contribution stats (rebased):")
    print(analysis["alts_contrib_stats"])

    print("\nLatest allocator snapshot:")
    print(analysis["snapshot"])

    print("\n=== Total commissions paid by asset over analysis window ===")
    print(analysis["asset_commission_analysis"].sum().sort_values(ascending=False))

    print("\n=== Total contracts traded by asset over analysis window ===")
    print(analysis["asset_trade_contracts_analysis"].sum().sort_values(ascending=False))

    print("\nSaved outputs:")
    print(f"  Charts:    {CHARTS_DIR.resolve()}")
    print(f"  Tables:    {TABLES_DIR.resolve()}")
    print(f"  Snapshots: {SNAPSHOTS_DIR.resolve()}")


# =========================================================
# Main
# =========================================================
def main():
    ensure_output_dirs()

    data = load_data()
    signals = build_signals(data)
    asset_layer = build_asset_leverage(data, signals)
    allocations = build_allocations(data, signals, asset_layer)
    sim_results = run_simulation(data, signals, asset_layer, allocations)
    analysis = build_analysis(data, signals, asset_layer, allocations, sim_results)

    save_outputs(data, signals, allocations, sim_results, analysis)
    plot_outputs(data, analysis)
    print_summary(data, analysis)


if __name__ == "__main__":
    main()