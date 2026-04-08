"""
Live monitor v3 for ES Core Relative Replacement on MT5

Key improvements vs v2
----------------------
- Writes output inside the deployed strategy folder
- Masks private account identifiers by default
- Supports manual monitoring inception date
- Distinguishes provisional performance from certified performance
- Uses clearer metadata and health checks
- Handles missing EA state export gracefully

What this monitor can do now
----------------------------
1. Read MT5 account, positions, orders, and deal history
2. Build a provisional post-deployment performance view
3. Compare against ES benchmark from the chosen inception date
4. Reconstruct current exposures by asset family
5. Run implementation checks when EA state export exists
6. Mark performance quality explicitly

What it still cannot fully certify without extra logging
--------------------------------------------------------
- Exact historical daily marked-to-market NAV
- Exact target-vs-actual implementation without EA state export
- Full attribution quality on a very short live window
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================================================
# Configuration
# =========================================================

DEPLOYED_DIR = r"C:\Users\Juanan\Downloads\riskresearch\02_alternative-risk-premia\equity-core-futures (deployed)"
OUTPUT_DIR = os.path.join(DEPLOYED_DIR, "monitor_output")
STATE_CSV_PATH = os.path.join(DEPLOYED_DIR, "allocator_state_latest.csv")

# Set this to the date the new EA actually started governing the account.
# Example: "2026-04-07"
MONITORING_INCEPTION_DATE: Optional[str] = "2026-04-05"

LOOKBACK_DAYS_DEALS = 730
LOOKBACK_DAYS_EQUITY = 730
BENCHMARK_SYMBOL_YF = "ES=F"

EXPECTED_ASSETS = ["ES", "NQ", "RTY", "ZN", "GC", "SI"]
EXPECTED_PREFIXES = {
    "ES": "ES_",
    "NQ": "NQ_",
    "RTY": "RTY_",
    "ZN": "ZN_",
    "GC": "GC_",
    "SI": "SI_",
}

MAX_LEVERAGE_EXPECTED = 3.0
TACTICAL_MAX_WEIGHT_EXPECTED = 0.50
TARGET_ACTUAL_LOTS_TOL = 1e-6
TARGET_ACTUAL_NOTIONAL_REL_TOL = 0.05
STALE_SIGNAL_DAYS_WARN = 3
MAX_RETRIES_WARN = 500
TRADING_DAYS = 252

# Privacy
MASK_ACCOUNT_LOGIN = True
SAVE_FULL_LOGIN_TO_FILES = False

# Performance quality controls
MIN_DAYS_FOR_ANNUALIZED_STATS = 60
PERFORMANCE_QUALITY_DEFAULT = "provisional"


# =========================================================
# Data classes
# =========================================================

@dataclass
class AccountSnapshot:
    login: int
    server: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    leverage: int
    currency: str
    timestamp: datetime


@dataclass
class HealthCheck:
    name: str
    status: str
    detail: str


# =========================================================
# Utilities
# =========================================================

def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_dt_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        try:
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        except Exception:
            pass
    return df


def mask_login(login: int) -> str:
    s = str(login)
    if len(s) <= 4:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


def safe_min_timestamp(values: List[Optional[pd.Timestamp]]) -> Optional[pd.Timestamp]:
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return min(vals)


def max_drawdown(index_series: pd.Series) -> float:
    index_series = index_series.dropna()
    if index_series.empty:
        return np.nan
    dd = index_series / index_series.cummax() - 1.0
    return float(dd.min())


def neg_annualized_vol_from_returns(rets: pd.Series, mar: float = 0.0, freq: int = 252) -> float:
    rets = rets.dropna()
    if rets.empty:
        return np.nan
    d = np.minimum(rets.to_numpy(dtype=float) - float(mar), 0.0)
    return float(np.sqrt(np.mean(d * d)) * np.sqrt(freq))


def perf_stats_from_equity(eq: pd.Series, freq: int = 252, mar_eq_ret: float = 0.0, min_days_for_ann: int = 60) -> Dict[str, float]:
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
    max_dd = float((eq / eq.cummax() - 1.0).min())

    if len(eq) < min_days_for_ann:
        return {
            "total_return": total_return,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "neg_ann_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_dd": max_dd,
        }

    years = (len(eq) - 1) / freq
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else np.nan
    ann_vol = float(rets.std(ddof=0) * np.sqrt(freq)) if len(rets) > 1 else np.nan
    neg_ann_vol = neg_annualized_vol_from_returns(rets, mar=mar_eq_ret, freq=freq)
    ann_ret = float(rets.mean() * freq) if len(rets) > 0 else np.nan
    sharpe = float(ann_ret / ann_vol) if ann_vol and ann_vol != 0 else np.nan
    sortino = float(ann_ret / neg_ann_vol) if neg_ann_vol and neg_ann_vol != 0 else np.nan

    return {
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "neg_ann_vol": neg_ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
    }


def rolling_alpha_beta(strat_ret: pd.Series, bench_ret: pd.Series, window: int = 63, ann_factor: int = 252) -> Tuple[pd.Series, pd.Series]:
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


# =========================================================
# MT5 source
# =========================================================

class MT5DataSource:
    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
        self.connected = True

    def shutdown(self) -> None:
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def get_account_snapshot(self) -> AccountSnapshot:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"account_info failed: {mt5.last_error()}")

        return AccountSnapshot(
            login=account.login,
            server=account.server,
            balance=account.balance,
            equity=account.equity,
            margin=account.margin,
            free_margin=account.margin_free,
            margin_level=account.margin_level,
            leverage=account.leverage,
            currency=account.currency,
            timestamp=datetime.now(),
        )

    def get_positions_df(self) -> pd.DataFrame:
        positions = mt5.positions_get()
        if not positions:
            return pd.DataFrame()
        df = pd.DataFrame([p._asdict() for p in positions])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_orders_df(self) -> pd.DataFrame:
        orders = mt5.orders_get()
        if not orders:
            return pd.DataFrame()
        df = pd.DataFrame([o._asdict() for o in orders])
        if "time_setup" in df.columns:
            df["time_setup"] = pd.to_datetime(df["time_setup"], unit="s")
        if "time_done" in df.columns:
            df["time_done"] = pd.to_datetime(df["time_done"], unit="s")
        return df

    def get_deals_df(self, date_from: datetime, date_to: datetime) -> pd.DataFrame:
        deals = mt5.history_deals_get(date_from, date_to)
        if not deals:
            return pd.DataFrame()
        df = pd.DataFrame([d._asdict() for d in deals])
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_symbol_info(self, symbol: str):
        return mt5.symbol_info(symbol)

    def get_tick(self, symbol: str):
        return mt5.symbol_info_tick(symbol)


# =========================================================
# EA state loader
# =========================================================

class EAStateLoader:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            return pd.DataFrame()

        df = pd.read_csv(self.csv_path)

        for col in ["timestamp", "signal_day", "last_roll_time"]:
            df = normalize_dt_col(df, col)

        for col in ["live_ok", "pending"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin(["1", "true", "yes"])

        return df


# =========================================================
# Exposure reconstruction
# =========================================================

class PortfolioReconstructor:
    def __init__(self, mt5_source: MT5DataSource) -> None:
        self.mt5_source = mt5_source

    @staticmethod
    def map_symbol_to_asset(symbol: str) -> Optional[str]:
        if not isinstance(symbol, str):
            return None
        for asset, prefix in EXPECTED_PREFIXES.items():
            if symbol.startswith(prefix):
                return asset
        return None

    def point_value(self, symbol: str) -> float:
        info = self.mt5_source.get_symbol_info(symbol)
        if info is None or info.trade_tick_size in (None, 0):
            return np.nan
        return info.trade_tick_value / info.trade_tick_size

    def mid_price(self, symbol: str) -> float:
        tk = self.mt5_source.get_tick(symbol)
        if tk is None or tk.bid <= 0 or tk.ask <= 0:
            return np.nan
        return 0.5 * (tk.bid + tk.ask)

    def build_positions_report(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        if positions_df.empty:
            return pd.DataFrame(columns=[
                "asset", "symbol", "type", "volume", "price_open", "price_current",
                "profit", "notional", "point_value", "opened_time"
            ])

        df = positions_df.copy()
        df["asset"] = df["symbol"].map(self.map_symbol_to_asset)
        df["point_value"] = df["symbol"].apply(self.point_value)
        df["mid_price"] = df["symbol"].apply(self.mid_price)
        df["notional"] = df["volume"] * df["mid_price"] * df["point_value"]
        df["opened_time"] = df["time"]

        keep_cols = [
            "asset", "symbol", "type", "volume", "price_open", "price_current",
            "profit", "notional", "point_value", "opened_time"
        ]
        return df[keep_cols].sort_values(["asset", "symbol", "opened_time"]).reset_index(drop=True)

    def build_asset_exposure_report(self, positions_df: pd.DataFrame, account_equity: float) -> pd.DataFrame:
        pos_report = self.build_positions_report(positions_df)

        if pos_report.empty:
            return pd.DataFrame({
                "asset": EXPECTED_ASSETS,
                "actual_lots": 0.0,
                "actual_notional": 0.0,
                "floating_pnl": 0.0,
                "n_positions": 0,
                "effective_weight_notional": 0.0,
            })

        out = (
            pos_report.groupby("asset", dropna=False)
            .agg(
                actual_lots=("volume", "sum"),
                actual_notional=("notional", "sum"),
                floating_pnl=("profit", "sum"),
                n_positions=("symbol", "count"),
            )
            .reset_index()
        )

        out["effective_weight_notional"] = out["actual_notional"] / account_equity if account_equity > 0 else np.nan

        all_assets = pd.DataFrame({"asset": EXPECTED_ASSETS})
        out = all_assets.merge(out, on="asset", how="left").fillna({
            "actual_lots": 0.0,
            "actual_notional": 0.0,
            "floating_pnl": 0.0,
            "n_positions": 0,
            "effective_weight_notional": 0.0,
        })

        return out

    def compute_gross_exposure(self, asset_exposure_df: pd.DataFrame, account_equity: float) -> float:
        if asset_exposure_df.empty or account_equity <= 0:
            return np.nan
        return asset_exposure_df["actual_notional"].abs().sum() / account_equity


# =========================================================
# Benchmark builder
# =========================================================

class BenchmarkBuilder:
    def __init__(self, ticker: str = BENCHMARK_SYMBOL_YF) -> None:
        self.ticker = ticker

    def fetch(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        df = yf.download(
            self.ticker,
            start=start_date.date(),
            end=(end_date + pd.Timedelta(days=2)).date(),
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame(columns=["date", "benchmark_close"])

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "date", "Close": "benchmark_close"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df[["date", "benchmark_close"]].copy()


# =========================================================
# Start-date detection
# =========================================================

class StrategyStartDetector:
    @staticmethod
    def manual_start() -> Optional[pd.Timestamp]:
        if MONITORING_INCEPTION_DATE is None:
            return None
        return pd.Timestamp(MONITORING_INCEPTION_DATE).normalize()

    @staticmethod
    def from_state_df(state_df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if state_df.empty:
            return None
        candidates = []

        if "signal_day" in state_df.columns:
            s = pd.to_datetime(state_df["signal_day"], errors="coerce").dropna()
            if not s.empty:
                candidates.append(s.min().normalize())

        if "timestamp" in state_df.columns:
            t = pd.to_datetime(state_df["timestamp"], errors="coerce").dropna()
            if not t.empty:
                candidates.append(t.min().normalize())

        return safe_min_timestamp(candidates)

    @staticmethod
    def from_deals_df(deals_df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if deals_df.empty or "time" not in deals_df.columns:
            return None
        df = deals_df.copy()
        df = df[df["symbol"].notna() & (df["symbol"] != "")]
        if df.empty:
            return None
        asset_mask = df["symbol"].astype(str).apply(
            lambda x: any(x.startswith(prefix) for prefix in EXPECTED_PREFIXES.values())
        )
        df = df[asset_mask]
        if df.empty:
            return None
        return pd.to_datetime(df["time"]).min().normalize()

    @staticmethod
    def from_positions_df(positions_df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if positions_df.empty or "time" not in positions_df.columns:
            return None
        df = positions_df.copy()
        asset_mask = df["symbol"].astype(str).apply(
            lambda x: any(x.startswith(prefix) for prefix in EXPECTED_PREFIXES.values())
        )
        df = df[asset_mask]
        if df.empty:
            return None
        return pd.to_datetime(df["time"]).min().normalize()

    def detect(
        self,
        state_df: pd.DataFrame,
        deals_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        fallback_start: pd.Timestamp,
    ) -> Tuple[pd.Timestamp, str]:
        manual = self.manual_start()
        if manual is not None:
            return manual, "manual_inception_date"

        inferred = safe_min_timestamp([
            self.from_state_df(state_df),
            self.from_deals_df(deals_df),
            self.from_positions_df(positions_df),
        ])
        if inferred is not None:
            return inferred, "inferred_from_state_or_activity"

        return fallback_start, "fallback_lookback"


# =========================================================
# Performance
# =========================================================

class PerformanceBuilder:
    def build_daily_equity_curve_from_deals(
        self,
        account_snapshot: AccountSnapshot,
        deals_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Provisional daily reconstruction:
        - realized cashflows from deals
        - latest account equity patched on the final day

        This is useful for short operational review but not certified NAV history.
        """
        date_index = pd.date_range(start_date, end_date, freq="B")
        out = pd.DataFrame({"date": date_index})

        if deals_df.empty:
            out["realized_cashflow"] = 0.0
            out["balance"] = account_snapshot.balance
            out["equity"] = account_snapshot.balance
            out["equity_source"] = "flat_no_deals_history"
            return out, "provisional"

        trade_deals = deals_df.copy()
        trade_deals = trade_deals[trade_deals["symbol"].notna() & (trade_deals["symbol"] != "")].copy()
        trade_deals = trade_deals[trade_deals["time"].dt.normalize() >= start_date]

        if trade_deals.empty:
            out["realized_cashflow"] = 0.0
            out["balance"] = account_snapshot.balance
            out["equity"] = account_snapshot.equity
            out["equity_source"] = "single_snapshot_from_inception"
            return out, "provisional"

        trade_deals["cashflow"] = (
            trade_deals["profit"].fillna(0.0)
            + trade_deals["commission"].fillna(0.0)
            + trade_deals["swap"].fillna(0.0)
        )
        trade_deals["date"] = pd.to_datetime(trade_deals["time"]).dt.normalize()

        daily_cashflows = trade_deals.groupby("date")["cashflow"].sum().sort_index()
        total_cashflows = float(trade_deals["cashflow"].sum())
        inferred_initial_balance = float(account_snapshot.balance - total_cashflows)

        out["realized_cashflow"] = out["date"].map(daily_cashflows).fillna(0.0)
        out["balance"] = inferred_initial_balance + out["realized_cashflow"].cumsum()
        out["equity"] = out["balance"]
        out["equity_source"] = "deal_based_reconstruction"

        if not out.empty:
            out.loc[out.index[-1], "equity"] = account_snapshot.equity
            out.loc[out.index[-1], "equity_source"] = "deal_based_plus_final_equity_patch"

        return out, "provisional"

    def merge_benchmark(self, perf_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
        out = perf_df.merge(benchmark_df, on="date", how="left")
        out["benchmark_close"] = out["benchmark_close"].ffill()
        out = out.dropna(subset=["benchmark_close"]).reset_index(drop=True)

        out["strategy_index"] = out["equity"] / out["equity"].iloc[0]
        out["balance_index"] = out["balance"] / out["balance"].iloc[0]
        out["benchmark_index"] = out["benchmark_close"] / out["benchmark_close"].iloc[0]

        out["strategy_ret"] = out["strategy_index"].pct_change().fillna(0.0)
        out["benchmark_ret"] = out["benchmark_index"].pct_change().fillna(0.0)

        return out

    def build_stats_table(self, perf_df: pd.DataFrame, performance_quality: str) -> pd.DataFrame:
        strat_stats = perf_stats_from_equity(
            perf_df["strategy_index"],
            freq=TRADING_DAYS,
            min_days_for_ann=MIN_DAYS_FOR_ANNUALIZED_STATS,
        )
        bench_stats = perf_stats_from_equity(
            perf_df["benchmark_index"],
            freq=TRADING_DAYS,
            min_days_for_ann=MIN_DAYS_FOR_ANNUALIZED_STATS,
        )

        beta, alpha = rolling_alpha_beta(
            perf_df["strategy_ret"],
            perf_df["benchmark_ret"],
            window=63,
            ann_factor=TRADING_DAYS,
        )

        latest_beta = beta.dropna().iloc[-1] if not beta.dropna().empty else np.nan
        latest_alpha = alpha.dropna().iloc[-1] if not alpha.dropna().empty else np.nan

        if len(perf_df) >= MIN_DAYS_FOR_ANNUALIZED_STATS:
            active_ret = perf_df["strategy_ret"] - perf_df["benchmark_ret"]
            tracking_error = active_ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
            info_ratio = np.nan
            if tracking_error and not pd.isna(tracking_error) and tracking_error != 0:
                info_ratio = active_ret.mean() * TRADING_DAYS / tracking_error
        else:
            tracking_error = np.nan
            info_ratio = np.nan

        stats = {
            "performance_quality": performance_quality,
            "start_date": perf_df["date"].min(),
            "end_date": perf_df["date"].max(),
            "n_business_days": len(perf_df),
            "strategy_total_return": strat_stats["total_return"],
            "strategy_cagr": strat_stats["cagr"],
            "strategy_ann_vol": strat_stats["ann_vol"],
            "strategy_neg_ann_vol": strat_stats["neg_ann_vol"],
            "strategy_sharpe": strat_stats["sharpe"],
            "strategy_sortino": strat_stats["sortino"],
            "strategy_max_dd": strat_stats["max_dd"],
            "benchmark_total_return": bench_stats["total_return"],
            "benchmark_cagr": bench_stats["cagr"],
            "benchmark_ann_vol": bench_stats["ann_vol"],
            "benchmark_max_dd": bench_stats["max_dd"],
            "latest_beta_vs_benchmark": latest_beta,
            "latest_alpha_vs_benchmark": latest_alpha,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "excess_total_return": (
                strat_stats["total_return"] - bench_stats["total_return"]
                if pd.notna(strat_stats["total_return"]) and pd.notna(bench_stats["total_return"])
                else np.nan
            ),
        }

        return pd.Series(stats, name="value").to_frame()

    def plot(self, perf_df: pd.DataFrame, output_dir: str, performance_quality: str) -> None:
        ensure_output_dir(output_dir)

        plt.figure(figsize=(12, 6))
        plt.plot(perf_df["date"], perf_df["strategy_index"], label="Strategy")
        plt.plot(perf_df["date"], perf_df["balance_index"], label="Balance")
        plt.plot(perf_df["date"], perf_df["benchmark_index"], label="ES Benchmark")
        plt.title(f"Cumulative Performance ({performance_quality.capitalize()})")
        plt.xlabel("Date")
        plt.ylabel("Index")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance.png"))
        plt.close()

        strat_dd = perf_df["strategy_index"] / perf_df["strategy_index"].cummax() - 1.0
        bal_dd = perf_df["balance_index"] / perf_df["balance_index"].cummax() - 1.0
        bench_dd = perf_df["benchmark_index"] / perf_df["benchmark_index"].cummax() - 1.0

        plt.figure(figsize=(12, 5))
        plt.plot(perf_df["date"], strat_dd, label="Strategy DD")
        plt.plot(perf_df["date"], bal_dd, label="Balance DD")
        plt.plot(perf_df["date"], bench_dd, label="ES Benchmark DD")
        plt.title(f"Drawdowns ({performance_quality.capitalize()})")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "drawdowns.png"))
        plt.close()


# =========================================================
# Implementation status
# =========================================================

class ImplementationStatusBuilder:
    def build(self, state_df: pd.DataFrame, asset_exposure_df: pd.DataFrame) -> pd.DataFrame:
        if state_df.empty:
            out = pd.DataFrame({"asset": EXPECTED_ASSETS})
            out["state_available"] = False
            out["implementation_certification"] = "unavailable"
            out["reason"] = "missing_ea_state_export"
            return out

        latest_state = state_df.copy()
        latest_state["asset"] = latest_state["asset"].astype(str)
        latest_state = latest_state.drop_duplicates(subset=["asset"], keep="last")

        out = latest_state.merge(asset_exposure_df, on="asset", how="outer")

        numeric_fill = {
            "target_weight": 0.0,
            "target_vol": 0.0,
            "leverage": 0.0,
            "target_notional": 0.0,
            "target_lots": 0.0,
            "actual_lots": 0.0,
            "actual_notional": 0.0,
            "floating_pnl": 0.0,
            "n_positions": 0,
        }
        for k, v in numeric_fill.items():
            if k in out.columns:
                out[k] = out[k].fillna(v)

        if {"actual_lots", "target_lots"}.issubset(out.columns):
            out["lots_gap"] = out["actual_lots"] - out["target_lots"]

        if {"actual_notional", "target_notional"}.issubset(out.columns):
            out["notional_gap"] = out["actual_notional"] - out["target_notional"]
            out["notional_gap_rel"] = np.where(
                out["target_notional"].abs() > 0,
                out["notional_gap"] / out["target_notional"],
                np.where(out["actual_notional"].abs() > 0, np.inf, 0.0)
            )

        out["state_available"] = True
        out["implementation_certification"] = "partial_or_available"

        preferred_cols = [
            "asset",
            "active_symbol",
            "target_weight",
            "target_vol",
            "leverage",
            "target_notional",
            "actual_notional",
            "notional_gap",
            "notional_gap_rel",
            "target_lots",
            "actual_lots",
            "lots_gap",
            "floating_pnl",
            "pending",
            "attempts_today",
            "live_ok",
            "ir_vs_es",
            "pair_penalty",
            "max_corr_to_stronger",
            "last_exec_retcode",
            "last_exec_comment",
            "last_roll_time",
            "last_rolled_from",
            "last_rolled_to",
            "n_positions",
            "state_available",
            "implementation_certification",
        ]

        existing_cols = [c for c in preferred_cols if c in out.columns]
        return out[existing_cols].sort_values("asset").reset_index(drop=True)


# =========================================================
# Health checks
# =========================================================

class HealthChecker:
    def run(
        self,
        account_snapshot: AccountSnapshot,
        positions_df: pd.DataFrame,
        state_df: pd.DataFrame,
        implementation_df: pd.DataFrame,
        performance_quality: str,
        strategy_start_source: str,
    ) -> pd.DataFrame:
        checks: List[HealthCheck] = []

        checks.append(HealthCheck(
            "performance_series_quality",
            "WARNING" if performance_quality != "certified" else "OK",
            f"Performance series quality is {performance_quality}. Source is not full daily marked-to-market NAV."
        ))

        checks.append(HealthCheck(
            "strategy_start_source",
            "OK",
            f"Strategy start source: {strategy_start_source}"
        ))

        if state_df.empty:
            checks.append(HealthCheck(
                "state_file_present",
                "WARNING",
                "EA state export not found or empty. Target-vs-actual certification is limited."
            ))
        else:
            checks.append(HealthCheck(
                "state_file_present",
                "OK",
                "EA state export loaded."
            ))

        if not state_df.empty:
            state_assets = sorted(state_df["asset"].astype(str).unique().tolist()) if "asset" in state_df.columns else []
            missing = sorted(set(EXPECTED_ASSETS) - set(state_assets))
            if missing:
                checks.append(HealthCheck(
                    "expected_assets_present",
                    "WARNING",
                    f"Missing assets in state export: {missing}"
                ))
            else:
                checks.append(HealthCheck(
                    "expected_assets_present",
                    "OK",
                    "All expected assets present in state export."
                ))

        if not state_df.empty and "target_weight" in state_df.columns:
            wsum = float(state_df["target_weight"].sum())
            if abs(wsum - 1.0) > 1e-4:
                checks.append(HealthCheck(
                    "weights_sum_to_one",
                    "FAIL",
                    f"Target weights sum to {wsum:.6f}, not 1.0."
                ))
            else:
                checks.append(HealthCheck(
                    "weights_sum_to_one",
                    "OK",
                    f"Target weights sum to {wsum:.6f}."
                ))

        if not state_df.empty and {"asset", "target_weight"}.issubset(state_df.columns):
            alt_weight = float(state_df.loc[state_df["asset"] != "ES", "target_weight"].sum())
            if alt_weight > TACTICAL_MAX_WEIGHT_EXPECTED + 1e-6:
                checks.append(HealthCheck(
                    "tactical_sleeve_cap",
                    "FAIL",
                    f"Alternative weight {alt_weight:.6f} exceeds expected cap {TACTICAL_MAX_WEIGHT_EXPECTED:.2f}."
                ))
            else:
                checks.append(HealthCheck(
                    "tactical_sleeve_cap",
                    "OK",
                    f"Alternative weight {alt_weight:.6f} within cap."
                ))

        if not state_df.empty and "leverage" in state_df.columns:
            bad = state_df[(state_df["leverage"] < -1e-8) | (state_df["leverage"] > MAX_LEVERAGE_EXPECTED + 1e-8)]
            if not bad.empty:
                checks.append(HealthCheck(
                    "leverage_bounds",
                    "FAIL",
                    f"Leverage outside bounds for assets: {bad['asset'].tolist()}"
                ))
            else:
                checks.append(HealthCheck(
                    "leverage_bounds",
                    "OK",
                    "All leverage values within expected bounds."
                ))

        if not positions_df.empty:
            symbols = positions_df["symbol"].astype(str).unique().tolist()
            unexpected = [sym for sym in symbols if not any(sym.startswith(prefix) for prefix in EXPECTED_PREFIXES.values())]
            if unexpected:
                checks.append(HealthCheck(
                    "unexpected_symbols_held",
                    "FAIL",
                    f"Unexpected open symbols found: {unexpected}"
                ))
            else:
                checks.append(HealthCheck(
                    "unexpected_symbols_held",
                    "OK",
                    "All open symbols belong to expected families."
                ))

        if (
            not implementation_df.empty
            and "state_available" in implementation_df.columns
            and "lots_gap" in implementation_df.columns
        ):
            pending_series = implementation_df["pending"] if "pending" in implementation_df.columns else pd.Series(False, index=implementation_df.index)
            bad = implementation_df[
                implementation_df["state_available"]
                & (implementation_df["lots_gap"].abs() > TARGET_ACTUAL_LOTS_TOL)
                & (~pending_series.fillna(False))
            ]
            if not bad.empty:
                checks.append(HealthCheck(
                    "target_vs_actual_lots",
                    "WARNING",
                    f"Lots mismatch without pending flag for assets: {bad['asset'].tolist()}"
                ))
            else:
                checks.append(HealthCheck(
                    "target_vs_actual_lots",
                    "OK",
                    "Target vs actual lots aligned within tolerance or explicitly pending."
                ))
        else:
            checks.append(HealthCheck(
                "target_vs_actual_lots",
                "WARNING",
                "Lots-gap check skipped because implementation state is incomplete."
            ))

        if (
            not implementation_df.empty
            and "state_available" in implementation_df.columns
            and "target_notional" in implementation_df.columns
            and "notional_gap_rel" in implementation_df.columns
        ):
            pending_series = implementation_df["pending"] if "pending" in implementation_df.columns else pd.Series(False, index=implementation_df.index)
            bad = implementation_df[
                implementation_df["state_available"]
                & implementation_df["target_notional"].abs().gt(0)
                & implementation_df["notional_gap_rel"].abs().gt(TARGET_ACTUAL_NOTIONAL_REL_TOL)
                & (~pending_series.fillna(False))
            ]
            if not bad.empty:
                checks.append(HealthCheck(
                    "target_vs_actual_notional",
                    "WARNING",
                    f"Notional gap beyond tolerance for assets: {bad['asset'].tolist()}"
                ))
            else:
                checks.append(HealthCheck(
                    "target_vs_actual_notional",
                    "OK",
                    "Target vs actual notional aligned within tolerance or pending."
                ))
        else:
            checks.append(HealthCheck(
                "target_vs_actual_notional",
                "WARNING",
                "Notional-gap check skipped because implementation state is incomplete."
            ))

        if not state_df.empty and "signal_day" in state_df.columns:
            latest_signal = pd.to_datetime(state_df["signal_day"], errors="coerce").max()
            if pd.isna(latest_signal):
                checks.append(HealthCheck(
                    "signal_freshness",
                    "WARNING",
                    "Signal day missing or unreadable."
                ))
            else:
                age_days = (pd.Timestamp.now().normalize() - latest_signal.normalize()).days
                if age_days > STALE_SIGNAL_DAYS_WARN:
                    checks.append(HealthCheck(
                        "signal_freshness",
                        "WARNING",
                        f"Latest signal day is {latest_signal.date()}, age {age_days} days."
                    ))
                else:
                    checks.append(HealthCheck(
                        "signal_freshness",
                        "OK",
                        f"Latest signal day is {latest_signal.date()}."
                    ))

        if not state_df.empty and {"pending", "attempts_today"}.issubset(state_df.columns):
            bad = state_df[state_df["pending"] & (state_df["attempts_today"] > MAX_RETRIES_WARN)]
            if not bad.empty:
                checks.append(HealthCheck(
                    "pending_retry_overload",
                    "WARNING",
                    f"High retry count for pending assets: {bad['asset'].tolist()}"
                ))
            else:
                checks.append(HealthCheck(
                    "pending_retry_overload",
                    "OK",
                    "No excessive pending retry count."
                ))

        if not positions_df.empty and "type" in positions_df.columns:
            sells = positions_df[positions_df["type"] != 0]
            if not sells.empty:
                checks.append(HealthCheck(
                    "long_only_constraint",
                    "FAIL",
                    f"Non-buy positions detected: {sells['symbol'].tolist()}"
                ))
            else:
                checks.append(HealthCheck(
                    "long_only_constraint",
                    "OK",
                    "Long-only constraint respected."
                ))

        if account_snapshot.equity <= 0:
            checks.append(HealthCheck(
                "equity_positive",
                "FAIL",
                f"Account equity non-positive: {account_snapshot.equity}"
            ))
        else:
            checks.append(HealthCheck(
                "equity_positive",
                "OK",
                f"Account equity positive: {account_snapshot.equity:.2f}"
            ))

        return pd.DataFrame([c.__dict__ for c in checks])


# =========================================================
# Reporting
# =========================================================

class ReportBuilder:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        ensure_output_dir(output_dir)

    def save_tables(
        self,
        account_snapshot: AccountSnapshot,
        positions_report: pd.DataFrame,
        asset_exposure_report: pd.DataFrame,
        implementation_report: pd.DataFrame,
        health_checks: pd.DataFrame,
        perf_stats: pd.DataFrame,
        perf_df: pd.DataFrame,
        deals_df: pd.DataFrame,
        orders_df: pd.DataFrame,
        meta_df: pd.DataFrame,
    ) -> None:
        account_payload = account_snapshot.__dict__.copy()
        if MASK_ACCOUNT_LOGIN and not SAVE_FULL_LOGIN_TO_FILES:
            account_payload["login"] = mask_login(account_snapshot.login)

        pd.Series(account_payload, name="value").to_frame().to_csv(
            os.path.join(self.output_dir, "account_snapshot.csv")
        )
        positions_report.to_csv(os.path.join(self.output_dir, "positions_report.csv"), index=False)
        asset_exposure_report.to_csv(os.path.join(self.output_dir, "asset_exposure_report.csv"), index=False)
        implementation_report.to_csv(os.path.join(self.output_dir, "implementation_report.csv"), index=False)
        health_checks.to_csv(os.path.join(self.output_dir, "health_checks.csv"), index=False)
        perf_stats.to_csv(os.path.join(self.output_dir, "performance_stats.csv"))
        perf_df.to_csv(os.path.join(self.output_dir, "performance_timeseries.csv"), index=False)
        deals_df.to_csv(os.path.join(self.output_dir, "deals_history.csv"), index=False)
        orders_df.to_csv(os.path.join(self.output_dir, "orders_snapshot.csv"), index=False)
        meta_df.to_csv(os.path.join(self.output_dir, "monitor_metadata.csv"), index=False)

    def print_console_summary(
        self,
        account_snapshot: AccountSnapshot,
        gross_exposure: float,
        implementation_report: pd.DataFrame,
        health_checks: pd.DataFrame,
        perf_stats: pd.DataFrame,
        strategy_start: pd.Timestamp,
        strategy_start_source: str,
        performance_quality: str,
    ) -> None:
        login_display = mask_login(account_snapshot.login) if MASK_ACCOUNT_LOGIN else str(account_snapshot.login)

        print("\n================ ACCOUNT SNAPSHOT ================")
        print(f"Login:          {login_display}")
        print(f"Server:         {account_snapshot.server}")
        print(f"Timestamp:      {account_snapshot.timestamp}")
        print(f"Balance:        {account_snapshot.balance:,.2f} {account_snapshot.currency}")
        print(f"Equity:         {account_snapshot.equity:,.2f} {account_snapshot.currency}")
        print(f"Margin:         {account_snapshot.margin:,.2f}")
        print(f"Free Margin:    {account_snapshot.free_margin:,.2f}")
        print(f"Margin Level:   {account_snapshot.margin_level:,.2f}")
        print(f"Account Lev:    {account_snapshot.leverage}")
        print(f"Strategy Start: {strategy_start.date() if pd.notna(strategy_start) else 'NA'}")
        print(f"Start Source:   {strategy_start_source}")
        print(f"Perf Quality:   {performance_quality}")
        print(f"Gross Exposure: {gross_exposure:,.4f}x")

        print("\n================ PERFORMANCE =====================")
        print(perf_stats.to_string())

        print("\n================ IMPLEMENTATION ==================")
        if implementation_report.empty:
            print("No implementation report available.")
        else:
            cols = [c for c in [
                "asset", "active_symbol", "target_weight", "leverage",
                "target_lots", "actual_lots", "lots_gap", "pending", "last_exec_retcode",
                "implementation_certification"
            ] if c in implementation_report.columns]
            print(implementation_report[cols].to_string(index=False))

        print("\n================ HEALTH CHECKS ===================")
        if health_checks.empty:
            print("No health checks.")
        else:
            print(health_checks.to_string(index=False))


# =========================================================
# Main
# =========================================================

def main() -> None:
    mt5_source = MT5DataSource()
    state_loader = EAStateLoader(STATE_CSV_PATH)
    reconstructor = PortfolioReconstructor(mt5_source)
    benchmark_builder = BenchmarkBuilder()
    start_detector = StrategyStartDetector()
    performance_builder = PerformanceBuilder()
    implementation_builder = ImplementationStatusBuilder()
    checker = HealthChecker()
    reporter = ReportBuilder(OUTPUT_DIR)

    try:
        mt5_source.connect()

        account_snapshot = mt5_source.get_account_snapshot()
        positions_df = mt5_source.get_positions_df()
        orders_df = mt5_source.get_orders_df()

        date_to = datetime.now()
        date_from = date_to - timedelta(days=LOOKBACK_DAYS_DEALS)
        deals_df = mt5_source.get_deals_df(date_from=date_from, date_to=date_to)

        state_df = state_loader.load()

        positions_report = reconstructor.build_positions_report(positions_df)
        asset_exposure_report = reconstructor.build_asset_exposure_report(positions_df, account_snapshot.equity)
        gross_exposure = reconstructor.compute_gross_exposure(asset_exposure_report, account_snapshot.equity)

        implementation_report = implementation_builder.build(state_df, asset_exposure_report)

        fallback_start = pd.Timestamp(datetime.now().date() - timedelta(days=LOOKBACK_DAYS_EQUITY))
        strategy_start, strategy_start_source = start_detector.detect(
            state_df=state_df,
            deals_df=deals_df,
            positions_df=positions_df,
            fallback_start=fallback_start,
        )

        perf_end = pd.Timestamp(datetime.now().date())
        perf_df, performance_quality = performance_builder.build_daily_equity_curve_from_deals(
            account_snapshot=account_snapshot,
            deals_df=deals_df,
            start_date=strategy_start,
            end_date=perf_end,
        )

        benchmark_df = benchmark_builder.fetch(
            start_date=perf_df["date"].min(),
            end_date=perf_df["date"].max(),
        )
        if benchmark_df.empty:
            raise RuntimeError("Benchmark download failed or returned empty data.")

        perf_df = performance_builder.merge_benchmark(perf_df, benchmark_df)
        perf_stats = performance_builder.build_stats_table(perf_df, performance_quality)
        performance_builder.plot(perf_df, OUTPUT_DIR, performance_quality)

        health_checks = checker.run(
            account_snapshot=account_snapshot,
            positions_df=positions_df,
            state_df=state_df,
            implementation_df=implementation_report,
            performance_quality=performance_quality,
            strategy_start_source=strategy_start_source,
        )

        meta_login = mask_login(account_snapshot.login) if MASK_ACCOUNT_LOGIN and not SAVE_FULL_LOGIN_TO_FILES else account_snapshot.login

        meta_df = pd.DataFrame([{
            "run_timestamp": datetime.now(),
            "account_login": meta_login,
            "strategy_start_used": strategy_start,
            "strategy_start_source": strategy_start_source,
            "performance_quality": performance_quality,
            "state_file_present": not state_df.empty,
            "positions_count": 0 if positions_df.empty else len(positions_df),
            "deals_count": 0 if deals_df.empty else len(deals_df),
            "orders_count": 0 if orders_df.empty else len(orders_df),
            "gross_exposure": gross_exposure,
            "state_csv_path": STATE_CSV_PATH,
            "output_dir": OUTPUT_DIR,
        }])

        reporter.save_tables(
            account_snapshot=account_snapshot,
            positions_report=positions_report,
            asset_exposure_report=asset_exposure_report,
            implementation_report=implementation_report,
            health_checks=health_checks,
            perf_stats=perf_stats,
            perf_df=perf_df,
            deals_df=deals_df,
            orders_df=orders_df,
            meta_df=meta_df,
        )

        reporter.print_console_summary(
            account_snapshot=account_snapshot,
            gross_exposure=gross_exposure,
            implementation_report=implementation_report,
            health_checks=health_checks,
            perf_stats=perf_stats,
            strategy_start=strategy_start,
            strategy_start_source=strategy_start_source,
            performance_quality=performance_quality,
        )

        print(f"\nSaved outputs to: {OUTPUT_DIR}")

    finally:
        mt5_source.shutdown()


if __name__ == "__main__":
    main()