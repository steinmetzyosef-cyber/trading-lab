#!/usr/bin/env python3
"""
TRADING LAB v2  (single file)  -  paper trading only by default. Not financial advice.
Most traders lose money. Nothing here guarantees profit; the backtester exists to tell you the truth.

SETUP (once):   pip install alpaca-py pandas numpy streamlit
KEYS:           ALPACA_KEY / ALPACA_SECRET (free paper account). Optional FINNHUB_KEY (free, earnings dates).

WHAT'S INSIDE
  MARKET   : regime check (SPY/QQQ/IWM trend + strength), sector leaders, breadth, earnings calendar
  SWING    : scans thousands of US stocks (daily bars) for pullbacks-in-uptrends and breakouts,
             ranks them by momentum / relative strength, sizes by risk, trailing-stop exits
  DAY      : 1-minute scanner on a dynamic list (most actives + top movers + your list), gated by
             the market's intraday direction (SPY vs VWAP)
  BACKTEST : both engines, realistic fills + costs, benchmark vs SPY, Monte Carlo, slippage stress test,
             walk-forward (train on past, test on unseen future)

COMMANDS
  python day_trading_lab.py dashboard
  python day_trading_lab.py swing-backtest --synthetic          # code test on random data
  python day_trading_lab.py swing-backtest --start 2023-01-03 --max-symbols 400
  python day_trading_lab.py swing-walkforward --start 2022-01-03
  python day_trading_lab.py swing-scan
  python day_trading_lab.py backtest --start 2026-06-01 --side both      # 1-minute day trading
  python day_trading_lab.py sweep --synthetic
  python day_trading_lab.py live                                # paper day-trading bot
"""
import os, sys, time, json, html, base64, pickle, argparse, itertools, subprocess, tempfile
import urllib.request, urllib.parse
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "SPY", "QQQ", "NFLX", "AVGO",
                     "PLTR", "SOFI", "COIN", "MSTR", "SMCI", "MU", "INTC", "BAC", "F", "NIO", "RIVN", "SNAP", "HOOD", "UBER"]
SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
           "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
           "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication"}
INDEX_ETFS = ["SPY", "QQQ", "IWM"]
ETF_SET = set(SECTORS) | set(INDEX_ETFS)
ETF_WORDS = ["etf", "etn", "fund", "ishares", "proshares", "direxion", "spdr", "invesco", "vanguard", "wisdomtree", "vaneck", "index"]
NON_STOCK_WORDS = ["warrant", "preferred", "depositary", "notes due", "rights"]
BIAS_NOTE = ("Universe bias: the stock list is built from TODAY's liquid, still-listed US stocks, so history "
             "ignores companies that failed or were delisted. Real results would be somewhat worse than this backtest.")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "dtlab_cache")


def cache_get(name, ttl_hours):
    try:
        f = os.path.join(_CACHE_DIR, name + ".pkl")
        if os.path.exists(f) and time.time() - os.path.getmtime(f) < ttl_hours * 3600:
            with open(f, "rb") as fh:
                return pickle.load(fh)
    except Exception:
        pass
    return None


def cache_set(name, obj):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, name + ".pkl"), "wb") as fh:
            pickle.dump(obj, fh)
    except Exception:
        pass


# =====================================================================
# 1. DAY-TRADING STRATEGY (1-minute bars, five setups)
# =====================================================================
KIND_NAMES = {1: "Momentum breakout", 2: "VWAP bounce", 3: "Opening-range breakout", 4: "Gap & go", 5: "Day runner"}
SETUP_CODES = {"breakout": 1, "vwap": 2, "orb": 3, "gap": 4, "runner": 5}


@dataclass
class Params:
    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.03
    rr: float = 2.0                 # target = 2x the risk
    atr_mult: float = 1.5           # stop = 1.5 x 1-min ATR ...
    min_stop_pct: float = 0.006     # ... but never tighter than 0.6% (double for stocks under $5)
    min_rvol: float = 1.2           # volume vs its own recent average
    rsi_long: tuple = (50, 80)
    rsi_short: tuple = (20, 50)
    max_open: int = 4
    max_pos_pct: float = 0.25
    cooldown_min: int = 10
    start_after: tuple = (9, 40)
    last_entry: tuple = (15, 30)    # no new trades after 3:30pm
    flatten_at: tuple = (15, 50)
    allow_long: bool = True
    allow_short: bool = False
    min_price: float = 1.0          # penny-stock floor (exchange-listed only)
    max_price: float = 1000.0
    min_active: int = 3             # bars with trades out of the last 5 (free IEX feed is sparse)
    market_gate: str = "soft"       # off | soft | strict  (SPY above/below VWAP)
    setups: tuple = ("breakout", "vwap", "orb", "gap", "runner")
    fresh_bars: int = 5             # a signal stays actionable for this many minutes
    gap_min: float = 0.03
    runner_min: float = 0.04
    hot_top: int = 150              # how many of the hottest stocks get the 1-minute treatment
    include_etf: bool = False


def indicators(df, pc=None):
    """ONE trading day of 1-min bars (optionally with yesterday's close). Past data only."""
    df = df.copy()
    tp = (df.high + df.low + df.close) / 3
    df["vwap"] = ((tp * df.volume).cumsum() / df.volume.cumsum().replace(0, np.nan)).ffill().fillna(df.close)
    delta = df.close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=8).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=8).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=8).mean()
    df["rvol"] = df.volume / df.volume.rolling(30, min_periods=10).mean()
    df["hi20"] = df.high.rolling(20, min_periods=10).max().shift()
    df["lo20"] = df.low.rolling(20, min_periods=10).min().shift()
    df["ema9"] = df.close.ewm(span=9, adjust=False).mean()
    df["ema20"] = df.close.ewm(span=20, adjust=False).mean()
    df["act"] = (df.volume > 0).rolling(5, min_periods=1).sum()
    df["hod"], df["lod"] = df.high.cummax(), df.low.cummin()
    df["lowmin4"], df["highmax4"] = df.low.rolling(4, min_periods=1).min(), df.high.rolling(4, min_periods=1).max()
    mins = np.asarray(df.index.hour * 60 + df.index.minute)
    om = mins < 585                                            # first 15 minutes = opening range
    df["orh"] = np.where(mins >= 585, df.high[om].max() if om.any() else np.nan, np.nan)
    df["orl"] = np.where(mins >= 585, df.low[om].min() if om.any() else np.nan, np.nan)
    day_open = float(df.open.iloc[0])
    df["chg"] = df.close / pc - 1 if pc else df.close / day_open - 1
    df["gap"] = (day_open / pc - 1) if pc else 0.0
    return df


def prep_day(df, pc=None):
    d = indicators(df, pc)
    cols = ["open", "high", "low", "close", "volume", "atr", "vwap", "rsi", "rvol", "hi20", "lo20", "ema9", "ema20",
            "act", "hod", "lod", "orh", "orl", "chg", "gap", "lowmin4", "highmax4"]
    out = {k: d[k].to_numpy(float) for k in cols}
    out["ts"] = d.index
    out["mins"] = (d.index.hour * 60 + d.index.minute).to_numpy()
    out["pc"] = pc
    return out


def signals(d, p, with_kind=False):
    """Per bar: +1 long, -1 short, 0 none (and which of the 5 setups fired). Live uses the last few bars."""
    m, n = d["mins"], len(d["mins"])
    lo, hi = p.start_after[0] * 60 + p.start_after[1], p.last_entry[0] * 60 + p.last_entry[1]
    c, o = d["close"], d["open"]
    sig, kind = np.zeros(n, int), np.zeros(n, int)
    with np.errstate(invalid="ignore"):
        base = ((m >= lo) & (m < hi) & (d["atr"] > 0) & (c >= p.min_price) & (c <= p.max_price)
                & (d["act"] >= p.min_active))
        rv, vw, e9, e20 = d["rvol"], d["vwap"], d["ema9"], d["ema20"]
        mk = d.get("mkt")

        def gate(side):                      # soft = trading against the market needs 1.5x the volume
            if mk is None or p.market_gate == "off":
                return np.ones(n, bool)
            against = (mk * side) < 0
            return ~against if p.market_gate == "strict" else (~against | (rv >= 1.5 * p.min_rvol))

        orb_t = m <= 11 * 60 + 30
        L = {2: (e9 > e20) & (c > vw) & (d["lowmin4"] <= np.maximum(vw, e20) * 1.002) & (c > e9) & (c > o) & (rv >= 0.7 * p.min_rvol),
             1: (c > vw) & (c > d["hi20"]) & (rv >= p.min_rvol) & (d["rsi"] >= p.rsi_long[0]) & (d["rsi"] <= p.rsi_long[1]),
             5: (d["chg"] >= p.runner_min) & (c > vw) & (c >= d["hod"] * 0.995) & (c > e9) & (rv >= p.min_rvol),
             3: orb_t & (c > d["orh"]) & (c <= d["orh"] * 1.01) & (c > vw) & (rv >= 0.8 * p.min_rvol),
             4: (d["gap"] >= p.gap_min) & (c > d["orh"]) & (c > vw) & (rv >= p.min_rvol)}
        S = {2: (e9 < e20) & (c < vw) & (d["highmax4"] >= np.minimum(vw, e20) * 0.998) & (c < e9) & (c < o) & (rv >= 0.7 * p.min_rvol),
             1: (c < vw) & (c < d["lo20"]) & (rv >= p.min_rvol) & (d["rsi"] >= p.rsi_short[0]) & (d["rsi"] <= p.rsi_short[1]),
             5: (d["chg"] <= -p.runner_min) & (c < vw) & (c <= d["lod"] * 1.005) & (c < e9) & (rv >= p.min_rvol),
             3: orb_t & (c < d["orl"]) & (c >= d["orl"] * 0.99) & (c < vw) & (rv >= 0.8 * p.min_rvol),
             4: (d["gap"] <= -p.gap_min) & (c < d["orl"]) & (c < vw) & (rv >= p.min_rvol)}
        names = {v: k for k, v in SETUP_CODES.items()}
        for code in (2, 1, 5, 3, 4):                         # later = higher priority
            if names[code] not in p.setups:
                continue
            if p.allow_long:
                msk = base & L[code] & gate(1)
                sig[msk], kind[msk] = 1, code
            if p.allow_short:
                msk = base & S[code] & gate(-1) & (c >= 5) & (sig <= 0)
                sig[msk], kind[msk] = -1, code
    return (sig, kind) if with_kind else sig


def levels(entry, atr, side, p):
    risk = max(p.atr_mult * atr, p.min_stop_pct * entry * (2.0 if entry < 5 else 1.0))
    return (entry - side * risk, entry + side * p.rr * risk)    # (stop, target)


def size(equity, entry, stop, p):
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    cap = p.max_pos_pct * (0.4 if entry < 5 else 1.0)           # smaller positions in penny stocks
    return max(0, min(int(equity * p.risk_per_trade / per_share), int(equity * cap / entry)))


def rnd(x):
    return round(float(x), 2 if x >= 1 else 4)


# =====================================================================
# 2. DATA LOADERS
# =====================================================================
def load_alpaca(symbols, start, end, key=None, secret=None):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    key, secret = key or os.environ["ALPACA_KEY"], secret or os.environ["ALPACA_SECRET"]
    client = StockHistoricalDataClient(key, secret)
    symbols = list(dict.fromkeys(list(symbols) + ["SPY"]))     # SPY powers the market gate
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=pd.Timestamp(start, tz=NY),
                           end=min(pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1),
                                   pd.Timestamp.now(tz=NY) - pd.Timedelta(minutes=20)),
                           feed=DataFeed.IEX)
    df = client.get_stock_bars(req).df
    out = {}
    for s in symbols:
        if s in df.index.get_level_values(0):
            g = df.loc[s].copy()
            g.index = g.index.tz_convert(NY)
            out[s] = g.between_time("09:30", "15:59")
    return out


def load_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY)
    return {s: g.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume"]]
            .between_time("09:30", "15:59") for s, g in df.groupby("symbol")}


def synthetic(symbols, days=60, seed=7):
    """Random 1-minute data (with some trending bursts and gaps) to test code. Says NOTHING about profitability."""
    rng, out = np.random.default_rng(seed), {}
    bdays = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days)
    for s in dict.fromkeys(list(symbols) + ["SPY"]):
        price, frames = 8 + rng.random() * 250, []
        for d in bdays:
            idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="min", tz=NY)
            ret = rng.normal(0, 0.0008, 390)
            ret[rng.integers(0, 330):][:25] += rng.normal(0.0007, 0.0007)
            if rng.random() < 0.12:
                ret[:20] += rng.normal(0.0012, 0.0006)                  # a morning "gap and go" style push
            close = price * np.exp(rng.normal(0, 0.01) + np.cumsum(ret))
            open_ = np.r_[price * np.exp(rng.normal(0, 0.01)), close[:-1]]
            wick = np.abs(rng.normal(0, 0.0004, 390)) * close
            vol = 1e5 * np.exp(rng.normal(0, 0.4, 390)) * (1 + 3 * (np.abs(ret) > 0.0015))
            frames.append(pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                                        "low": np.minimum(open_, close) - wick,
                                        "close": close, "volume": vol}, index=idx))
            price = close[-1]
        out[s] = pd.concat(frames)
    return out


def prep_data(data):
    """{symbol: 1-min bars} -> {day: {symbol: arrays}} with yesterday's close and the SPY gate attached."""
    prep, mk = {}, {}
    if "SPY" in data:
        for day, g in data["SPY"].groupby(data["SPY"].index.date):
            if len(g) > 60:
                s = indicators(g)
                mk[day] = pd.Series(np.sign(s.close - s.vwap).to_numpy(), index=s.index)
    for sym, df in data.items():
        pc = None
        for day, g in df.groupby(df.index.date):
            if len(g) > 60:
                d = prep_day(g, pc)
                if day in mk:
                    d["mkt"] = mk[day].reindex(d["ts"]).ffill().fillna(0).to_numpy()
                prep.setdefault(day, {})[sym] = d
                pc = float(g.close.iloc[-1])
    return prep


# =====================================================================
# 3. DAY-TRADING BACKTEST
# =====================================================================
def day_trades(sym, d, p, slip, comm):
    sig, kd = signals(d, p, True)
    n = len(sig)
    flat = p.flatten_at[0] * 60 + p.flatten_at[1]
    o, h, l, c, m = d["open"], d["high"], d["low"], d["close"], d["mins"]
    out, last_j, cool = [], -1, -1
    for i in np.flatnonzero(sig):
        if i <= last_j or i >= n - 1 or m[i] < cool:
            continue
        side = int(sig[i])
        entry = o[i + 1] * (1 + slip * side)
        stop, target = levels(entry, d["atr"][i], side, p)
        px = reason = None
        for j in range(i + 1, n):
            if side == 1:
                if l[j] <= stop: px, reason = min(o[j], stop), "stop"
                elif h[j] >= target: px, reason = max(o[j], target), "target"
            else:
                if h[j] >= stop: px, reason = max(o[j], stop), "stop"
                elif l[j] <= target: px, reason = min(o[j], target), "target"
            if px is None and (m[j] >= flat or j == n - 1):
                px, reason = c[j], "time"
            if px is not None:
                break
        px *= (1 - slip * side)
        out.append(dict(symbol=sym, side="long" if side == 1 else "short", dir=side, kind=KIND_NAMES[int(kd[i])],
                        entry_time=d["ts"][i + 1], exit_time=d["ts"][j],
                        entry=entry, exit=px, stop=stop, reason=reason))
        last_j, cool = j, m[j] + p.cooldown_min
    return out


def run(prep, p, eq0=25000.0, slip=0.0005, comm=0.0):
    equity, accepted, curve = eq0, [], {}
    for day in sorted(prep):
        cand = []
        for sym, d in prep[day].items():
            cand += day_trades(sym, d, p, slip, comm)
        cand.sort(key=lambda t: t["entry_time"])
        start, open_, taken = equity, [], []
        for t in cand:
            open_ = [x for x in open_ if x["exit_time"] > t["entry_time"]]
            realized = sum(x["pnl"] for x in taken if x["exit_time"] <= t["entry_time"])
            if realized <= -p.daily_loss_limit * start or len(open_) >= p.max_open:
                continue
            qty = size(start, t["entry"], t["stop"], p)
            if qty == 0:
                continue
            t["qty"] = qty
            t["pnl"] = (t["exit"] - t["entry"]) * t["dir"] * qty - 2 * comm * qty
            t["R"] = t["pnl"] / (abs(t["entry"] - t["stop"]) * qty)
            t["ret"] = t["pnl"] / start
            open_.append(t); taken.append(t)
        equity += sum(x["pnl"] for x in taken)
        accepted += taken
        curve[pd.Timestamp(day)] = equity
    return pd.DataFrame(accepted), pd.Series(curve, dtype=float)


# =====================================================================
# 4. SHARED ANALYTICS (stats, Monte Carlo, slippage stress, printing)
# =====================================================================
def stats(tr, cv, eq0):
    nan = float("nan")
    if tr.empty or cv.empty:
        return dict(days=len(cv), trades=0, win_rate=nan, avg_R=nan, expectancy=nan, profit_factor=nan,
                    ret=0.0, cagr=0.0, max_dd=0.0, sharpe=nan, sortino=nan, end_equity=eq0, avg_hold=nan)
    w, l = tr.loc[tr.pnl > 0, "pnl"].sum(), -tr.loc[tr.pnl <= 0, "pnl"].sum()
    full = pd.concat([pd.Series([eq0]), cv.reset_index(drop=True)])
    daily = full.pct_change().dropna()
    neg = daily[daily < 0]
    yrs = max(len(cv) / 252, 1e-9)
    end = float(cv.iloc[-1])
    return dict(days=len(cv), trades=len(tr), win_rate=(tr.pnl > 0).mean(), avg_R=tr.R.mean(),
                expectancy=tr.ret.mean() if "ret" in tr else nan,
                profit_factor=w / l if l else float("inf"), ret=end / eq0 - 1,
                cagr=(end / eq0) ** (1 / yrs) - 1 if end > 0 else -1.0,
                max_dd=(full / full.cummax() - 1).min(), end_equity=end,
                sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else nan,
                sortino=daily.mean() / neg.std() * np.sqrt(252) if len(neg) > 1 and neg.std() > 0 else nan,
                avg_hold=tr.hold.mean() if "hold" in tr else nan)


def monte_carlo(tr, n=1000, seed=1):
    """Reshuffle/resample the trade returns to see the RANGE of outcomes luck alone could produce."""
    if tr.empty or "ret" not in tr or len(tr) < 20:
        return None
    r = tr["ret"].to_numpy(float)
    n = min(n, max(100, int(2_000_000 / len(r))))
    rng = np.random.default_rng(seed)
    paths = np.cumprod(1 + r[rng.integers(0, len(r), (n, len(r)))], axis=1)
    final = paths[:, -1] - 1
    dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    return dict(ret_p5=np.percentile(final, 5), ret_p50=np.percentile(final, 50), ret_p95=np.percentile(final, 95),
                dd_p50=np.percentile(dd, 50), dd_p95=np.percentile(dd, 5), prob_loss=(final < 0).mean())


def stress_table(runner, slip, eq0, mults=(1, 2, 3)):
    """Re-run the same backtest with 1x/2x/3x slippage. An edge that dies at 2x was never real."""
    rows = []
    for m in mults:
        tr, cv = runner(slip * m)
        s = stats(tr, cv, eq0)
        rows.append({"slippage": f"{m}x ({slip * m * 1e4:.0f} bps/side)", "trades": s["trades"],
                     "profit_factor": round(s["profit_factor"], 2) if s["trades"] else np.nan,
                     "return": f"{s['ret']:+.1%}", "max_drawdown": f"{s['max_dd']:.1%}"})
    return pd.DataFrame(rows)


def yearly_returns(cv, eq0):
    if cv.empty:
        return pd.Series(dtype=float)
    ye = cv.groupby(cv.index.year).last()
    return (ye / ye.shift(1).fillna(eq0) - 1)


def print_stats(s, tr, eq0, bench=None, title="BACKTEST RESULTS"):
    print(f"\n========== {title} ==========")
    print(f"Days tested     : {s['days']}")
    print(f"Trades          : {s['trades']}")
    if s["trades"]:
        print(f"Win rate        : {s['win_rate']:.1%}")
        print(f"Avg R / trade   : {s['avg_R']:.2f}   (1R = the amount risked)")
        print(f"Profit factor   : {s['profit_factor']:.2f}")
        print(f"Equity          : ${eq0:,.0f} -> ${s['end_equity']:,.0f} ({s['ret']:+.1%}, CAGR {s['cagr']:+.1%})")
        print(f"Max drawdown    : {s['max_dd']:.1%}")
        print(f"Sharpe / Sortino: {s['sharpe']:.2f} / {s['sortino']:.2f}")
        if s["avg_hold"] == s["avg_hold"]:
            print(f"Avg hold        : {s['avg_hold']:.1f} days")
        print("Exits           :", tr.reason.value_counts().to_dict())
        if "kind" in tr:
            print("By setup        :", tr.groupby("kind").pnl.agg(["count", "sum"]).round(0).to_dict("index"))
    if bench is not None:
        print(f"SPY buy & hold  : {bench['ret']:+.1%} (max drawdown {bench['dd']:.1%})")
    print("=" * 40)


def bench_stats(spy, cv):
    if spy is None or cv.empty:
        return None
    b = spy.reindex(cv.index).ffill().dropna()
    if len(b) < 2:
        return None
    return dict(curve=b / b.iloc[0], ret=b.iloc[-1] / b.iloc[0] - 1, dd=(b / b.cummax() - 1).min())


# =====================================================================
# 5. SWING ENGINE (daily bars)
# =====================================================================
@dataclass
class SwingParams:
    setup: str = "all"             # pullback | breakout | leader | both (pullback+breakout) | all
    risk_per_trade: float = 0.01
    max_positions: int = 8
    max_pos_pct: float = 0.15
    stop_atr: float = 2.0          # initial stop = entry - 2 ATR
    trail_atr: float = 3.5         # chandelier trailing stop = highest close - 3.5 ATR
    target_rr: float = 3.0         # profit target = 3x the risk (0 = trail only)
    max_hold: int = 25             # time stop (trading days)
    adx_min: float = 20.0          # trend strength required
    rsi_pull: tuple = (30, 50)
    breakout_n: int = 55
    min_rvol_bo: float = 1.3
    min_price: float = 5.0
    min_dollar_vol: float = 10e6   # 20-day average $ volume
    use_regime: bool = True        # only enter when SPY > its 200-day average
    trend_exit: bool = True        # breakout trades exit on a close below the 50-day average
    earnings_buffer: int = 5       # live only: skip stocks reporting within N days


def _wilder(df, n=14):
    return df.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bars_to_panels(df, min_bars=220):
    """Alpaca multi-index (symbol, timestamp) daily bars -> wide panels O/H/L/C/V (date x symbol)."""
    d = df.reset_index()
    d["date"] = pd.to_datetime(d["timestamp"], utc=True).dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    P = {k: d.pivot(index="date", columns="symbol", values=col).sort_index().astype("float32")
         for k, col in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close"), ("V", "volume"))}
    if min_bars:
        keep = [c for c in P["C"].columns if P["C"][c].count() >= min_bars or c in ETF_SET]
        P = {k: v[keep] for k, v in P.items()}
    return P


def swing_features(P):
    C, H, L, V = P["C"], P["H"], P["L"], P["V"]
    F = {"sma50": C.rolling(50).mean(), "sma200": C.rolling(200).mean(), "ema21": C.ewm(span=21, adjust=False).mean()}
    d = C.diff()
    F["rsi"] = 100 - 100 / (1 + _wilder(d.clip(lower=0)) / _wilder((-d).clip(lower=0)))
    pc = C.shift()
    atr = _wilder(np.maximum(H - L, np.maximum((H - pc).abs(), (L - pc).abs())))
    up, dn = H.diff(), -L.diff()
    pdi = 100 * _wilder(up.where((up > dn) & (up > 0), 0.0)) / atr
    mdi = 100 * _wilder(dn.where((dn > up) & (dn > 0), 0.0)) / atr
    F["atr"] = atr
    F["adx"] = _wilder(100 * (pdi - mdi).abs() / (pdi + mdi))
    F["rvol"] = V / V.rolling(20).mean().shift()
    F["dvol"] = (C * V).rolling(20).mean()
    F["ret63"] = C.pct_change(63, fill_method=None)
    F["ret126"] = C.pct_change(126, fill_method=None)
    F["rs63"] = F["ret63"].sub(F["ret63"]["SPY"], axis=0) if "SPY" in C else F["ret63"]
    F["score"] = (0.4 * F["ret126"].rank(axis=1, pct=True) + 0.4 * F["ret63"].rank(axis=1, pct=True)
                  + 0.2 * F["rvol"].rank(axis=1, pct=True))
    F["hi52"] = H.rolling(252, min_periods=100).max()
    return F


def swing_signals(P, F, sp):
    """Entry signals at each day's CLOSE (traded next open). Returns dict of boolean DataFrames."""
    C, O, H = P["C"], P["O"], P["H"]
    base = ((C > F["sma200"]) & (F["sma50"] > F["sma200"]) & (F["adx"] >= sp.adx_min)
            & (C >= sp.min_price) & (F["dvol"] >= sp.min_dollar_vol))
    pull = (base & (F["rsi"] >= sp.rsi_pull[0]) & (F["rsi"] <= sp.rsi_pull[1])
            & (C > O) & (C > C.shift()))
    hi = H.rolling(sp.breakout_n).max().shift()
    bo = base & (C > hi) & (F["rvol"] >= sp.min_rvol_bo) & (F["rs63"] > 0) & (C < F["sma50"] * 1.3)
    lead = (base & (F["score"] >= 0.8) & (C >= F["ema21"] * 0.99) & (C <= F["ema21"] * 1.03)
            & (C > O) & (F["rsi"] > 40) & (F["rsi"] < 65))          # market leader dipping to its 21-day average
    use_p, use_b = sp.setup in ("pullback", "both", "all"), sp.setup in ("breakout", "both", "all")
    use_l = sp.setup in ("leader", "all")
    pull_e, bo_e, ld_e = (pull if use_p else pull & False), (bo if use_b else bo & False), (lead if use_l else lead & False)
    sig = pull_e | bo_e | ld_e
    for c in ETF_SET:
        if c in sig.columns:
            sig[c] = False
    return dict(sig=sig, bo=bo_e, pull=pull_e, lead=ld_e, base=base, hi=hi)


def regime_bool(P):
    C = P["C"]
    if "SPY" not in C:
        return pd.Series(True, index=C.index)
    spy = C["SPY"]
    return (spy > spy.rolling(200).mean()).fillna(False)


def first_valid(F):
    v = F["sma200"].notna().any(axis=1).to_numpy()
    return int(v.argmax()) + 1 if v.any() else None


def simulate_swing(P, F, sig, bo, reg, sp, i0, i1, eq0=25000.0, slip=0.0005, comm=0.0, ld=None):
    """Daily event loop. Signal at close t-1 -> buy at open t. Stops/targets use that day's range
    (stop assumed to hit first). Trailing stop ratchets at each close. Gaps fill at the open."""
    names, dates = list(P["C"].columns), P["C"].index
    O, H, L, C = (P[k].to_numpy(float) for k in ("O", "H", "L", "C"))
    atr = F["atr"].to_numpy(float)
    sc = np.nan_to_num(F["score"].to_numpy(float), nan=-1.0)
    sg, bo_ = sig.to_numpy(bool), bo.to_numpy(bool)
    ld_ = ld.to_numpy(bool) if ld is not None else np.zeros_like(sg)
    brk = (P["C"] < F["sma50"]).to_numpy(bool)
    rg = reg.reindex(dates).fillna(False).to_numpy(bool)
    cash, pos, trades, curve, inv, pend = eq0, {}, [], [], [], set()

    def close_pos(j, px, t, reason):
        nonlocal cash
        q = pos.pop(j)
        px = px * (1 - slip)
        cash += q["qty"] * px - comm * q["qty"]
        pnl = (px - q["entry"]) * q["qty"] - 2 * comm * q["qty"]
        trades.append(dict(symbol=names[j], kind=q["kind"], entry_date=dates[q["t0"]], exit_date=dates[t],
                           entry=q["entry"], exit=px, qty=q["qty"], pnl=pnl, R=pnl / q["risk"],
                           ret=pnl / q["eq"], hold=t - q["t0"], reason=reason))

    for t in range(max(i0, 1), i1):
        for j in list(pend):                                   # 1) exits decided at yesterday's close
            if j in pos:
                px = O[t, j]
                close_pos(j, pos[j]["last"] if np.isnan(px) else px, t, "signal/time")
        pend.clear()
        if len(pos) < sp.max_positions and (not sp.use_regime or rg[t - 1]):   # 2) new entries at today's open
            eq_now = cash + sum(q["qty"] * q["last"] for q in pos.values())
            cand = [j for j in np.flatnonzero(sg[t - 1])
                    if j not in pos and not np.isnan(O[t, j]) and atr[t - 1, j] > 0]
            cand.sort(key=lambda j: -sc[t - 1, j])             # best-ranked momentum first
            for j in cand:
                if len(pos) >= sp.max_positions:
                    break
                entry, rps = O[t, j] * (1 + slip), sp.stop_atr * atr[t - 1, j]
                if not rps > 0:
                    continue
                qty = int(min(eq_now * sp.risk_per_trade / rps, eq_now * sp.max_pos_pct / entry, cash / (entry + comm)))
                if qty < 1:
                    continue
                cash -= qty * entry + comm * qty
                pos[j] = dict(qty=qty, entry=entry, stop=entry - rps, hi=entry, days=0, t0=t, last=entry,
                              kind="breakout" if bo_[t - 1, j] else ("leader" if ld_[t - 1, j] else "pullback"), risk=rps * qty, eq=eq_now,
                              target=entry + sp.target_rr * rps if sp.target_rr > 0 else None)
        for j in list(pos):                                    # 3) stops / targets inside today's range
            q, o, h, l = pos[j], O[t, j], H[t, j], L[t, j]
            if np.isnan(o) or np.isnan(l):
                continue
            if o <= q["stop"]: close_pos(j, o, t, "stop")
            elif l <= q["stop"]: close_pos(j, q["stop"], t, "stop")
            elif q["target"] is not None and h >= q["target"]: close_pos(j, max(o, q["target"]), t, "target")
        for j in list(pos):                                    # 4) end of day: trail, time stop, trend exit
            q, c = pos[j], C[t, j]
            if np.isnan(c):
                continue
            q["last"], q["days"], q["hi"] = c, q["days"] + 1, max(q["hi"], c)
            if not np.isnan(atr[t, j]):
                q["stop"] = max(q["stop"], q["hi"] - sp.trail_atr * atr[t, j])
            if q["days"] >= sp.max_hold or (sp.trend_exit and q["kind"] == "breakout" and brk[t, j]):
                pend.add(j)
        eq = cash + sum(q["qty"] * q["last"] for q in pos.values())
        curve.append(eq)
        inv.append(1 - cash / eq if eq > 0 else 0.0)
    for j in list(pos):
        close_pos(j, pos[j]["last"], i1 - 1, "end")
    cv = pd.Series(curve, index=dates[max(i0, 1):i1], dtype=float)
    return pd.DataFrame(trades), cv, (float(np.mean(inv)) if inv else 0.0)


def run_swing(P, F, sp, eq0=25000.0, slip=0.0005, comm=0.0):
    S = swing_signals(P, F, sp)
    return simulate_swing(P, F, S["sig"], S["bo"], regime_bool(P), sp, first_valid(F), len(P["C"]), eq0, slip, comm, ld=S["lead"])


def walk_forward_swing(P, F, base, eq0=25000.0, slip=0.0005, comm=0.0, folds=3, min_trades=25):
    """Pick settings on PAST data, then trade the NEXT unseen chunk. Chunks are stitched into one
    out-of-sample result. This is the most honest number the tool produces."""
    reg, dates, n = regime_bool(P), P["C"].index, len(P["C"])
    s0 = first_valid(F)
    if s0 is None or n - s0 < 200:
        return pd.DataFrame(), None, pd.Series(dtype=float), pd.DataFrame(), "Not enough history (need ~2+ years)."
    sigs = {st: swing_signals(P, F, replace(base, setup=st)) for st in ("pullback", "breakout", "leader", "all")}
    edges = np.linspace(s0, n, folds + 2).astype(int)
    grid = list(itertools.product(("pullback", "breakout", "leader", "all"), (2.0, 3.0), (3.0, 4.5)))
    rows, oos_tr, oos_cv, eq = [], [], [], eq0
    for k in range(1, folds + 1):
        tr_end, te_end = edges[k], edges[k + 1]
        best = None
        for st, sa, ta in grid:
            sp = replace(base, setup=st, stop_atr=sa, trail_atr=ta)
            tr, cv, _ = simulate_swing(P, F, sigs[st]["sig"], sigs[st]["bo"], reg, sp, s0, tr_end, eq0, slip, comm, ld=sigs[st]["lead"])
            s = stats(tr, cv, eq0)
            if s["trades"] >= min_trades and (best is None or s["profit_factor"] > best[0]["profit_factor"]):
                best = (s, sp)
        if best is None:
            rows.append(dict(fold=k, note="too few trades in training window"))
            continue
        s, sp = best
        tr2, cv2, _ = simulate_swing(P, F, sigs[sp.setup]["sig"], sigs[sp.setup]["bo"], reg, sp, tr_end, te_end, eq, slip, comm, ld=sigs[sp.setup]["lead"])
        s2 = stats(tr2, cv2, eq)
        rows.append(dict(fold=k, trained=f"{dates[s0]:%b %Y}-{dates[tr_end - 1]:%b %Y}",
                         tested=f"{dates[tr_end]:%b %Y}-{dates[te_end - 1]:%b %Y}", setup=sp.setup,
                         stop_atr=sp.stop_atr, trail_atr=sp.trail_atr, train_PF=round(s["profit_factor"], 2),
                         test_trades=s2["trades"], test_PF=round(s2["profit_factor"], 2) if s2["trades"] else np.nan,
                         test_return=f"{s2['ret']:+.1%}"))
        if not tr2.empty:
            oos_tr.append(tr2)
        if not cv2.empty:
            oos_cv.append(cv2)
            eq = float(cv2.iloc[-1])
    if not oos_cv:
        return pd.DataFrame(rows), None, pd.Series(dtype=float), pd.DataFrame(), "No usable folds. Use more history or more symbols."
    tr_all = pd.concat(oos_tr) if oos_tr else pd.DataFrame()
    cv_all = pd.concat(oos_cv)
    s_all = stats(tr_all, cv_all, eq0)
    ok = s_all["trades"] >= 30 and s_all["profit_factor"] > 1.2 and s_all["ret"] > 0
    verdict = ("Some evidence of an edge on unseen data. Not proof: re-check with doubled slippage, then paper trade for weeks."
               if ok else "NO robust edge on unseen data. Do not trade this with real money. Try other rules or more history.")
    return pd.DataFrame(rows), s_all, cv_all, tr_all, verdict


def synthetic_daily(n_sym=150, years=4, seed=11):
    """Random daily data (market + stocks with drifting trends) in Alpaca's format. Tests code only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=int(years * 252))
    n = len(days)
    mkt = rng.normal(0.0003, 0.009, n) + 0.0006 * np.sin(np.arange(n) / 90)
    ts = pd.DatetimeIndex(days).tz_localize(NY).tz_convert("UTC")
    frames = []

    def make(sym, ret, p0):
        close = p0 * np.cumprod(1 + ret)
        open_ = np.r_[p0, close[:-1]] * (1 + rng.normal(0, 0.004, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
        vol = rng.uniform(2e5, 5e6) * np.exp(rng.normal(0, 0.35, n)) * (1 + 2 * (np.abs(ret) > 0.03))
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                          index=pd.MultiIndex.from_arrays([[sym] * n, ts], names=["symbol", "timestamp"]))
        frames.append(df)

    make("SPY", mkt + rng.normal(0, 0.001, n), 400.0)
    for i in range(n_sym):
        drift = np.zeros(n)
        for t in range(1, n):
            drift[t] = 0.97 * drift[t - 1] + rng.normal(0, 0.0004)
        ret = rng.uniform(0.6, 1.6) * mkt + drift + rng.normal(0, rng.uniform(0.012, 0.03), n)
        make(f"S{i:03d}", np.clip(ret, -0.4, 0.4), rng.uniform(15, 200))
    return pd.concat(frames)


# ---------- real-data plumbing (Alpaca) ----------
def universe_info(tc, include_etf=False):
    """Every exchange-listed US stock Alpaca lets you trade (OTC excluded), with shortable flags."""
    name = f"uni_info_{int(include_etf)}"
    cached = cache_get(name, 24)
    if cached is not None:
        return cached
    try:       # raw JSON: immune to new exchange names (e.g. TXSE) that the SDK's strict models reject
        assets = tc.get("/assets", {"status": "active", "asset_class": "us_equity"})
    except Exception:
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        assets = tc.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    f = lambda a, k: a.get(k) if isinstance(a, dict) else getattr(a, k, None)
    skip = NON_STOCK_WORDS if include_etf else NON_STOCK_WORDS + ETF_WORDS
    rows = []
    for a in assets:
        ex = f(a, "exchange")
        ex = str(getattr(ex, "value", ex))
        sym, nm = f(a, "symbol") or "", (f(a, "name") or "").lower()
        if (f(a, "tradable") and ex in ("NYSE", "NASDAQ", "AMEX", "ARCA") and sym.isalpha() and len(sym) <= 5
                and not any(w in nm for w in skip)):
            rows.append(dict(symbol=sym, shortable=bool(f(a, "shortable")), etb=bool(f(a, "easy_to_borrow"))))
    df = pd.DataFrame(rows).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    cache_set(name, df)
    return df


def get_universe(tc):
    return list(universe_info(tc, False).symbol)


def fetch_daily(dc, symbols, start, end, batch=100, progress=None):
    """Daily bars for many symbols. Uses full-market SIP data (free plan allows it when 15+ min old),
    falls back to IEX. Splits failed batches automatically."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment
    feed = DataFeed.SIP
    try:
        dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=["SPY"], timeframe=TimeFrame.Day,
                                           start=end - timedelta(days=7), end=end, feed=DataFeed.SIP))
    except Exception:
        feed = DataFeed.IEX
    frames = []

    def one(chunk):
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start,
                                                    end=end, feed=feed, adjustment=Adjustment.SPLIT)).df
            if not df.empty:
                frames.append(df)
        except Exception:
            if len(chunk) > 20:
                h = len(chunk) // 2
                one(chunk[:h]); one(chunk[h:])

    chunks = [symbols[i:i + batch] for i in range(0, len(symbols), batch)]
    for i, ch in enumerate(chunks):
        one(ch)
        if progress:
            progress((i + 1) / len(chunks))
        time.sleep(0.3)                                        # stay under the 200 calls/min free limit
    out = pd.concat(frames) if frames else pd.DataFrame()
    out.attrs["feed"] = "SIP (full market)" if feed == DataFeed.SIP else "IEX only"
    return out


def liquid_universe(dc, tc, sp, max_symbols, progress=None):
    """Stage 1: every tradable US stock -> keep the most liquid (price + dollar volume)."""
    uni = cache_get("universe", 24)
    if uni is None:
        uni = get_universe(tc)
        cache_set("universe", uni)
    s1 = cache_get("stage1", 12)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    if s1 is None:
        s1 = fetch_daily(dc, uni, end - timedelta(days=14), end, batch=200, progress=progress)
        cache_set("stage1", s1)
    d = s1.copy()
    d["dv"] = d["close"] * d["volume"]
    g = d.groupby(level=0).agg(close=("close", "last"), dv=("dv", "mean"), n=("close", "count"))
    g = g[(g.close >= sp.min_price) & (g.dv >= sp.min_dollar_vol) & (g.n >= 4)].sort_values("dv", ascending=False)
    return list(g.head(max_symbols).index), len(uni)


def load_swing_data(source, start=None, end=None, max_symbols=300, key=None, secret=None, progress=None):
    """Returns (P, F, note). source: 'synthetic' or 'alpaca'."""
    if source == "synthetic":
        P = bars_to_panels(synthetic_daily(), min_bars=0)
        return P, swing_features(P), "SYNTHETIC random data: tests the code only, says nothing about profitability."
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    key, secret = key or os.environ["ALPACA_KEY"], secret or os.environ["ALPACA_SECRET"]
    dc, tc = StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=True)
    syms, n_uni = liquid_universe(dc, tc, SwingParams(), max_symbols,
                                  (lambda f: progress(f * 0.4)) if progress else None)
    syms = list(dict.fromkeys(syms + INDEX_ETFS + list(SECTORS)))
    st = pd.Timestamp(start, tz="UTC") - timedelta(days=420)           # warm-up for 200-day averages
    en = min(pd.Timestamp(end, tz="UTC") + timedelta(days=1), pd.Timestamp.now(tz="UTC") - timedelta(minutes=20))
    df = fetch_daily(dc, syms, st.to_pydatetime(), en.to_pydatetime(), batch=100,
                     progress=(lambda f: progress(0.4 + f * 0.6)) if progress else None)
    if df.empty:
        raise RuntimeError("No daily data returned. Check keys / dates.")
    P = bars_to_panels(df)
    P = {k: v[v.index >= pd.Timestamp(start) - pd.Timedelta(days=420)] for k, v in P.items()}
    return P, swing_features(P), f"{len(P['C'].columns)} stocks from {n_uni} US listings, {df.attrs.get('feed')}. {BIAS_NOTE}"


# ---------- live swing scan, market overview, earnings, news ----------
def trend_label(c, s50, s200, adx):
    if c > s50 > s200:
        lab = "Strong uptrend" if adx >= 25 else "Uptrend"
    elif c > s200:
        lab = "Pullback in uptrend"
    elif c < s200 and s50 < s200:
        lab = "Downtrend"
    else:
        lab = "Mixed / transition"
    return lab, ("trending" if adx >= 25 else "weak trend" if adx >= 20 else "range-bound / choppy")


def overview(P, F):
    C = P["C"]
    idx_rows, sec_rows = [], []
    for s in INDEX_ETFS:
        if s in C and C[s].count() > 200:
            c, s50, s200, adx = (float(x[s].iloc[-1]) for x in (C, F["sma50"], F["sma200"], F["adx"]))
            lab, strength = trend_label(c, s50, s200, adx)
            idx_rows.append(dict(symbol=s, price=round(c, 2), vs_200d=f"{c / s200 - 1:+.1%}", vs_50d=f"{c / s50 - 1:+.1%}",
                                 adx=round(adx), rsi=round(float(F["rsi"][s].iloc[-1])), trend=lab, strength=strength))
    spy63 = float(F["ret63"]["SPY"].iloc[-1]) if "SPY" in C else 0.0
    for s, name in SECTORS.items():
        if s in C and C[s].count() > 70:
            r63 = float(F["ret63"][s].iloc[-1])
            sec_rows.append(dict(sector=name, etf=s, month=f"{C[s].iloc[-1] / C[s].iloc[-22] - 1:+.1%}",
                                 quarter=f"{r63:+.1%}", vs_SPY=r63 - spy63,
                                 above_50d="yes" if C[s].iloc[-1] > F["sma50"][s].iloc[-1] else "no"))
    sectors = pd.DataFrame(sec_rows).sort_values("vs_SPY", ascending=False) if sec_rows else pd.DataFrame()
    if not sectors.empty:
        sectors["vs_SPY"] = sectors["vs_SPY"].map(lambda x: f"{x:+.1%}")
    breadth = None
    stocks = [c for c in C.columns if c not in ETF_SET]
    if len(stocks) > 60:
        last = C[stocks].iloc[-1]
        breadth = dict(above_50d=float((last > F["sma50"][stocks].iloc[-1]).mean()),
                       above_200d=float((last > F["sma200"][stocks].iloc[-1]).mean()),
                       near_high=float((last >= 0.98 * F["hi52"][stocks].iloc[-1]).mean()), n=len(stocks))
    level, text = "warning", "No SPY data."
    if idx_rows:
        spy = idx_rows[0]
        c, s50, s200 = (float(x["SPY"].iloc[-1]) for x in (C, F["sma50"], F["sma200"]))
        if c > s50 > s200:
            level = "success"
        elif c > s200:
            level = "warning"
        else:
            level = "error"
        ok = c > s200
        text = (f"SPY: {spy['trend']} ({spy['strength']}). Swing longs {'ENABLED' if ok else 'PAUSED (below 200-day average)'}. "
                f"{'Favor pullbacks and breakouts.' if level == 'success' else 'Be selective, smaller size.' if level == 'warning' else 'Cash is a position.'}")
    return dict(indices=pd.DataFrame(idx_rows), sectors=sectors, breadth=breadth, level=level, text=text,
                asof=str(C.index[-1].date()))


def earnings_calendar(token, days=10):
    if not token:
        return pd.DataFrame()
    today = datetime.now(NY).date()
    url = "https://finnhub.io/api/v1/calendar/earnings?" + urllib.parse.urlencode(
        {"from": str(today), "to": str(today + timedelta(days=days)), "token": token})
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            df = pd.DataFrame(json.load(r).get("earningsCalendar", []))
        cols = [c for c in ["symbol", "date", "hour", "epsEstimate", "revenueEstimate"] if c in df.columns]
        return df[cols].sort_values("date") if not df.empty else df
    except Exception:
        return pd.DataFrame()


def fetch_news(key, secret, symbols, limit=8):
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest
        ns = NewsClient(key, secret).get_news(NewsRequest(symbols=",".join(symbols), limit=limit))
        items = ns.data.get("news", []) if hasattr(ns, "data") else []
        return [(getattr(n, "created_at", None), getattr(n, "headline", ""), getattr(n, "source", "")) for n in items]
    except Exception:
        return []


def swing_candidates(P, F, sp, equity, earn=None):
    """Latest daily bar -> buy candidates, strongest trending stocks, near-breakout watchlist.
    Every row carries entry / target / stop / shares so it is directly actionable."""
    C = P["C"]
    S = swing_signals(P, F, sp)
    earn = earn or {}
    today = datetime.now(NY).date()
    mkt_ok = (not sp.use_regime) or bool(regime_bool(P).iloc[-1])
    rr = sp.target_rr if sp.target_rr > 0 else 3.0
    last = lambda df, s: float(df[s].iloc[-1])
    sg = S["sig"].iloc[-1]

    def build(s, state, setup, entry, why):
        price, a = last(C, s), last(F["atr"], s)
        stop = entry - sp.stop_atr * a
        rps = entry - stop
        qty = int(min(equity * sp.risk_per_trade / rps, equity * sp.max_pos_pct / entry)) if rps > 0 else 0
        sc = last(F["score"], s)
        e = earn.get(s)
        if state in ("BUY SIGNAL", "Near breakout"):
            if not mkt_ok:
                state, qty = "WAIT: market below 200-day", 0
            elif e and sp.earnings_buffer > 0:
                dd = (pd.Timestamp(e).date() - today).days
                if 0 <= dd <= sp.earnings_buffer:
                    state, qty = f"SKIP: earnings in {dd}d", 0
        return dict(symbol=s, state=state, setup=setup, side="LONG", price=round(price, 2), entry=round(entry, 2),
                    target=round(entry + rr * rps, 2), stop=round(stop, 2), rr=rr, qty=qty,
                    risk_usd=round(qty * rps), reward_usd=round(qty * rr * rps),
                    score=int(round(sc * 100)) if sc == sc else 0,
                    ret63=f"{last(F['ret63'], s):+.0%}", vs_SPY=f"{last(F['rs63'], s):+.0%}",
                    adx=round(last(F["adx"], s)), rsi=round(last(F["rsi"], s)), rvol=round(last(F["rvol"], s), 1),
                    from_52w_high=f"{price / last(F['hi52'], s) - 1:+.0%}", earnings=e or "", why=why,
                    spark=[round(float(v), 4) for v in C[s].dropna().tail(60)])

    cands = []
    for s in sg.index[sg]:
        price = last(C, s)
        is_bo = bool(S["bo"][s].iloc[-1])
        is_ld = (not is_bo) and bool(S["lead"][s].iloc[-1]) and not bool(S["pull"][s].iloc[-1])
        rsi, adx, rvol = round(last(F["rsi"], s)), round(last(F["adx"], s)), round(last(F["rvol"], s), 1)
        why = (f"Broke its {sp.breakout_n}-day high on {rvol}x volume, strong trend (ADX {adx}), beating the market" if is_bo else
               "Top-20% momentum leader dipping to its 21-day average and bouncing" if is_ld else
               f"Uptrend, dipped to RSI {rsi} and bounced today: buying the dip in a strong stock")
        cands.append(build(s, "BUY SIGNAL", "Breakout" if is_bo else "Leader dip" if is_ld else "Pullback", price, why))
    cands = pd.DataFrame(cands)
    if not cands.empty:
        cands = cands.sort_values("score", ascending=False)

    base_last = S["base"].iloc[-1]
    lead = []
    for s in [x for x in base_last.index[base_last] if x not in ETF_SET]:
        price, hi = last(C, s), last(S["hi"], s)
        near = hi == hi and 0.97 * hi <= price <= hi
        ext = price > 1.3 * last(F["sma50"], s)
        if sg.get(s, False):
            continue
        if near:
            lead.append(build(s, "Near breakout", "Breakout", hi,
                              f"Within 3% of its {sp.breakout_n}-day high. Buy only if it trades above {hi:.2f} on strong volume"))
        else:
            lead.append(build(s, "Extended (wait for pullback)" if ext else "Strong trend", "Trend", price,
                              "Strong uptrend; too far above its 50-day average to chase" if ext else "Strong uptrend, no entry trigger yet"))
    lead = pd.DataFrame(lead)
    if not lead.empty:
        lead = lead.sort_values("score", ascending=False)
    watch = lead[lead.state == "Near breakout"].head(25) if not lead.empty else lead
    allrows = pd.concat([cands, lead]) if not cands.empty else lead
    return cands, (allrows.head(40) if not allrows.empty else allrows), watch


def swing_scan(dc, tc, sp, equity, max_symbols=1000, fh=None, progress=None):
    syms, n_uni = liquid_universe(dc, tc, sp, max_symbols, (lambda f: progress(f * 0.4)) if progress else None)
    syms = list(dict.fromkeys(syms + INDEX_ETFS + list(SECTORS)))
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    ck = f"stage2_{max_symbols}_{sp.min_price}_{int(sp.min_dollar_vol)}"
    df = cache_get(ck, 3)
    if df is None:
        df = fetch_daily(dc, syms, end - timedelta(days=520), end, batch=100,
                         progress=(lambda f: progress(0.4 + f * 0.6)) if progress else None)
        cache_set(ck, df)
    P = bars_to_panels(df)
    now = datetime.now(NY)
    if P["C"].index[-1].date() == now.date() and (now.hour, now.minute) < (16, 20):
        P = {k: v.iloc[:-1] for k, v in P.items()}             # drop today's unfinished bar
    F = swing_features(P)
    earn_df = earnings_calendar(fh, days=max(10, sp.earnings_buffer + 3))
    earn = {}
    if not earn_df.empty:
        for s, d in zip(earn_df.symbol, earn_df.date):
            earn.setdefault(s, d)
    cands, leaders, watch = swing_candidates(P, F, sp, equity, earn)
    return dict(cands=cands, leaders=leaders, watch=watch, ov=overview(P, F), earn=earn_df,
                asof=str(P["C"].index[-1].date()), n=len(P["C"].columns), n_uni=n_uni,
                feed=df.attrs.get("feed", "?"))


def market_light(dc):
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    df = fetch_daily(dc, INDEX_ETFS + list(SECTORS), end - timedelta(days=520), end, batch=30)
    P = bars_to_panels(df, min_bars=0)
    return overview(P, swing_features(P))


def swing_positions(tc, dc, sp):
    pos = tc.get_all_positions()
    if not pos:
        return pd.DataFrame()
    syms = [p.symbol for p in pos]
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    df = fetch_daily(dc, syms, end - timedelta(days=240), end, batch=50)
    P = bars_to_panels(df, min_bars=0)
    F = swing_features(P)
    rows = []
    for p in pos:
        s = p.symbol
        row = dict(symbol=s, side=str(getattr(p.side, "value", p.side)), qty=p.qty, avg_entry=round(float(p.avg_entry_price), 2),
                   price=round(float(p.current_price), 2), pl=f"{float(p.unrealized_plpc):+.1%}")
        if s in P["C"]:
            c = P["C"][s].dropna()
            a = float(F["atr"][s].iloc[-1])
            trail = float(c.tail(22).max()) - sp.trail_atr * a
            row.update(suggested_trail_stop=round(trail, 2),
                       signal="EXIT: below trail stop" if float(p.current_price) < trail else
                              "EXIT?: closed below 50-day" if c.iloc[-1] < F["sma50"][s].iloc[-1] else "hold")
        rows.append(row)
    return pd.DataFrame(rows)


def place_swing_order(tc, sym, qty, stop, target=None):
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    kw = dict(symbol=sym, qty=int(qty), side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
              stop_loss=StopLossRequest(stop_price=rnd(stop)))
    if target:
        tc.submit_order(MarketOrderRequest(order_class=OrderClass.BRACKET,
                                           take_profit=TakeProfitRequest(limit_price=rnd(target)), **kw))
    else:
        tc.submit_order(MarketOrderRequest(order_class=OrderClass.OTO, **kw))


# =====================================================================
# 6. LIVE DAY-TRADING: whole-market "hot list" -> 1-minute signals -> orders
# =====================================================================
def get_keys():
    return os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")


def live_clients(key, secret, paper=True):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    return StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=paper)


def shortable_set(tc, include_etf=False):
    u = universe_info(tc, include_etf)
    return set(u[u.shortable & u.etb].symbol)


def dynamic_watchlist(key, secret, base, top=40, min_price=1.0):
    """Fallback if the whole-market snapshot fails: base list + most actives + movers."""
    try:
        from alpaca.data.historical.screener import ScreenerClient
        from alpaca.data.requests import MostActivesRequest, MarketMoversRequest
        sc = ScreenerClient(key, secret)
        act = sc.get_most_actives(MostActivesRequest(top=top))
        mv = sc.get_market_movers(MarketMoversRequest(top=top))
        syms = [a.symbol for a in act.most_actives]
        syms += [m.symbol for m in list(mv.gainers) + list(mv.losers) if float(getattr(m, "price", min_price)) >= min_price]
        syms = [s for s in syms if s.isalpha() and len(s) <= 5]
        return list(dict.fromkeys(list(base) + syms))[:150], f"{len(set(syms))} extra symbols (most active + movers)"
    except Exception as e:
        return list(base), f"Dynamic list unavailable ({str(e)[:80]}); using your watchlist."


def hot_list(dc, tc, p, progress=None):
    """Scan the WHOLE US market (every exchange-listed stock incl. low-priced ones) with snapshots and rank by
    how hot each is today: size of move, volume vs yesterday, dollars traded. Returns (DataFrame, diagnostics)."""
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    uni = universe_info(tc, p.include_etf)
    syms = list(uni.symbol)
    today, rows = datetime.now(NY).date(), []

    def one(ch):
        try:
            snaps = dc.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=ch, feed=DataFeed.IEX))
        except Exception:
            if len(ch) > 25:
                h = len(ch) // 2
                one(ch[:h]); one(ch[h:])
            return
        for s, sn in snaps.items():
            try:
                lt, db, pdb = sn.latest_trade, sn.daily_bar, sn.previous_daily_bar
                if not (lt and db and pdb):
                    continue
                ts = db.timestamp if db.timestamp.tzinfo else db.timestamp.replace(tzinfo=timezone.utc)
                if ts.astimezone(NY).date() != today:              # no trades today
                    continue
                price, pc = float(lt.price), float(pdb.close)
                if pc <= 0 or price <= 0:
                    continue
                rows.append(dict(symbol=s, price=price, prev_close=pc, chg=price / pc - 1,
                                 gap=float(db.open) / pc - 1, vol=float(db.volume), prev_vol=float(pdb.volume),
                                 dvol=price * float(db.volume)))
            except Exception:
                continue

    chunks = [syms[i:i + 250] for i in range(0, len(syms), 250)]
    for i, ch in enumerate(chunks):
        one(ch)
        if progress:
            progress((i + 1) / len(chunks))
        time.sleep(0.25)                                         # stay under the 200 calls/min free limit
    diag = dict(universe=len(syms), snapped=len(rows), eligible=0)
    if not rows:
        return pd.DataFrame(), diag
    df = pd.DataFrame(rows).drop_duplicates("symbol")
    df = df[(df.price >= p.min_price) & (df.price <= p.max_price) & (df.dvol >= 15000)].copy()
    diag["eligible"] = len(df)
    if df.empty:
        return df, diag
    df["vol_ratio"] = df.vol / df.prev_vol.clip(lower=1)
    df["hot"] = (0.45 * df.chg.abs().rank(pct=True) + 0.30 * df.vol_ratio.rank(pct=True)
                 + 0.25 * df.dvol.rank(pct=True))
    return df.sort_values("hot", ascending=False).reset_index(drop=True), diag


def fetch_today(dc, symbols):
    """Today's 1-minute bars (free IEX feed). Missing minutes are filled so indicators mean real time."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    now = datetime.now(NY)
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    out = {}

    def one(ch):
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=ch, timeframe=TimeFrame.Minute,
                                                    start=start, feed=DataFeed.IEX)).df
        except Exception:
            if len(ch) > 10:
                h = len(ch) // 2
                one(ch[:h]); one(ch[h:])
            return
        if df.empty:
            return
        have = set(df.index.get_level_values(0))
        for s in ch:
            if s not in have:
                continue
            g = df.loc[s][["open", "high", "low", "close", "volume"]].copy()
            g.index = g.index.tz_convert(NY)
            g = g[~g.index.duplicated()]
            idx = pd.date_range(g.index[0].floor("min"), max(now.replace(second=0, microsecond=0), g.index[-1]), freq="min")
            g = g.reindex(idx)
            g["close"] = g["close"].ffill()
            for col in ("open", "high", "low"):
                g[col] = g[col].fillna(g["close"])
            g["volume"] = g["volume"].fillna(0)
            g = g.dropna(subset=["close"])
            if len(g) >= 15:
                out[s] = g

    for i in range(0, len(symbols), 100):
        one(symbols[i:i + 100])
    return out


def day_scan(dc, p, equity, hot=None, shortable=None, avoid=None):
    """Run the 5 day-trade setups over the hot list + your watchlist.
    Returns dict(signals, watch, mkt, diag). A signal stays actionable for p.fresh_bars minutes while price is
    still near its entry. 'watch' = closest-to-firing names with the exact trigger price."""
    now = datetime.now(NY)
    syms = list(hot.sort_values("hot", ascending=False).symbol.head(p.hot_top)) if hot is not None and not hot.empty else []
    syms = list(dict.fromkeys(syms + list(p.watchlist)))
    bars = fetch_today(dc, list(dict.fromkeys(syms + ["SPY"])))
    mkt = 0
    if "SPY" in bars:
        d0 = prep_day(bars["SPY"])
        mkt = int(np.sign(d0["close"][-1] - d0["vwap"][-1]))
    pcmap = dict(zip(hot.symbol, hot.prev_close)) if hot is not None and not hot.empty else {}
    sigs, watch = [], []
    diag = dict(with_data=0, above_vwap=0, volume_ok=0, near_high=0, signals=0)
    last_min = now.hour * 60 + now.minute
    for s in syms:
        g = bars.get(s)
        if g is None:
            continue
        d = prep_day(g, pcmap.get(s))
        n = len(d["mins"])
        d["mkt"] = np.full(n, float(mkt))
        sg, kd = signals(d, p, True)
        price = float(d["close"][-1])
        rv = float(d["rvol"][-1]) if d["rvol"][-1] == d["rvol"][-1] else 0.0
        chg = float(d["chg"][-1])
        rsi = float(d["rsi"][-1]) if d["rsi"][-1] == d["rsi"][-1] else 50.0
        hi20, lo20, vw = float(d["hi20"][-1]), float(d["lo20"][-1]), float(d["vwap"][-1])
        diag["with_data"] += 1
        diag["above_vwap"] += int(price > vw)
        diag["volume_ok"] += int(rv >= p.min_rvol)
        diag["near_high"] += int(hi20 == hi20 and price >= 0.992 * hi20)
        spark = [round(float(x), 4) for x in d["close"][-60:]]

        def row(setup, side, entry, why, age=0, state="SIGNAL"):
            stop, target = levels(entry, float(d["atr"][-1]) if d["atr"][-1] == d["atr"][-1] else 0.01 * entry, side, p)
            qty = size(equity, entry, stop, p)
            score = int(min(99, 40 + 22 * max(rv - 1, 0) + 300 * min(abs(chg), 0.2) + (8 if setup in ("Gap & go", "Opening-range breakout") else 0)))
            return dict(symbol=s, state=state, setup=setup, side="LONG" if side == 1 else "SHORT",
                        signal="LONG" if side == 1 else "SHORT", price=rnd(price), entry=rnd(entry), target=rnd(target),
                        stop=rnd(stop), rr=p.rr, qty=qty, risk_usd=round(qty * abs(entry - stop)),
                        reward_usd=round(qty * abs(target - entry)), chg=f"{chg:+.1%}", rvol=round(rv, 1), rsi=round(rsi),
                        score=score, age_min=age, earnings="", why=why, spark=spark)

        lo = max(0, n - p.fresh_bars)
        idx = np.flatnonzero(sg[lo:])
        made = False
        if len(idx):
            i = lo + int(idx[-1])
            side = int(sg[i])
            entry = float(d["close"][i])
            stop, target = levels(entry, float(d["atr"][i]), side, p)
            ok = (price <= entry * 1.006 and price > stop) if side == 1 else (price >= entry * 0.994 and price < stop)
            if side == -1 and ((shortable is not None and s not in shortable) or price < 5):
                ok = False
            if avoid and s in avoid:
                ok = False
            if ok:
                nm = KIND_NAMES[int(kd[i])]
                dirn = "above" if side == 1 else "below"
                sigs.append(row(nm, side, entry, f"{nm}: price {dirn} VWAP, {chg:+.1%} today, volume {rv:.1f}x normal, RSI {rsi:.0f}", n - 1 - i))
                made = True
        if made or last_min >= p.last_entry[0] * 60 + p.last_entry[1]:
            continue
        # not firing yet -> maybe close to firing
        if p.allow_long and hi20 == hi20 and price > vw and 0.992 * hi20 <= price < hi20 and d["act"][-1] >= p.min_active \
                and rv >= 0.6 * p.min_rvol and price >= p.min_price:
            watch.append(row("Near breakout", 1, hi20 * 1.0005,
                             f"Above VWAP and within {(1 - price / hi20) * 100:.1f}% of its 20-minute high. Buy only if it breaks {rnd(hi20)}",
                             state="WATCH"))
        elif p.allow_short and lo20 == lo20 and price < vw and lo20 < price <= 1.008 * lo20 and price >= 5 \
                and (shortable is None or s in shortable) and d["act"][-1] >= p.min_active and rv >= 0.6 * p.min_rvol:
            watch.append(row("Near breakdown", -1, lo20 * 0.9995,
                             f"Below VWAP and within {(price / lo20 - 1) * 100:.1f}% of its 20-minute low. Short only if it breaks {rnd(lo20)}",
                             state="WATCH"))
    diag["signals"] = len(sigs)
    sg_df = pd.DataFrame(sigs).sort_values("score", ascending=False) if sigs else pd.DataFrame()
    wt_df = pd.DataFrame(watch).sort_values("score", ascending=False) if watch else pd.DataFrame()
    return dict(signals=sg_df, watch=wt_df, mkt=mkt, diag=diag, n=len(bars), asof=now)


def place_order(tc, sym, signal, qty, stop, target):
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    tc.submit_order(MarketOrderRequest(
        symbol=sym, qty=int(qty), side=OrderSide.BUY if signal == "LONG" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=rnd(target)),
        stop_loss=StopLossRequest(stop_price=rnd(stop))))


def live_loop(p, paper=True, poll=60):
    key, secret = get_keys()
    if not (key and secret):
        sys.exit("Set ALPACA_KEY and ALPACA_SECRET first.")
    dc, tc = live_clients(key, secret, paper)
    start_eq, cool, hot, hot_t, short_ok = float(tc.get_account().last_equity), {}, None, 0, None
    print(f"Running ({'PAPER' if paper else '*** REAL MONEY ***'}). Ctrl+C to stop.")
    while True:
        now = datetime.now(NY)
        hm = (now.hour, now.minute)
        if now.weekday() >= 5:
            print("Weekend: market closed."); break
        eq = float(tc.get_account().equity)
        if hm >= p.flatten_at or eq <= start_eq * (1 - p.daily_loss_limit):
            tc.close_all_positions(cancel_orders=True)
            print("Flattened (end of day or daily loss limit hit). Done."); break
        if hm < p.start_after:
            time.sleep(poll); continue
        if hot is None or time.time() - hot_t > 180:
            hot, hd = hot_list(dc, tc, p)
            hot_t, short_ok = time.time(), shortable_set(tc, p.include_etf)
            print(f"{now:%H:%M:%S} hot list: {hd}")
        held = {x.symbol for x in tc.get_all_positions()}
        if len(held) < p.max_open:
            res = day_scan(dc, p, eq, hot, short_ok)
            for _, r in res["signals"].iterrows() if not res["signals"].empty else []:
                if r.symbol in held or cool.get(r.symbol, datetime.min) > datetime.now() or r.qty < 1:
                    continue
                try:
                    place_order(tc, r.symbol, r.signal, r.qty, r.stop, r.target)
                    cool[r.symbol] = datetime.now() + timedelta(minutes=p.cooldown_min)
                    held.add(r.symbol)
                    print(f"{now:%H:%M:%S} {r.signal} {int(r.qty)} {r.symbol} [{r.setup}] @~{r.entry} stop {r.stop} target {r.target}")
                except Exception as e:
                    print("Order failed:", e)
                if len(held) >= p.max_open:
                    break
        time.sleep(poll)


# =====================================================================
# 7. DAY-TRADING OPTIMIZER (train/test split)
# =====================================================================
def sweep(prep, base, eq0=25000.0, slip=0.0005, comm=0.0, train_frac=0.7, min_trades=30, top=5):
    days = sorted(prep)
    k = int(len(days) * train_frac)
    train, test = {d: prep[d] for d in days[:k]}, {d: prep[d] for d in days[k:]}
    grid = dict(rr=[1.5, 2.0, 3.0], atr_mult=[1.0, 1.5, 2.0], min_rvol=[1.5, 2.0, 3.0])
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(grid, combo))
        tr, cv = run(train, replace(base, **kw), eq0, slip, comm)
        s = stats(tr, cv, eq0)
        if s["trades"] >= min_trades:
            rows.append((kw, s))
    if not rows:
        return pd.DataFrame(), "No setting produced enough trades in the training period. Use more data."
    rows.sort(key=lambda r: r[1]["profit_factor"], reverse=True)
    out = []
    for kw, s in rows[:top]:
        tr2, cv2 = run(test, replace(base, **kw), eq0, slip, comm)
        s2 = stats(tr2, cv2, eq0)
        out.append({**kw, "train_trades": s["trades"], "train_PF": round(s["profit_factor"], 2),
                    "test_trades": s2["trades"], "test_PF": round(s2["profit_factor"], 2) if s2["trades"] else np.nan,
                    "test_return": f"{s2['ret']:+.1%}"})
    df = pd.DataFrame(out)
    good = (df.test_PF > 1.2) & (df.test_trades >= 30)
    return df, ("Some settings held up on unseen data; still re-check with higher slippage and more history."
                if good.any() else "NO robust edge: the best in-sample settings did not hold up on unseen data. Do not trade this live.")


# =====================================================================
# 8. DASHBOARD  (python day_trading_lab.py dashboard)
# =====================================================================
GUIDE = """
### How to read a pick card
- **BUY AT**: the entry price (a small zone just above it is still fine). Shorts say **SHORT AT**.
- **TAKE PROFIT**: where to sell for the gain (2x to 3x what you risk by default).
- **STOP LOSS**: where the idea is wrong. If price gets there you are out. Always use it.
- **Shares / Risk / Goal**: size so a stop-out costs about 1% of the account, and the gain if the target hits.
- The colored tag is the **live status** (updates every second). *IN BUY ZONE* = green light. *RAN PAST ENTRY* = don't chase.
- The little chart shows the recent price with your entry (blue), target (green) and stop (red) lines.

### How day trading finds trades now
1. **Whole-market scan.** Every exchange-listed US stock (including low-priced ones) is ranked by how hot it is today:
   size of the move, volume vs yesterday, dollars traded. Refreshed every 3 minutes.
2. **The hottest ~150 get a 1-minute look** with five setups: *Momentum breakout*, *VWAP bounce*, *Opening-range breakout*,
   *Gap & go*, *Day runner*. Shorts are the mirror images (only stocks Alpaca says you can short, and not penny stocks).
3. **Signals stay valid for several minutes** while price is still near the entry, so you don't miss them by a few seconds.
4. **No signal yet?** You get a *closest to firing* list with the exact price to wait for.
5. **Stops are never tighter than 0.6%** (1.2% under $5) so normal noise doesn't stop you out instantly.
6. The **signals slider** controls how picky it is. More signals = more trades of lower average quality.

### Swing trading
Scans thousands of stocks on daily bars for pullbacks in uptrends, breakouts, and leaders dipping to their 21-day average.
Needs the market (SPY) above its 200-day average. Earnings within 5 days are skipped if you add a Finnhub key.

### Honest limits
- The free Alpaca feed (IEX) sees only a few percent of real volume, so thin and penny stocks give fewer, noisier signals.
  The paid full-market feed fixes that.
- Penny stocks gap, halt and have wide spreads. Position sizes are cut for them. Treat them as high risk.
- More signals does not mean more profit. Use **Test It** to see how each setting actually performed.
- Paper trade for weeks first. This is not financial advice.
"""

CSS = """<style>
.block-container{max-width:760px;padding-top:.8rem;padding-bottom:5rem}
.stButton>button{width:100%;border-radius:14px;font-weight:700;padding:.6rem 1rem}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,#6366f1,#06b6d4);border:0;color:#fff;box-shadow:0 6px 16px rgba(99,102,241,.35)}
.stTabs [data-baseweb="tab-list"]{gap:6px;flex-wrap:wrap}
.stTabs [data-baseweb="tab"]{border-radius:999px;padding:6px 14px;background:rgba(128,128,128,.13);font-weight:700;height:auto}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1,#06b6d4);color:#fff}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
[data-testid="stMetric"]{background:rgba(128,128,128,.08);border:1px solid rgba(128,128,128,.25);border-radius:16px;padding:10px 14px}
.tl-hero{display:flex;align-items:center;border-radius:22px;padding:16px 18px;margin:0 0 12px 0;color:#fff;background:linear-gradient(135deg,#4f46e5 0%,#0891b2 58%,#10b981 100%);box-shadow:0 10px 30px rgba(79,70,229,.35)}
.tl-logo{font-size:2rem;margin-right:14px;background:rgba(255,255,255,.2);border-radius:16px;width:54px;height:54px;line-height:54px;text-align:center}
.tl-h1{font-size:1.55rem;font-weight:800}
.tl-sub{font-size:.82rem;opacity:.92}
.tl-bar{margin:6px 0 8px 0}
.tl-bar .tl-chip{margin:0 6px 6px 0}
.tl-chip{display:inline-block;font-size:.76rem;font-weight:700;padding:3px 10px;border-radius:999px;color:#fff;white-space:nowrap}
.c-green{background:#16a34a}.c-amber{background:#d97706}.c-red{background:#dc2626}.c-gray{background:#6b7280}.c-blue{background:#2563eb}.c-purple{background:#7c3aed}
.tl-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#fff;margin-right:6px;animation:tlp 1.1s infinite}
@keyframes tlp{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
.tl-pill{display:inline-block;margin:0 6px 6px 0;padding:5px 11px;border-radius:12px;background:rgba(128,128,128,.12);border:1px solid rgba(128,128,128,.28);font-size:.8rem;font-weight:700}
.hot-up{background:rgba(22,163,74,.15);border-color:rgba(22,163,74,.55)}
.hot-dn{background:rgba(220,38,38,.15);border-color:rgba(220,38,38,.55)}
.tl-card{border:1px solid rgba(128,128,128,.28);border-left:7px solid #6366f1;border-radius:20px;padding:14px 16px;margin:14px 0 6px 0;background:rgba(128,128,128,.06);box-shadow:0 6px 18px rgba(0,0,0,.14)}
.tl-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap}
.tl-sym{font-size:1.75rem;font-weight:800;letter-spacing:.5px;margin-right:6px}
.tl-rank{font-size:.78rem;font-weight:700;opacity:.75;text-align:right}
.tl-meter{height:6px;border-radius:99px;background:rgba(128,128,128,.28);overflow:hidden;margin-top:4px;width:110px}
.tl-meter div{height:100%;background:linear-gradient(90deg,#f59e0b,#16a34a)}
.tl-live{margin:6px 0 2px 0;font-size:1.05rem;font-weight:700}
.tl-live small{font-weight:500;opacity:.7}
.tl-spark{width:100%;height:auto;display:block;margin:4px 0}
.tl-grid{table-layout:fixed;border-collapse:separate;border-spacing:8px 0;margin:8px -8px;width:calc(100% + 16px)}
.tl-grid td{padding:0;border:0}
.tl-box{border-radius:14px;padding:9px 4px;text-align:center}
.tl-box small{display:block;font-size:.62rem;font-weight:800;letter-spacing:.07em;opacity:.85}
.tl-box b{font-size:1.28rem}
.b-entry{background:rgba(37,99,235,.16);border:1px solid rgba(37,99,235,.55)}
.b-target{background:rgba(22,163,74,.16);border:1px solid rgba(22,163,74,.55)}
.b-stop{background:rgba(220,38,38,.16);border:1px solid rgba(220,38,38,.55)}
.tl-meta{font-size:.84rem;opacity:.92}
.tl-meta span{display:inline-block;margin:2px 14px 2px 0}
.tl-why{font-size:.8rem;opacity:.72;margin-top:6px}
.tl-banner{border-radius:14px;padding:10px 14px;font-weight:700;margin:6px 0 10px 0}
.k-green{background:rgba(22,163,74,.16);border:1px solid rgba(22,163,74,.55)}
.k-amber{background:rgba(217,119,6,.16);border:1px solid rgba(217,119,6,.55)}
.k-red{background:rgba(220,38,38,.16);border:1px solid rgba(220,38,38,.55)}
</style>"""

HERO = ("<div class='tl-hero'><div class='tl-logo'>📈</div><div><div class='tl-h1'>Trading Lab</div>"
        "<div class='tl-sub'>Scans the whole market · clear entries, targets and stops · paper trading</div></div></div>")

PRESETS = {
    "Careful": dict(note="Smaller bets, wider stops, fewer positions.", risk=0.5, stop=2.5, trail=4.0, rr=3.0, maxpos=5, adx=25,
                    d_risk=0.5, d_rr=2.0, d_open=2, d_loss=2.0, lowpx=5.0, s_minp=5.0, s_dv=10),
    "Balanced": dict(note="About 1% risked per trade.", risk=1.0, stop=2.0, trail=3.5, rr=3.0, maxpos=8, adx=20,
                     d_risk=1.0, d_rr=2.0, d_open=3, d_loss=3.0, lowpx=2.0, s_minp=3.0, s_dv=5),
    "Aggressive": dict(note="More trades, lower price floor ($1 stocks), shorts on, bigger swings in your account.", risk=1.5,
                       stop=1.5, trail=3.0, rr=2.5, maxpos=10, adx=18, d_risk=1.5, d_rr=2.0, d_open=5, d_loss=4.0,
                       lowpx=1.0, s_minp=2.0, s_dv=3),
}
SENS = ["Fewer, stronger", "Balanced", "More signals", "Maximum"]
SENS_NOTE = {"Fewer, stronger": "Picky: higher volume bar, market must agree. Few trades, higher quality.",
             "Balanced": "Middle ground.",
             "More signals": "Looser filters, signals stay valid longer, trades alongside or against the market with extra volume.",
             "Maximum": "Everything on, market direction ignored. Most signals, lowest average quality. Use small size."}


def apply_sensitivity(p, sp, level):
    i = SENS.index(level)
    p2 = replace(p, min_rvol=[2.0, 1.5, 1.1, 0.8][i], rsi_long=[(58, 72), (55, 75), (50, 80), (45, 85)][i],
                 rsi_short=[(28, 42), (25, 45), (20, 50), (15, 55)][i], fresh_bars=[2, 3, 6, 10][i],
                 min_active=[4, 3, 2, 2][i], market_gate=["strict", "soft", "soft", "off"][i],
                 gap_min=[0.05, 0.04, 0.03, 0.02][i], runner_min=[0.06, 0.05, 0.04, 0.03][i],
                 setups=[("breakout", "orb", "gap"), ("breakout", "vwap", "orb", "gap"),
                         ("breakout", "vwap", "orb", "gap", "runner"), ("breakout", "vwap", "orb", "gap", "runner")][i])
    sp2 = replace(sp, adx_min=[25, 20, 18, 15][i], rsi_pull=[(30, 45), (30, 50), (30, 55), (25, 60)][i],
                  min_rvol_bo=[1.5, 1.3, 1.1, 1.0][i])
    return p2, sp2


def fmt(x):
    return f"${x:,.2f}" if x >= 1 else f"${x:,.4f}"


def latest_prices(dc, symbols):
    """One cheap API call for the live price of every pick (IEX real-time feed)."""
    if not symbols:
        return {}
    try:
        from alpaca.data.requests import StockLatestTradeRequest
        from alpaca.data.enums import DataFeed
        r = dc.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX))
        return {s: float(t.price) for s, t in r.items()}
    except Exception:
        return {}


def prev_closes(dc, symbols):
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data.enums import DataFeed
        sn = dc.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=list(symbols), feed=DataFeed.IEX))
        return {s: float(v.previous_daily_bar.close) for s, v in sn.items() if v.previous_daily_bar}
    except Exception:
        return {}


def pick_status(pk, live):
    """Plain-English live status for a pick."""
    if not live:
        return "waiting for price", "gray"
    d = 1 if pk["side"] == "LONG" else -1
    e, t, s = pk["entry"], pk["target"], pk["stop"]
    if d * (live - s) <= 0:
        return "STOP LEVEL HIT - skip", "red"
    if d * (live - t) >= 0:
        return "TARGET REACHED - too late", "amber"
    off = d * (live - e) / e
    if pk["kind"] == "watch":
        if off < -0.004:
            return f"WAIT: needs to break {fmt(e)}", "gray"
        return ("BREAKING OUT NOW" if d == 1 else "BREAKING DOWN NOW", "green") if off < 0.01 else ("ALREADY MOVED - don't chase", "amber")
    if off > 0.006:
        return "RAN PAST ENTRY - don't chase", "amber"
    if off < -0.012:
        return "BELOW ENTRY - wait for strength" if d == 1 else "ABOVE ENTRY - wait", "gray"
    return "IN BUY ZONE - good entry" if d == 1 else "IN SHORT ZONE - good entry", "green"


def spark_uri(vals, entry, target, stop, w=420, h=84):
    """Tiny price chart (as an image) with entry / target / stop lines."""
    vals = [float(v) for v in vals if v == v]
    if len(vals) < 3:
        return ""
    allv = vals + [entry, target, stop]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08 or max(abs(hi) * 0.01, 0.01)
    lo, hi = lo - pad, hi + pad
    X = lambda i: 6 + i * (w - 12) / (len(vals) - 1)
    Y = lambda v: h - 6 - (v - lo) / (hi - lo) * (h - 12)
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    col = "#16a34a" if vals[-1] >= vals[0] else "#dc2626"
    lines = "".join(f'<line x1="0" x2="{w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{c}" stroke-width="1.3" stroke-dasharray="5 4" opacity=".9"/>'
                    for v, c in ((entry, "#2563eb"), (target, "#16a34a"), (stop, "#dc2626")))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{lines}'
           f'<polygon points="{X(0):.1f},{h - 6} {pts} {X(len(vals) - 1):.1f},{h - 6}" fill="{col}" opacity=".13"/>'
           f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>'
           f'<circle cx="{X(len(vals) - 1):.1f}" cy="{Y(vals[-1]):.1f}" r="3.8" fill="{col}"/></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def card_html(pk, live):
    status, color = pick_status(pk, live)
    long_ = pk["side"] == "LONG"
    risk_ps = abs(pk["entry"] - pk["stop"])
    rr = abs(pk["target"] - pk["entry"]) / risk_ps if risk_ps else 0
    lbl = {"swing": "SWING BUY", "watch": "WATCH", "day": "DAY LONG" if long_ else "DAY SHORT"}[pk["kind"]]
    if pk["kind"] == "watch" and not long_:
        lbl = "WATCH SHORT"
    lcol = {"swing": "c-blue", "watch": "c-gray", "day": "c-green" if long_ else "c-red"}[pk["kind"]]
    accent = {"swing": "#2563eb", "watch": "#6b7280", "day": "#16a34a" if long_ else "#dc2626"}[pk["kind"]]
    l1 = ("BUY IF ABOVE" if pk["kind"] == "watch" else "BUY AT") if long_ else ("SHORT IF BELOW" if pk["kind"] == "watch" else "SHORT AT")
    l2 = "TAKE PROFIT" if long_ else "COVER (PROFIT)"
    vs = f" <small>({(live / pk['entry'] - 1) * 100:+.1f}% vs entry)</small>" if live else ""
    live_txt = f"{fmt(live)}{vs}" if live else "<small>no live price yet</small>"
    chips = []
    if pk.get("setup"):
        chips.append(f"<span class='tl-chip c-purple'>{html.escape(str(pk['setup']))}</span>")
    if pk["entry"] < 5:
        chips.append("<span class='tl-chip c-amber'>PENNY: wide spreads, can gap or halt</span>")
    if pk.get("earnings"):
        chips.append(f"<span class='tl-chip c-amber'>earnings {html.escape(str(pk['earnings']))}</span>")
    series = list(pk.get("spark") or [])
    if live and series:
        series = series[:-1] + [live] if len(series) > 3 else series + [live]
    img = spark_uri(series, pk["entry"], pk["target"], pk["stop"])
    score = max(5, min(100, int(pk.get("score", 50))))
    out = "".join([
        f"<div class='tl-card' style='border-left-color:{accent}'>",
        f"<div class='tl-top'><div><span class='tl-sym'>{html.escape(pk['symbol'])}</span>",
        f"<span class='tl-chip {lcol}'>{lbl}</span></div>",
        f"<div class='tl-rank'>{html.escape(str(pk.get('rank_txt', '')))}<div class='tl-meter'><div style='width:{score}%'></div></div></div></div>",
        f"<div style='margin-top:6px'>{''.join(chips)}</div>",
        f"<div class='tl-live'>Now {live_txt} <span class='tl-chip c-{color}'>{status}</span></div>",
        f"<img class='tl-spark' src='{img}'/>" if img else "",
        "<table class='tl-grid'><tr>",
        f"<td><div class='tl-box b-entry'><small>{l1}</small><b>{fmt(pk['entry'])}</b></div></td>",
        f"<td><div class='tl-box b-target'><small>{l2}</small><b>{fmt(pk['target'])}</b></div></td>",
        f"<td><div class='tl-box b-stop'><small>STOP LOSS</small><b>{fmt(pk['stop'])}</b></div></td></tr></table>",
        f"<div class='tl-meta'><span>Reward : risk <b>{rr:.1f} : 1</b></span><span><b>{int(pk['qty'])}</b> shares</span>",
        f"<span>Risk <b>{fmt(pk['risk_usd'])}</b></span><span>Goal <b>+{fmt(pk['reward_usd'])}</b></span></div>",
        f"<div class='tl-why'>{html.escape(str(pk.get('why', '')))}</div></div>"])
    return out.replace("$", "&#36;")


def esc(s):
    return str(s).replace("$", "\\$")


def build_swing_picks(sc, n):
    c = sc["cands"]
    ok = c[c.state == "BUY SIGNAL"] if not c.empty else c
    if not ok.empty:
        return [dict(r, kind="swing", rank_txt=f"Strength {r['score']}") for r in ok.head(n).to_dict("records")], \
            "✅ Clearest swing setups right now"
    w = sc["watch"]
    if not w.empty:
        return [dict(r, kind="watch", rank_txt=f"Strength {r['score']}") for r in w.head(n).to_dict("records")], \
            "👀 No clean buys today. Closest to triggering (wait for the price shown)"
    return [], "No setups found"


def build_day_picks(res, n):
    sg, wt = res["signals"], res["watch"]
    if not sg.empty:
        return [dict(r, kind="day", rank_txt=f"Strength {r['score']}") for r in sg.head(n).to_dict("records")], \
            f"⚡ {len(sg)} live day-trade signal{'s' if len(sg) != 1 else ''}"
    if not wt.empty:
        return [dict(r, kind="watch", rank_txt=f"Strength {r['score']}") for r in wt.head(n).to_dict("records")], \
            "👀 No trigger yet. Closest to firing (wait for the price shown)"
    return [], "Nothing close to firing right now"


def hot_chips_html(hot, k=16):
    if hot is None or hot.empty:
        return ""
    t = hot.head(k)
    return "<div>" + "".join(f"<span class='tl-pill {'hot-up' if r.chg >= 0 else 'hot-dn'}'>{html.escape(r.symbol)} {r.chg * 100:+.1f}%</span>"
                             for r in t.itertuples()) + "</div>"


def funnel_html(hd, p, res):
    items = [f"🌎 {hd.get('universe', 0):,} US stocks", f"📊 {hd.get('snapped', 0):,} traded today",
             f"🔥 {hd.get('eligible', 0):,} above ${p.min_price:g}", f"🔎 {res['n']} scanned by the minute",
             f"✅ {len(res['signals'])} signals", f"👀 {len(res['watch'])} watch"]
    return ("<div>" + "".join(f"<span class='tl-pill'>{i}</span>" for i in items) + "</div>").replace("$", "&#36;")


def show_df(st, df, **kw):
    if df is None:
        return
    st.dataframe(df.drop(columns=[c for c in ("spark",) if c in df.columns]), hide_index=True, **kw)


def _fragment(st, run_every):
    f = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
    if f is None:
        return lambda fn: fn
    try:
        return f(run_every=run_every)
    except TypeError:
        return f


def dashboard():
    import streamlit as st
    st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    ss = st.session_state
    for k, env in (("k_key", "ALPACA_KEY"), ("k_sec", "ALPACA_SECRET"), ("k_fh", "FINNHUB_KEY")):
        ss.setdefault(k, os.environ.get(env, ""))
    st.markdown(HERO, unsafe_allow_html=True)
    with st.expander("🔑 Connect your Alpaca paper account", expanded=not (ss["k_key"] and ss["k_sec"])):
        st.text_input("API key", type="password", key="k_key")
        st.text_input("API secret", type="password", key="k_sec")
        st.text_input("Finnhub key (optional, shows earnings dates)", type="password", key="k_fh")
        st.caption("Paper keys only. Type them each visit; nothing is saved.")
    key, secret, fh = ss["k_key"], ss["k_sec"], ss["k_fh"]

    def clients():
        if not (key and secret):
            return None
        try:
            return live_clients(key, secret, paper=True)
        except Exception as e:
            st.error(f"Alpaca error: {e}")
            return None

    cl = clients()
    dc, tc = (cl if cl else (None, None))
    T = st.tabs(["🏆 Top Picks", "🌎 Market", "🧪 Test It", "⚙️ Settings", "📘 Guide"])

    # ---------------- Settings (built first so every tab can use it) ----------------
    with T[3]:
        style = st.radio("Trading style", list(PRESETS), index=2, horizontal=True, key="style")
        ps = PRESETS[style]
        st.caption(ps["note"])
        n_picks = st.slider("How many picks to show", 3, 12, 6)
        lowpx = st.slider("Lowest stock price to scan ($): lower = more penny stocks", 0.5, 20.0, float(ps["lowpx"]), 0.5, key=f"lowpx_{style}")
        inc_etf = st.checkbox("Include ETFs in day-trade scan", False, key="inc_etf")
        with st.expander("Fine-tune swing rules"):
            sp = SwingParams(
                setup=st.selectbox("Setups", ["all", "both", "pullback", "breakout", "leader"], key="s_setup"),
                risk_per_trade=st.slider("Risk per trade (%)", 0.25, 2.0, float(ps["risk"]), 0.25, key=f"s_risk_{style}") / 100,
                max_positions=st.slider("Max positions", 1, 20, int(ps["maxpos"]), key=f"s_mp_{style}"),
                stop_atr=st.slider("Stop distance (x ATR)", 1.0, 4.0, float(ps["stop"]), 0.5, key=f"s_stop_{style}"),
                trail_atr=st.slider("Trailing stop (x ATR)", 2.0, 6.0, float(ps["trail"]), 0.5, key=f"s_trail_{style}"),
                target_rr=st.slider("Profit target (x risk)", 1.0, 6.0, float(ps["rr"]), 0.5, key=f"s_rr_{style}"),
                max_hold=st.slider("Max hold (days)", 5, 60, 25, key="s_hold"),
                adx_min=st.slider("Min trend strength (ADX)", 10, 35, int(ps["adx"]), key=f"s_adx_{style}"),
                min_price=st.slider("Min price ($)", 0.5, 50.0, float(ps["s_minp"]), 0.5, key=f"s_minp_{style}"),
                min_dollar_vol=st.slider("Min avg $ volume (millions)", 1, 100, int(ps["s_dv"]), key=f"s_dv_{style}") * 1e6,
                use_regime=st.checkbox("Only buy when SPY > 200-day average", True, key="s_reg"),
                earnings_buffer=st.slider("Skip if earnings within (days)", 0, 14, 5, key="s_earn"))
        with st.expander("Fine-tune day-trade rules"):
            syms = [x.strip().upper() for x in st.text_area("Always-scan watchlist", ", ".join(DEFAULT_WATCHLIST), key="d_wl").split(",") if x.strip()]
            side = st.radio("Direction", ["long", "short", "both"], index=2, horizontal=True, key="d_side")
            p = Params(watchlist=syms, min_price=lowpx, include_etf=inc_etf,
                       risk_per_trade=st.slider("Risk per trade (%)", 0.1, 3.0, float(ps["d_risk"]), 0.1, key=f"d_risk_{style}") / 100,
                       rr=st.slider("Reward : risk", 1.0, 4.0, float(ps["d_rr"]), 0.5, key=f"d_rr_{style}"),
                       atr_mult=st.slider("Stop (x ATR)", 0.5, 3.0, 1.5, 0.1, key="d_atr"),
                       min_stop_pct=st.slider("Minimum stop distance (%)", 0.2, 2.0, 0.6, 0.1, key="d_msp") / 100,
                       daily_loss_limit=st.slider("Daily loss limit (%)", 1.0, 10.0, float(ps["d_loss"]), 0.5, key=f"d_loss_{style}") / 100,
                       max_open=st.slider("Max open trades", 1, 10, int(ps["d_open"]), key=f"d_open_{style}"),
                       hot_top=st.slider("Stocks to scan by the minute", 50, 300, 150, 10, key="d_hot"),
                       allow_long=side in ("long", "both"), allow_short=side in ("short", "both"))
        with st.expander("Test costs"):
            eq0 = st.number_input("Start equity for tests ($)", 1000.0, 10_000_000.0, 25000.0, key="c_eq")
            slip = st.slider("Slippage per side (bps)", 0, 50, 8, key="c_slip") / 10000
            comm = st.number_input("Commission ($/share/side)", 0.0, 0.1, 0.0, 0.001, format="%.3f", key="c_comm")

    # ---------------- Top Picks ----------------
    with T[0]:
        if cl is None:
            st.info("👆 Open **Connect** at the top and paste your paper keys to begin.")
        mode = st.radio("What kind of trade?", ["Day trade (in and out today)", "Swing (hold days to weeks)"],
                        horizontal=True, key="mode")
        swing = mode.startswith("Swing")
        mode_key = "swing" if swing else "day"
        sens = st.select_slider("How many signals do you want?", options=SENS, value="More signals", key="sens")
        st.caption(SENS_NOTE[sens])
        p_eff, sp_eff = apply_sensitivity(p, sp, sens)
        live = st.toggle("Live updates every second", True, key="live")
        now0 = datetime.now(NY)
        mkt_open = now0.weekday() < 5 and (9, 30) <= (now0.hour, now0.minute) < (16, 0)
        run_every = 1 if (live and mkt_open and cl) else None

        def run_day_scan(progress=None):
            try:
                eq = float(tc.get_account().equity)
                pr = p_eff
                if ss.get("hot") is None or time.time() - ss.get("hot_t", 0) > 180:
                    hot, hd = hot_list(dc, tc, p_eff, progress)
                    ss["hot"], ss["hot_diag"], ss["hot_t"] = hot, hd, time.time()
                    ss["short_ok"] = shortable_set(tc, p_eff.include_etf)
                if ss.get("hot") is None or ss["hot"].empty:        # snapshot unavailable -> fall back to movers list
                    lst, _ = dynamic_watchlist(key, secret, p_eff.watchlist)
                    pr = replace(p_eff, watchlist=lst)
                avoid = set()
                if fh:
                    e = earnings_calendar(fh, 2)
                    avoid = set(e.symbol) if not e.empty else set()
                res = day_scan(dc, pr, eq, ss.get("hot"), ss.get("short_ok"), avoid)
                ss["dres"], ss["t_scan"] = res, time.time()
                ss["picks"], ss["picks_title"] = build_day_picks(res, n_picks)
                ss["picks_mode"] = "day"
            except Exception as e:
                st.error(f"Scan failed: {e}")

        if swing:
            maxsym = st.slider("How many stocks to scan (most liquid first)", 200, 4000, 1500, 100, key="maxsym")
            if st.button("🔍 Find the best swing stocks", type="primary", key="go_swing") and cl:
                bar = st.progress(0.0, text="Scanning the market...")
                try:
                    eq = float(tc.get_account().equity)
                    sc = swing_scan(dc, tc, sp_eff, eq, maxsym, fh or None,
                                    lambda f: bar.progress(min(float(f), 1.0), text="Downloading market data..."))
                    ss["sscan"] = sc
                    ss["picks"], ss["picks_title"] = build_swing_picks(sc, n_picks)
                    ss["picks_mode"] = "swing"
                except Exception as e:
                    st.error(f"Scan failed: {e}")
                bar.empty()
            sc = ss.get("sscan")
            if sc:
                kc = {"success": "green", "warning": "amber", "error": "red"}[sc["ov"]["level"]]
                st.markdown(f"<div class='tl-banner k-{kc}'>{html.escape(sc['ov']['text'])}</div>", unsafe_allow_html=True)
                st.caption(f"Scanned {sc['n']} liquid stocks from {sc['n_uni']} US listings | data as of {sc['asof']} close | {sc['feed']}")
            else:
                st.caption("Tap the button. The first scan takes a few minutes; after that it's quick.")
        else:
            if st.button("⚡ Scan the whole market now", type="primary", key="go_day") and cl:
                bar = st.progress(0.0, text="Scanning every US stock...")
                ss["hot"] = None
                run_day_scan(lambda f: bar.progress(min(float(f), 1.0), text="Scanning every US stock..."))
                bar.empty()
            if not mkt_open:
                st.caption("Market is closed. Day-trade signals appear from 9:40am to 3:30pm New York time.")
            res = ss.get("dres")
            if res is not None and ss.get("picks_mode") == "day":
                txt, kc = {1: ("Market is leaning UP (SPY above VWAP).", "green"), -1: ("Market is leaning DOWN (SPY below VWAP).", "red"),
                           0: ("Market direction unclear.", "amber")}[res["mkt"]]
                st.markdown(f"<div class='tl-banner k-{kc}'>{txt}</div>", unsafe_allow_html=True)
                if ss.get("hot_diag"):
                    st.markdown(funnel_html(ss["hot_diag"], p_eff, res), unsafe_allow_html=True)
                st.markdown("**🔥 Hottest right now**")
                st.markdown(hot_chips_html(ss.get("hot")), unsafe_allow_html=True)

        @_fragment(st, run_every)
        def board():
            now = datetime.now(NY)
            if not swing and live and mkt_open and cl and time.time() - ss.get("t_scan", 0) > 60:
                run_day_scan()
            on = run_every is not None
            pulse = {}
            if dc:
                if "pcmap" not in ss or time.time() - ss.get("pcmap_t", 0) > 6 * 3600:
                    ss["pcmap"], ss["pcmap_t"] = prev_closes(dc, INDEX_ETFS), time.time()
                picks0 = ss.get("picks") or [] if ss.get("picks_mode") == mode_key else []
                allp = latest_prices(dc, list(dict.fromkeys([pk["symbol"] for pk in picks0] + INDEX_ETFS)))
                ss.setdefault("last_px", {}).update(allp)
                pulse = {s: allp[s] / ss["pcmap"][s] - 1 for s in INDEX_ETFS if s in allp and s in ss["pcmap"]}
            chips = [f"<span class='tl-chip c-gray'>🕒 {now:%H:%M:%S} New York</span>",
                     (f"<span class='tl-chip c-green'><span class='tl-dot'></span>LIVE 1s</span>" if on else "<span class='tl-chip c-gray'>○ paused</span>"),
                     f"<span class='tl-chip c-{'green' if mkt_open else 'gray'}'>Market {'OPEN' if mkt_open else 'CLOSED'}</span>"]
            chips += [f"<span class='tl-chip c-{'green' if v >= 0 else 'red'}'>{s} {v * 100:+.2f}%</span>" for s, v in pulse.items()]
            st.markdown("<div class='tl-bar'>" + "".join(chips) + "</div>", unsafe_allow_html=True)
            if ss.get("msg"):
                st.success(esc(ss.pop("msg")))
            picks = (ss.get("picks") or []) if ss.get("picks_mode") == mode_key else []
            if not picks:
                st.write("No picks yet. Tap the button above." if ss.get("picks_mode") != mode_key else
                         "Nothing close to firing right now. Try **More signals** or **Maximum** above.")
                return
            st.subheader(ss.get("picks_title", "Top picks"))
            for i, pk in enumerate(picks):
                lp = ss["last_px"].get(pk["symbol"])
                st.markdown(card_html(pk, lp), unsafe_allow_html=True)
                if pk["kind"] == "watch":
                    continue
                stat, _ = pick_status(pk, lp)
                verb = "BUY" if pk["side"] == "LONG" else "SELL SHORT"
                if st.button(f"Paper {verb} {int(pk['qty'])} {pk['symbol']}", key=f"ord_{mode_key}_{pk['symbol']}_{i}",
                             disabled=(int(pk["qty"]) < 1 or stat.startswith("STOP"))):
                    try:
                        if pk["kind"] == "swing":
                            place_swing_order(tc, pk["symbol"], pk["qty"], pk["stop"], pk["target"])
                        else:
                            place_order(tc, pk["symbol"], pk["side"], pk["qty"], pk["stop"], pk["target"])
                        ss["msg"] = f"Paper order sent: {verb} {int(pk['qty'])} {pk['symbol']} with stop {fmt(pk['stop'])} and target {fmt(pk['target'])}."
                    except Exception as e:
                        st.error(f"Order failed: {e}")

        board()
        st.caption("Live prices come from the free IEX feed (a slice of total volume), so thinly traded stocks can look stale. "
                   "Day signals refresh each minute; the whole-market list every 3 minutes; swing signals are daily.")
        if not swing and ss.get("dres") is not None:
            with st.expander("🔬 Why so few signals? (filter counts)"):
                dg = ss["dres"]["diag"]
                st.write(f"Of **{dg['with_data']}** stocks with fresh 1-minute data: **{dg['above_vwap']}** are above VWAP, "
                         f"**{dg['volume_ok']}** have enough volume, **{dg['near_high']}** are within 0.8% of their 20-minute high, "
                         f"**{dg['signals']}** fired a signal.")
                st.write("Tips: slide to **More signals** or **Maximum**, lower the price floor in Settings, switch Direction to *both*, "
                         "or check back between 9:45am and 11:30am when most setups fire. The free feed sees only a few percent of "
                         "total volume, so thin stocks often fail the activity filter.")
        with st.expander("📂 My open paper positions"):
            if st.button("Load positions", key="load_pos") and cl:
                try:
                    ss["pos"] = swing_positions(tc, dc, sp_eff)
                except Exception as e:
                    st.error(f"Could not load: {e}")
            pos = ss.get("pos")
            if pos is not None:
                st.dataframe(pos, hide_index=True) if not pos.empty else st.write("No open positions.")
        with st.expander("📊 Full scan details"):
            if swing and ss.get("sscan"):
                st.write("**Strongest trending stocks (what might go up)**")
                show_df(st, ss["sscan"]["leaders"])
                if not ss["sscan"]["earn"].empty:
                    st.write("**Earnings coming up**")
                    st.dataframe(ss["sscan"]["earn"], hide_index=True)
            elif not swing and ss.get("dres") is not None:
                st.write("**Signals**")
                show_df(st, ss["dres"]["signals"])
                st.write("**Closest to firing**")
                show_df(st, ss["dres"]["watch"])
                st.write("**Whole-market hot list (top 60)**")
                if ss.get("hot") is not None:
                    st.dataframe(ss["hot"].head(60).round(3), hide_index=True)
            else:
                st.write("Run a scan first.")

    # ---------------- Market ----------------
    with T[1]:
        if st.button("Refresh market view", key="ref_mkt") and cl:
            with st.spinner("Loading..."):
                try:
                    ss["ov"] = market_light(dc)
                except Exception as e:
                    st.error(f"Could not load: {e}")
        ov = ss.get("ov") or (ss.get("sscan") or {}).get("ov")
        if ov:
            kc = {"success": "green", "warning": "amber", "error": "red"}[ov["level"]]
            st.markdown(f"<div class='tl-banner k-{kc}'>{html.escape(ov['text'])}</div>", unsafe_allow_html=True)
            st.caption(f"As of {ov['asof']} close")
            st.dataframe(ov["indices"], hide_index=True)
            st.subheader("Sector leaders (vs SPY, 3 months)")
            st.dataframe(ov["sectors"], hide_index=True)
            if ov.get("breadth"):
                b = ov["breadth"]
                c = st.columns(3)
                c[0].metric("Above 50-day", f"{b['above_50d']:.0%}")
                c[1].metric("Above 200-day", f"{b['above_200d']:.0%}")
                c[2].metric("Near 52-wk high", f"{b['near_high']:.0%}")
                st.caption(f"Across {b['n']} liquid stocks. Over 60% above their 50-day average = healthy market.")
        else:
            st.write("Tap **Refresh market view** (or run the swing scan, it includes this).")

    # ---------------- Test It ----------------
    with T[2]:
        st.write("Check whether the strategy has actually worked before you trust it. Uses the same settings as Top Picks.")
        kind = st.radio("What to test", ["Day-trading strategy", "Swing strategy"], horizontal=True, key="t_kind")
        real = st.radio("Data", ["Real market data", "Practice data (fake, only checks the code)"], key="t_src").startswith("Real")
        if kind.startswith("Swing"):
            c1, c2 = st.columns(2)
            yrs = c1.slider("Years of history", 1, 5, 3)
            nsym = c2.slider("Stocks to test", 100, 1000, 300, 50)
            if st.button("▶ Run full swing test", type="primary", key="t_run_swing"):
                if real and not cl:
                    st.error("Connect your keys first.")
                else:
                    bar = st.progress(0.0, text="Loading data...")
                    try:
                        today = datetime.now().date()
                        P, F, note = load_swing_data("alpaca" if real else "synthetic", today - timedelta(days=365 * yrs), today,
                                                     nsym, key or None, secret or None,
                                                     lambda f: bar.progress(min(float(f), 1.0), text="Downloading history..."))
                        bar.progress(1.0, text="Simulating...")
                        tr, cv, expo = run_swing(P, F, sp_eff, eq0, slip, comm)
                        spy = P["C"]["SPY"] if "SPY" in P["C"] else None
                        s, bm = stats(tr, cv, eq0), bench_stats(spy, cv)
                        stress = stress_table(lambda sl: run_swing(P, F, sp_eff, eq0, sl, comm)[:2], max(slip, 0.0005), eq0)
                        mc = monte_carlo(tr)
                        rows, s_all, cv_all, tr_all, verdict = walk_forward_swing(P, F, sp_eff, eq0, slip, comm)
                        bar.empty()
                        pf2 = stress["profit_factor"].iloc[1] if len(stress) > 1 else float("nan")
                        passed = verdict.startswith("Some") and pf2 == pf2 and pf2 > 1.0
                        (st.success if passed else st.error)(
                            ("PROMISING: held up on unseen data and with worse fills. Still paper trade for weeks. " if passed else
                             "NOT PROVEN: it did not hold up on unseen data or with worse fills. Don't use real money. ") + verdict)
                        st.caption(note)
                        m = st.columns(4)
                        m[0].metric("Return", f"{s['ret']:+.1%}", f"SPY {bm['ret']:+.1%}" if bm else None)
                        m[1].metric("Worst drop", f"{s['max_dd']:.1%}", f"SPY {bm['dd']:.1%}" if bm else None, delta_color="off")
                        m[2].metric("Profit factor", "-" if not s["trades"] else f"{s['profit_factor']:.2f}")
                        m[3].metric("Trades", s["trades"])
                        if s["trades"]:
                            chart = pd.DataFrame({"Strategy": cv / eq0})
                            if bm is not None:
                                chart["SPY buy & hold"] = bm["curve"]
                            st.line_chart(chart)
                            st.write("**By setup**")
                            st.dataframe(tr.groupby("kind").agg(trades=("pnl", "count"), win_rate=("pnl", lambda x: (x > 0).mean()),
                                                                 avg_R=("R", "mean"), total_pnl=("pnl", "sum")).round(2))
                            st.write("**Walk-forward** (tuned on the past, graded on unseen future)")
                            st.dataframe(rows, hide_index=True)
                            if s_all:
                                st.write(f"Unseen-data result: return {s_all['ret']:+.1%}, profit factor {s_all['profit_factor']:.2f}, "
                                         f"worst drop {s_all['max_dd']:.1%}, {s_all['trades']} trades.")
                            st.write("**Worse fills test**")
                            st.dataframe(stress, hide_index=True)
                            if mc:
                                st.write(f"**Luck range:** typical {mc['ret_p50']:+.0%}, bad luck {mc['ret_p5']:+.0%}, good luck {mc['ret_p95']:+.0%}. "
                                         f"Chance of losing money: {mc['prob_loss']:.0%}.")
                            with st.expander("All trades"):
                                st.dataframe(tr)
                                st.download_button("Download trades CSV", tr.to_csv(index=False), "swing_trades.csv")
                    except Exception as e:
                        bar.empty()
                        st.error(f"Test failed: {e}")
        else:
            days = st.slider("Days of 1-minute history", 5, 40, 20)
            st.caption("Tests your always-scan watchlist (one-minute history for the whole market is too big to download). "
                       "Add volatile names there to test more setups.")
            if st.button("▶ Run day-trading test", type="primary", key="t_run_day"):
                if real and not cl:
                    st.error("Connect your keys first.")
                else:
                    try:
                        with st.spinner("Loading and simulating..."):
                            today = datetime.now().date()
                            data = load_alpaca(p_eff.watchlist, str(today - timedelta(days=days)), str(today), key, secret) if real else synthetic(p_eff.watchlist)
                            prep = prep_data(data)
                            tr, cv = run(prep, p_eff, eq0, slip, comm)
                            s = stats(tr, cv, eq0)
                            stress = stress_table(lambda sl: run(prep, p_eff, eq0, sl, comm), max(slip, 0.0005), eq0)
                            res, verdict = sweep(prep, p_eff, eq0, slip, comm)
                        m = st.columns(4)
                        m[0].metric("Return", f"{s['ret']:+.1%}")
                        m[1].metric("Worst drop", f"{s['max_dd']:.1%}")
                        m[2].metric("Profit factor", "-" if not s["trades"] else f"{s['profit_factor']:.2f}")
                        m[3].metric("Trades", s["trades"])
                        if s["trades"]:
                            st.line_chart(cv)
                            st.write("**Which setups work** (the honest way to decide what to keep)")
                            st.dataframe(tr.groupby("kind").agg(trades=("pnl", "count"), win_rate=("pnl", lambda x: (x > 0).mean()),
                                                                 avg_R=("R", "mean"), total_pnl=("pnl", "sum")).round(2))
                            st.write("**Worse fills test**")
                            st.dataframe(stress, hide_index=True)
                        if not res.empty:
                            st.write("**Tune on early days, grade on later days**")
                            st.dataframe(res, hide_index=True)
                        (st.success if verdict.startswith("Some") else st.error)(verdict)
                        st.caption("Day trading has the highest cost drag. Few retail day traders beat it.")
                    except Exception as e:
                        st.error(f"Test failed: {e}")

    with T[4]:
        st.markdown(GUIDE)


# =====================================================================
# 9. COMMAND LINE
# =====================================================================
def build_params(a):
    p = Params()
    if getattr(a, "symbols", None): p.watchlist = a.symbols
    for arg, attr in [("rr", "rr"), ("atr", "atr_mult"), ("rvol", "min_rvol"), ("risk", "risk_per_trade")]:
        if getattr(a, arg, None) is not None:
            setattr(p, attr, getattr(a, arg))
    p.allow_long, p.allow_short = a.side in ("long", "both"), a.side in ("short", "both")
    return p


def build_swing(a):
    sp = SwingParams(setup=a.setup)
    if a.stop_atr: sp.stop_atr = a.stop_atr
    if a.trail_atr: sp.trail_atr = a.trail_atr
    if a.target_rr is not None: sp.target_rr = a.target_rr
    sp.use_regime = not a.no_regime
    return sp


def get_data(a, p):
    if a.synthetic:
        print("SYNTHETIC random data: tests the code only, says nothing about profitability.")
        return synthetic(p.watchlist)
    if a.csv:
        return load_csv(a.csv)
    if a.start:
        return load_alpaca(p.watchlist, a.start, a.end)
    sys.exit("Choose --synthetic, --csv FILE, or --start YYYY-MM-DD")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def money(sp):
        sp.add_argument("--equity", type=float, default=25000)
        sp.add_argument("--slippage", type=float, default=0.0008, help="per side, fraction (0.0008 = 8 bps)")
        sp.add_argument("--commission", type=float, default=0.0, help="$ per share per side")

    def day_common(sp):
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--csv")
        sp.add_argument("--start")
        sp.add_argument("--end", default=str(datetime.now().date()))
        sp.add_argument("--symbols", nargs="*")
        sp.add_argument("--side", choices=["long", "short", "both"], default="both")
        money(sp)
        for n in ("rr", "atr", "rvol", "risk"):
            sp.add_argument(f"--{n}", type=float)

    def swing_common(sp):
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--start", default=str((datetime.now() - timedelta(days=3 * 365)).date()))
        sp.add_argument("--end", default=str(datetime.now().date()))
        sp.add_argument("--max-symbols", type=int, default=300)
        sp.add_argument("--setup", choices=["all", "both", "pullback", "breakout", "leader"], default="all")
        sp.add_argument("--stop-atr", type=float)
        sp.add_argument("--trail-atr", type=float)
        sp.add_argument("--target-rr", type=float)
        sp.add_argument("--no-regime", action="store_true")
        money(sp)

    day_common(sub.add_parser("backtest", help="1-minute day-trading backtest (your watchlist)"))
    sw = sub.add_parser("sweep", help="day-trading train/test sweep"); day_common(sw)
    sw.add_argument("--train-frac", type=float, default=0.7)
    sw.add_argument("--min-trades", type=int, default=30)
    swing_common(sub.add_parser("swing-backtest", help="daily-bar swing backtest across many stocks"))
    swing_common(sub.add_parser("swing-walkforward", help="swing walk-forward (honest out-of-sample test)"))
    sub.add_parser("swing-scan", help="scan the whole market for swing setups now")
    ds = sub.add_parser("day-scan", help="scan the whole market for day-trade signals now")
    ds.add_argument("--side", choices=["long", "short", "both"], default="both")
    ds.add_argument("--symbols", nargs="*")
    for n in ("rr", "atr", "rvol", "risk"):
        ds.add_argument(f"--{n}", type=float)
    lv = sub.add_parser("live", help="run the paper day-trading bot")
    lv.add_argument("--side", choices=["long", "short", "both"], default="both")
    lv.add_argument("--symbols", nargs="*")
    lv.add_argument("--real-money", action="store_true", help="NOT recommended")
    for n in ("rr", "atr", "rvol", "risk"):
        lv.add_argument(f"--{n}", type=float)
    sub.add_parser("dashboard", help="launch the web dashboard")

    a = ap.parse_args()
    if a.cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
    elif a.cmd == "live":
        paper = not a.real_money
        if not paper and input("Type I ACCEPT THE RISK to trade real money: ") != "I ACCEPT THE RISK":
            sys.exit("Cancelled.")
        live_loop(build_params(a), paper)
    elif a.cmd == "day-scan":
        key, secret = get_keys()
        if not (key and secret):
            sys.exit("Set ALPACA_KEY and ALPACA_SECRET first.")
        dc, tc = live_clients(key, secret, True)
        p = build_params(a)
        hot, hd = hot_list(dc, tc, p, lambda f: print(f"\rscanning market {f:.0%}", end=""))
        print(f"\n{hd}")
        res = day_scan(dc, p, float(tc.get_account().equity), hot, shortable_set(tc, p.include_etf))
        cols = ["symbol", "setup", "side", "entry", "target", "stop", "qty", "rvol", "chg", "score", "why"]
        for name in ("signals", "watch"):
            df = res[name]
            print(f"\n{name.upper()}:")
            print(df[cols].to_string(index=False) if not df.empty else "none")
        print("\nFilter counts:", res["diag"])
    elif a.cmd in ("backtest", "sweep"):
        p = build_params(a)
        prep = prep_data(get_data(a, p))
        if a.cmd == "backtest":
            tr, cv = run(prep, p, a.equity, a.slippage, a.commission)
            print_stats(stats(tr, cv, a.equity), tr, a.equity, title="DAY-TRADING BACKTEST")
            if not tr.empty:
                print("\nSlippage stress test:\n", stress_table(lambda s: run(prep, p, a.equity, s, a.commission),
                                                                   max(a.slippage, 0.0005), a.equity).to_string(index=False))
                mc = monte_carlo(tr)
                if mc:
                    print(f"\nMonte Carlo: typical {mc['ret_p50']:+.0%}, bad luck {mc['ret_p5']:+.0%}, "
                          f"chance of losing money {mc['prob_loss']:.0%}")
                tr.to_csv("backtest_trades.csv", index=False)
                print("Saved backtest_trades.csv")
        else:
            df, verdict = sweep(prep, p, a.equity, a.slippage, a.commission, a.train_frac, a.min_trades)
            if not df.empty:
                print(df.to_string(index=False))
            print("\n" + verdict)
    elif a.cmd in ("swing-backtest", "swing-walkforward"):
        P, F, note = load_swing_data("synthetic" if a.synthetic else "alpaca", a.start, a.end, a.max_symbols)
        print(note)
        sp = build_swing(a)
        if a.cmd == "swing-backtest":
            tr, cv, expo = run_swing(P, F, sp, a.equity, a.slippage, a.commission)
            spy = P["C"]["SPY"] if "SPY" in P["C"] else None
            print_stats(stats(tr, cv, a.equity), tr, a.equity, bench_stats(spy, cv), "SWING BACKTEST")
            if not tr.empty:
                print(f"Time invested   : {expo:.0%}")
                print("\nSlippage stress test:\n", stress_table(lambda s: run_swing(P, F, sp, a.equity, s, a.commission)[:2],
                                                                   max(a.slippage, 0.0005), a.equity).to_string(index=False))
                mc = monte_carlo(tr)
                if mc:
                    print(f"\nMonte Carlo: typical {mc['ret_p50']:+.0%} (bad luck {mc['ret_p5']:+.0%}, good luck {mc['ret_p95']:+.0%}); "
                          f"typical worst drawdown {mc['dd_p50']:.0%}; chance of losing money {mc['prob_loss']:.0%}")
                tr.to_csv("swing_trades.csv", index=False)
                print("Saved swing_trades.csv")
        else:
            rows, s_all, cv_all, tr_all, verdict = walk_forward_swing(P, F, sp, a.equity, a.slippage, a.commission)
            if not rows.empty:
                print(rows.to_string(index=False))
            if s_all:
                print_stats(s_all, tr_all, a.equity, bench_stats(P["C"]["SPY"] if "SPY" in P["C"] else None, cv_all),
                            "WALK-FORWARD (OUT-OF-SAMPLE ONLY)")
            print("\n" + verdict)
    elif a.cmd == "swing-scan":
        key, secret = get_keys()
        if not (key and secret):
            sys.exit("Set ALPACA_KEY and ALPACA_SECRET first.")
        dc, tc = live_clients(key, secret, True)
        res = swing_scan(dc, tc, SwingParams(), float(tc.get_account().equity), 1500, os.environ.get("FINNHUB_KEY"),
                         lambda f: print(f"\rdownloading {f:.0%}", end=""))
        print("\n" + res["ov"]["text"])
        cols = ["symbol", "setup", "entry", "target", "stop", "qty", "score", "why"]
        print(f"\nBuy candidates ({res['asof']}):")
        print(res["cands"][cols].to_string(index=False) if not res["cands"].empty else "none today")
    else:
        ap.print_help()


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    dashboard() if _in_streamlit() else main()
