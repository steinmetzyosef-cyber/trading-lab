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
import os, sys, time, json, pickle, argparse, itertools, subprocess, tempfile
import urllib.request, urllib.parse
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "SPY", "QQQ"]
SECTORS = {"XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Health Care",
           "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLI": "Industrials",
           "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication"}
INDEX_ETFS = ["SPY", "QQQ", "IWM"]
ETF_SET = set(SECTORS) | set(INDEX_ETFS)
BAD_WORDS = ["etf", "etn", "fund", "ishares", "proshares", "direxion", "spdr", "invesco", "vanguard",
             "wisdomtree", "vaneck", "index", "warrant", "preferred", "depositary", "notes due"]
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
# 1. DAY-TRADING STRATEGY (1-minute bars)
# =====================================================================
@dataclass
class Params:
    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.03
    rr: float = 2.0
    atr_mult: float = 1.5
    min_rvol: float = 1.5
    rsi_long: tuple = (55, 75)
    rsi_short: tuple = (25, 45)
    max_open: int = 3
    max_pos_pct: float = 0.25
    cooldown_min: int = 15
    start_after: tuple = (9, 45)
    flatten_at: tuple = (15, 50)
    allow_long: bool = True
    allow_short: bool = False
    min_price: float = 5.0
    market_filter: bool = True     # longs only if SPY > its VWAP, shorts only if below


def indicators(df):
    """ONE trading day of 1-min bars. Past data only (no lookahead)."""
    df = df.copy()
    tp = (df.high + df.low + df.close) / 3
    df["vwap"] = (tp * df.volume).cumsum() / df.volume.cumsum()
    delta = df.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["rvol"] = df.volume / df.volume.rolling(30).mean()
    df["hi20"] = df.high.rolling(20).max().shift()
    df["lo20"] = df.low.rolling(20).min().shift()
    return df


def prep_day(df):
    d = indicators(df)
    out = {k: d[k].to_numpy(float) for k in
           ["open", "high", "low", "close", "volume", "atr", "vwap", "rsi", "rvol", "hi20", "lo20"]}
    out["ts"] = d.index
    out["mins"] = (d.index.hour * 60 + d.index.minute).to_numpy()
    return out


def signals(d, p):
    """+1 long, -1 short, 0 none for every bar. Live trading uses the last element."""
    m, n = d["mins"], len(d["mins"])
    lo, hi = p.start_after[0] * 60 + p.start_after[1], p.flatten_at[0] * 60 + p.flatten_at[1]
    sig = np.zeros(n, int)
    with np.errstate(invalid="ignore"):
        ok = ((m >= lo) & (m < hi) & (d["atr"] > 0) & (d["rvol"] >= p.min_rvol)
              & (d["close"] >= p.min_price))
        if p.allow_long:
            sig[ok & (d["close"] > d["vwap"]) & (d["close"] > d["hi20"]) &
                (d["rsi"] >= p.rsi_long[0]) & (d["rsi"] <= p.rsi_long[1])] = 1
        if p.allow_short:
            sig[ok & (d["close"] < d["vwap"]) & (d["close"] < d["lo20"]) &
                (d["rsi"] >= p.rsi_short[0]) & (d["rsi"] <= p.rsi_short[1])] = -1
        if p.market_filter and "mkt" in d:
            sig[(sig == 1) & (d["mkt"] < 0)] = 0
            sig[(sig == -1) & (d["mkt"] > 0)] = 0
    return sig


def levels(entry, atr, side, p):
    risk = p.atr_mult * atr
    return (entry - side * risk, entry + side * p.rr * risk)


def size(equity, entry, stop, p):
    per_share = abs(entry - stop)
    if per_share <= 0:
        return 0
    return max(0, min(int(equity * p.risk_per_trade / per_share), int(equity * p.max_pos_pct / entry)))


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
    symbols = list(dict.fromkeys(list(symbols) + ["SPY"]))     # SPY powers the market filter
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
    """Random 1-minute data to test code. Results say NOTHING about profitability."""
    rng, out = np.random.default_rng(seed), {}
    bdays = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days)
    for s in dict.fromkeys(list(symbols) + ["SPY"]):
        price, frames = 50 + rng.random() * 200, []
        for d in bdays:
            idx = pd.date_range(f"{d.date()} 09:30", periods=390, freq="min", tz=NY)
            ret = rng.normal(0, 0.0007, 390)
            ret[rng.integers(0, 330):][:25] += rng.normal(0.0006, 0.0006)
            close = price * np.exp(np.cumsum(ret))
            open_ = np.r_[price, close[:-1]]
            wick = np.abs(rng.normal(0, 0.0004, 390)) * close
            vol = 1e5 * np.exp(rng.normal(0, 0.4, 390)) * (1 + 3 * (np.abs(ret) > 0.0015))
            frames.append(pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick,
                                        "low": np.minimum(open_, close) - wick,
                                        "close": close, "volume": vol}, index=idx))
            price = close[-1]
        out[s] = pd.concat(frames)
    return out


def prep_data(data):
    """{symbol: 1-min bars} -> {day: {symbol: arrays}} with the SPY market-direction gate attached."""
    prep, mk = {}, {}
    if "SPY" in data:
        for day, g in data["SPY"].groupby(data["SPY"].index.date):
            if len(g) > 60:
                s = indicators(g)
                mk[day] = pd.Series(np.sign(s.close - s.vwap).to_numpy(), index=s.index)
    for sym, df in data.items():
        for day, g in df.groupby(df.index.date):
            if len(g) > 60:
                d = prep_day(g)
                if day in mk:
                    d["mkt"] = mk[day].reindex(d["ts"]).ffill().fillna(0).to_numpy()
                prep.setdefault(day, {})[sym] = d
    return prep


# =====================================================================
# 3. DAY-TRADING BACKTEST
# =====================================================================
def day_trades(sym, d, p, slip, comm):
    sig, n = signals(d, p), len(d["mins"])
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
        out.append(dict(symbol=sym, side="long" if side == 1 else "short", dir=side,
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
    setup: str = "both"            # pullback | breakout | both
    risk_per_trade: float = 0.01
    max_positions: int = 8
    max_pos_pct: float = 0.15
    stop_atr: float = 2.0          # initial stop = entry - 2 ATR
    trail_atr: float = 3.5         # chandelier trailing stop = highest close - 3.5 ATR
    target_rr: float = 0.0         # 0 = no fixed target (let winners run on the trail)
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
    F = {"sma50": C.rolling(50).mean(), "sma200": C.rolling(200).mean()}
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
    use_p, use_b = sp.setup in ("pullback", "both"), sp.setup in ("breakout", "both")
    pull_e, bo_e = (pull if use_p else pull & False), (bo if use_b else bo & False)
    sig = pull_e | bo_e
    for c in ETF_SET:
        if c in sig.columns:
            sig[c] = False
    return dict(sig=sig, bo=bo_e, pull=pull_e, base=base, hi=hi)


def regime_bool(P):
    C = P["C"]
    if "SPY" not in C:
        return pd.Series(True, index=C.index)
    spy = C["SPY"]
    return (spy > spy.rolling(200).mean()).fillna(False)


def first_valid(F):
    v = F["sma200"].notna().any(axis=1).to_numpy()
    return int(v.argmax()) + 1 if v.any() else None


def simulate_swing(P, F, sig, bo, reg, sp, i0, i1, eq0=25000.0, slip=0.0005, comm=0.0):
    """Daily event loop. Signal at close t-1 -> buy at open t. Stops/targets use that day's range
    (stop assumed to hit first). Trailing stop ratchets at each close. Gaps fill at the open."""
    names, dates = list(P["C"].columns), P["C"].index
    O, H, L, C = (P[k].to_numpy(float) for k in ("O", "H", "L", "C"))
    atr = F["atr"].to_numpy(float)
    sc = np.nan_to_num(F["score"].to_numpy(float), nan=-1.0)
    sg, bo_ = sig.to_numpy(bool), bo.to_numpy(bool)
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
                              kind="breakout" if bo_[t - 1, j] else "pullback", risk=rps * qty, eq=eq_now,
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
    return simulate_swing(P, F, S["sig"], S["bo"], regime_bool(P), sp, first_valid(F), len(P["C"]), eq0, slip, comm)


def walk_forward_swing(P, F, base, eq0=25000.0, slip=0.0005, comm=0.0, folds=3, min_trades=25):
    """Pick settings on PAST data, then trade the NEXT unseen chunk. Chunks are stitched into one
    out-of-sample result. This is the most honest number the tool produces."""
    reg, dates, n = regime_bool(P), P["C"].index, len(P["C"])
    s0 = first_valid(F)
    if s0 is None or n - s0 < 200:
        return pd.DataFrame(), None, pd.Series(dtype=float), pd.DataFrame(), "Not enough history (need ~2+ years)."
    sigs = {st: swing_signals(P, F, replace(base, setup=st)) for st in ("pullback", "breakout", "both")}
    edges = np.linspace(s0, n, folds + 2).astype(int)
    grid = list(itertools.product(("pullback", "breakout", "both"), (2.0, 3.0), (3.0, 4.5)))
    rows, oos_tr, oos_cv, eq = [], [], [], eq0
    for k in range(1, folds + 1):
        tr_end, te_end = edges[k], edges[k + 1]
        best = None
        for st, sa, ta in grid:
            sp = replace(base, setup=st, stop_atr=sa, trail_atr=ta)
            tr, cv, _ = simulate_swing(P, F, sigs[st]["sig"], sigs[st]["bo"], reg, sp, s0, tr_end, eq0, slip, comm)
            s = stats(tr, cv, eq0)
            if s["trades"] >= min_trades and (best is None or s["profit_factor"] > best[0]["profit_factor"]):
                best = (s, sp)
        if best is None:
            rows.append(dict(fold=k, note="too few trades in training window"))
            continue
        s, sp = best
        tr2, cv2, _ = simulate_swing(P, F, sigs[sp.setup]["sig"], sigs[sp.setup]["bo"], reg, sp, tr_end, te_end, eq, slip, comm)
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
def get_universe(tc):
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    assets = tc.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    out = []
    for a in assets:
        ex = str(getattr(a.exchange, "value", a.exchange))
        name = (a.name or "").lower()
        if (a.tradable and ex in ("NYSE", "NASDAQ", "AMEX", "ARCA") and a.symbol.isalpha() and len(a.symbol) <= 5
                and not any(w in name for w in BAD_WORDS)):
            out.append(a.symbol)
    return sorted(set(out))


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
    """Turn the latest daily bar into: buy candidates, 'what might go up' leaders, near-breakout watchlist."""
    C = P["C"]
    S = swing_signals(P, F, sp)
    earn = earn or {}
    today = datetime.now(NY).date()
    mkt_ok = (not sp.use_regime) or bool(regime_bool(P).iloc[-1])
    last = lambda df, s: float(df[s].iloc[-1])

    def row(s, state):
        price, a = last(C, s), last(F["atr"], s)
        stop = price - sp.stop_atr * a
        return dict(symbol=s, state=state, price=round(price, 2), stop=round(stop, 2),
                    score=int(round(last(F["score"], s) * 100)) if last(F["score"], s) == last(F["score"], s) else 0,
                    ret63=f"{last(F['ret63'], s):+.0%}", vs_SPY=f"{last(F['rs63'], s):+.0%}", adx=round(last(F["adx"], s)),
                    rsi=round(last(F["rsi"], s)), rvol=round(last(F["rvol"], s), 1),
                    from_52w_high=f"{price / last(F['hi52'], s) - 1:+.0%}"), price, stop

    cands = []
    sg = S["sig"].iloc[-1]
    for s in [x for x in sg.index[sg]]:
        r, price, stop = row(s, "BUY SIGNAL")
        is_bo = bool(S["bo"][s].iloc[-1])
        rps = price - stop
        qty = int(min(equity * sp.risk_per_trade / rps, equity * sp.max_pos_pct / price)) if rps > 0 else 0
        r.update(setup="Breakout" if is_bo else "Pullback", qty=qty, risk_usd=round(qty * rps),
                 target=round(price + sp.target_rr * rps, 2) if sp.target_rr > 0 else None,
                 buy_zone=f"{price:.2f}-{price * 1.005:.2f}",
                 why=(f"Broke {sp.breakout_n}-day high on {r['rvol']}x volume; trending (ADX {r['adx']}); beating SPY {r['vs_SPY']}"
                      if is_bo else f"Uptrend (above 50 & 200-day), pulled back to RSI {r['rsi']}, bounce candle today"))
        e = earn.get(s)
        r["earnings"] = e or ""
        if not mkt_ok:
            r["state"], r["qty"] = "WAIT: market below 200-day", 0
        elif e and sp.earnings_buffer > 0:
            dd = (pd.Timestamp(e).date() - today).days
            if 0 <= dd <= sp.earnings_buffer:
                r["state"], r["qty"] = f"SKIP: earnings in {dd}d", 0
        cands.append(r)
    cands = pd.DataFrame(cands)
    if not cands.empty:
        cands = cands.sort_values("score", ascending=False)

    base_last = S["base"].iloc[-1]
    stocks = [s for s in base_last.index[base_last] if s not in ETF_SET]
    lead = []
    for s in stocks:
        price = last(C, s)
        near = price >= 0.97 * last(S["hi"], s) and price <= last(S["hi"], s) if last(S["hi"], s) == last(S["hi"], s) else False
        ext = price > 1.3 * last(F["sma50"], s)
        state = "Extended (wait for pullback)" if ext else "Near breakout" if near else "Strong trend"
        if sg.get(s, False):
            state = "BUY SIGNAL"
        r, _, _ = row(s, state)
        r["earnings"] = earn.get(s, "")
        lead.append(r)
    lead = pd.DataFrame(lead)
    if not lead.empty:
        lead = lead.sort_values("score", ascending=False)
    watch = lead[lead.state == "Near breakout"].head(25) if not lead.empty else lead
    return cands, (lead.head(40) if not lead.empty else lead), watch


def swing_scan(dc, tc, sp, equity, max_symbols=1000, fh=None, progress=None):
    syms, n_uni = liquid_universe(dc, tc, sp, max_symbols, (lambda f: progress(f * 0.4)) if progress else None)
    syms = list(dict.fromkeys(syms + INDEX_ETFS + list(SECTORS)))
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    df = cache_get(f"stage2_{max_symbols}", 3)
    if df is None:
        df = fetch_daily(dc, syms, end - timedelta(days=520), end, batch=100,
                         progress=(lambda f: progress(0.4 + f * 0.6)) if progress else None)
        cache_set(f"stage2_{max_symbols}", df)
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
              stop_loss=StopLossRequest(stop_price=round(float(stop), 2)))
    if target:
        tc.submit_order(MarketOrderRequest(order_class=OrderClass.BRACKET,
                                           take_profit=TakeProfitRequest(limit_price=round(float(target), 2)), **kw))
    else:
        tc.submit_order(MarketOrderRequest(order_class=OrderClass.OTO, **kw))


# =====================================================================
# 6. LIVE DAY-TRADING (scanner + bot)
# =====================================================================
def get_keys():
    return os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")


def live_clients(key, secret, paper=True):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    return StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=paper)


def dynamic_watchlist(key, secret, base, top=40, min_price=5.0):
    """Base list + Alpaca's most-active stocks + biggest gainers/losers right now."""
    try:
        from alpaca.data.historical.screener import ScreenerClient
        from alpaca.data.requests import MostActivesRequest, MarketMoversRequest
        sc = ScreenerClient(key, secret)
        act = sc.get_most_actives(MostActivesRequest(top=top))
        mv = sc.get_market_movers(MarketMoversRequest(top=top))
        syms = [a.symbol for a in act.most_actives]
        syms += [m.symbol for m in list(mv.gainers) + list(mv.losers) if float(getattr(m, "price", min_price)) >= min_price]
        syms = [s for s in syms if s.isalpha() and len(s) <= 5]
        return list(dict.fromkeys(list(base) + syms))[:120], f"{len(set(syms))} dynamic symbols added (most active + movers)"
    except Exception as e:
        return list(base), f"Dynamic list unavailable ({str(e)[:80]}); using your watchlist."


def fetch_today(dc, symbols):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=datetime.now(timezone.utc) - timedelta(hours=8), feed=DataFeed.IEX)
    df = dc.get_stock_bars(req).df
    out = {}
    if df.empty:
        return out
    have = set(df.index.get_level_values(0))
    for s in symbols:
        if s in have:
            g = df.loc[s].copy()
            g.index = g.index.tz_convert(NY)
            g = g[g.index.date == datetime.now(NY).date()].between_time("09:30", "15:59")
            if len(g) >= 40:
                out[s] = g
    return out


def scan(dc, p, equity, symbols=None, avoid=None):
    """1-minute scan now. df.attrs['mkt'] = +1 (SPY above VWAP) / -1 / 0."""
    symbols = list(dict.fromkeys(symbols or p.watchlist))
    bars = fetch_today(dc, list(dict.fromkeys(symbols + ["SPY"])))
    mkt = 0
    if "SPY" in bars:
        d = prep_day(bars["SPY"])
        mkt = int(np.sign(d["close"][-1] - d["vwap"][-1]))
    rows = []
    for s in symbols:
        if s not in bars:
            continue
        d = prep_day(bars[s])
        d["mkt"] = np.full(len(d["mins"]), float(mkt))
        sig = int(signals(d, p)[-1])
        age = (datetime.now(NY) - d["ts"][-1]).total_seconds() / 60
        if age > 5 or (avoid and s in avoid):
            sig = 0
        price = float(d["close"][-1])
        row = dict(symbol=s, price=round(price, 2), vwap=round(float(d["vwap"][-1]), 2),
                   rsi=round(float(d["rsi"][-1]), 1), rvol=round(float(d["rvol"][-1]), 2),
                   age_min=round(age, 1), signal={1: "LONG", -1: "SHORT", 0: ""}[sig])
        if sig:
            stop, target = levels(price, float(d["atr"][-1]), sig, p)
            row.update(stop=round(stop, 2), target=round(target, 2), qty=size(equity, price, stop, p))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["mkt"] = mkt
    return df


def place_order(tc, sym, signal, qty, stop, target):
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    tc.submit_order(MarketOrderRequest(
        symbol=sym, qty=int(qty), side=OrderSide.BUY if signal == "LONG" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(float(target), 2)),
        stop_loss=StopLossRequest(stop_price=round(float(stop), 2))))


def live_loop(p, paper=True, poll=60, dynamic=True):
    key, secret = get_keys()
    if not (key and secret):
        sys.exit("Set ALPACA_KEY and ALPACA_SECRET first.")
    dc, tc = live_clients(key, secret, paper)
    start_eq, cool, syms, last_dyn = float(tc.get_account().last_equity), {}, list(p.watchlist), 0
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
        if dynamic and time.time() - last_dyn > 600:
            syms, note = dynamic_watchlist(key, secret, p.watchlist)
            last_dyn = time.time()
            print(note)
        held = {x.symbol for x in tc.get_all_positions()}
        if len(held) < p.max_open:
            df = scan(dc, p, eq, syms)
            if not df.empty:
                for _, r in df[df.signal != ""].iterrows():
                    if r.symbol in held or cool.get(r.symbol, datetime.min) > datetime.now() or r.qty < 1:
                        continue
                    try:
                        place_order(tc, r.symbol, r.signal, r.qty, r.stop, r.target)
                        cool[r.symbol] = datetime.now() + timedelta(minutes=p.cooldown_min)
                        held.add(r.symbol)
                        print(f"{now:%H:%M:%S} {r.signal} {int(r.qty)} {r.symbol} @~{r.price} stop {r.stop} target {r.target}")
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
### How this works (plain English)
**1. Market check first.** Most stocks rise and fall with the market. If SPY is below its 200-day average, the swing
scanner pauses new buys. ADX tells you if a trend is real (above 25) or choppy (below 20).

**2. Scan thousands of stocks.** It pulls every US stock, keeps the liquid ones (price, dollar volume), and downloads daily
bars. Nothing is hand-picked, so leaders rotate in and out automatically.

**3. Two setups, both only in uptrends** (price above 50 & 200-day averages, ADX strong):
- *Pullback*: the stock dipped (RSI 30-50) and bounced today. Buy the dip in an uptrend.
- *Breakout*: price clears its 55-day high on heavy volume while beating SPY.

**4. Rank.** Candidates are ranked by momentum (3 and 6-month strength) and volume. "What might go up" shows the strongest
trending names, including ones *near* a breakout.

**5. Risk first.** Size is set so a stop-out loses ~1% of the account. Initial stop = 2 ATR below entry (ATR = typical daily
range). The stop then trails up (3.5 ATR under the highest close) so winners can run. Time stop: 25 days.

**6. Earnings.** Add a free Finnhub key to see earnings dates; stocks reporting within 5 days are skipped (earnings gaps can blow
through any stop).

**7. Prove it.** Backtest tab: costs included, compared to just holding SPY, Monte Carlo (luck range) and a slippage stress test.
Optimizer tab: walk-forward = tune on the past, grade on unseen future. Believe that number, not the in-sample one.

### Honest limits
- No rule set works in every market. The tool shows you when it doesn't.
- Backtest universe is today's survivors (flattering). Real results are usually worse.
- Day trading on 1-minute bars has the highest cost drag; swing mode has the better odds of surviving costs.
- Paper trade for weeks first. This is not financial advice.
"""


def dashboard():
    import streamlit as st
    st.set_page_config(page_title="Trading Lab", page_icon="📈", layout="wide")
    st.title("📈 Trading Lab")
    st.caption("Paper trading only. Educational tool, not financial advice. Most traders lose money.")
    sb = st.sidebar
    sb.header("Keys")
    key = sb.text_input("Alpaca paper key", os.environ.get("ALPACA_KEY", ""), type="password")
    secret = sb.text_input("Alpaca paper secret", os.environ.get("ALPACA_SECRET", ""), type="password")
    fh = sb.text_input("Finnhub key (optional: earnings)", os.environ.get("FINNHUB_KEY", ""), type="password")
    with sb.expander("Swing settings", expanded=False):
        sp = SwingParams(
            setup=st.selectbox("Setups", ["both", "pullback", "breakout"]),
            risk_per_trade=st.slider("Risk per trade (%)", 0.25, 2.0, 1.0, 0.25, key="sr") / 100,
            max_positions=st.slider("Max positions", 1, 20, 8),
            stop_atr=st.slider("Initial stop (x ATR)", 1.0, 4.0, 2.0, 0.5),
            trail_atr=st.slider("Trailing stop (x ATR)", 2.0, 6.0, 3.5, 0.5),
            target_rr=st.slider("Fixed target (R, 0 = none)", 0.0, 6.0, 0.0, 0.5),
            max_hold=st.slider("Max hold (days)", 5, 60, 25),
            adx_min=st.slider("Min trend strength (ADX)", 10, 35, 20),
            min_price=st.slider("Min price ($)", 1.0, 50.0, 5.0, 1.0),
            min_dollar_vol=st.slider("Min avg $ volume (millions)", 1, 100, 10) * 1e6,
            use_regime=st.checkbox("Only buy when SPY > 200-day average", True),
            earnings_buffer=st.slider("Skip if earnings within (days)", 0, 14, 5))
    with sb.expander("Day-trade settings", expanded=False):
        syms = [s.strip().upper() for s in st.text_area("Base watchlist", ", ".join(DEFAULT_WATCHLIST)).split(",") if s.strip()]
        side = st.radio("Direction", ["long", "short", "both"], horizontal=True)
        p = Params(watchlist=syms,
                   risk_per_trade=st.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1, key="dr") / 100,
                   rr=st.slider("Reward : risk", 1.0, 4.0, 2.0, 0.5),
                   atr_mult=st.slider("Stop (x ATR)", 0.5, 3.0, 1.5, 0.1),
                   min_rvol=st.slider("Min relative volume", 1.0, 5.0, 1.5, 0.1),
                   daily_loss_limit=st.slider("Daily loss limit (%)", 1.0, 10.0, 3.0, 0.5) / 100,
                   max_open=st.slider("Max open trades", 1, 10, 3),
                   market_filter=st.checkbox("Market gate (SPY vs VWAP)", True),
                   allow_long=side in ("long", "both"), allow_short=side in ("short", "both"))
    with sb.expander("Backtest costs", expanded=False):
        eq0 = st.number_input("Start equity ($)", 1000.0, 10_000_000.0, 25000.0)
        slip = st.slider("Slippage per side (bps)", 0, 30, 5) / 10000
        comm = st.number_input("Commission ($/share/side)", 0.0, 0.1, 0.0, 0.001, format="%.3f")

    def clients():
        if not (key and secret):
            st.info("Enter your Alpaca PAPER keys in the sidebar (top-left arrow).")
            return None
        try:
            return live_clients(key, secret, paper=True)
        except Exception as e:
            st.error(f"Alpaca error: {e}")
            return None

    def banner(ov):
        getattr(st, ov["level"])(ov["text"])

    T = st.tabs(["🌎 Market", "🔎 Swing", "⚡ Day", "🧪 Backtest", "🎛 Optimizer", "📘 Guide"])

    # ---------------- Market ----------------
    with T[0]:
        if st.button("Refresh market view"):
            cl = clients()
            if cl:
                with st.spinner("Loading indexes and sectors..."):
                    try:
                        st.session_state["ov"] = market_light(cl[0])
                    except Exception as e:
                        st.error(f"Could not load: {e}")
        ov = st.session_state.get("ov") or (st.session_state.get("sscan") or {}).get("ov")
        if ov:
            banner(ov)
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
                st.caption(f"Breadth across {b['n']} liquid stocks. Above 60% above the 50-day = healthy market.")
        else:
            st.write("Tap **Refresh market view**, or run the Swing scan (it includes this).")
        sc = st.session_state.get("sscan")
        if sc is not None and not sc["earn"].empty:
            st.subheader("Earnings this week or next")
            st.dataframe(sc["earn"], hide_index=True)

    # ---------------- Swing ----------------
    with T[1]:
        maxsym = st.slider("How many stocks to scan (most liquid first)", 200, 3000, 1000, 100)
        st.caption("First scan downloads thousands of symbols and can take a few minutes. Later scans reuse cached data.")
        if st.button("Scan the whole market"):
            cl = clients()
            if cl:
                bar = st.progress(0.0, text="Starting...")
                try:
                    eq = float(cl[1].get_account().equity)
                    st.session_state["sscan"] = swing_scan(cl[0], cl[1], sp, eq, maxsym, fh or None,
                                                           lambda f: bar.progress(min(float(f), 1.0), text="Downloading market data..."))
                    bar.empty()
                except Exception as e:
                    bar.empty()
                    st.error(f"Scan failed: {e}")
        sc = st.session_state.get("sscan")
        if sc:
            banner(sc["ov"])
            st.caption(f"As of {sc['asof']} close | {sc['n']} stocks scanned from {sc['n_uni']} US listings | data: {sc['feed']}")
            a, b, c, d = st.tabs(["Buy candidates", "What might go up", "Near breakout", "My positions"])
            with a:
                cands = sc["cands"]
                if cands.empty:
                    st.write("No buy signals today. That is normal. Waiting is a position.")
                else:
                    st.dataframe(cands, hide_index=True)
                    cl = clients()
                    ok = cands[cands.state == "BUY SIGNAL"]
                    for _, r in ok.head(10).iterrows():
                        if cl and st.button(f"Paper BUY {int(r.qty)} {r.symbol} (stop {r.stop})", key=f"sb_{r.symbol}"):
                            try:
                                place_swing_order(cl[1], r.symbol, r.qty, r.stop, r.get("target"))
                                st.success(f"Paper order sent for {r.symbol} (GTC, stop attached).")
                            except Exception as e:
                                st.error(f"Order failed: {e}")
                    pick = st.selectbox("Headlines for", [""] + list(cands.symbol))
                    if pick and key and secret:
                        for ts, h, src in fetch_news(key, secret, [pick]):
                            st.write(f"- {h} _({src})_")
            with b:
                st.caption("Strongest trending stocks right now, ranked by momentum + relative strength. Not a prediction, a ranking.")
                st.dataframe(sc["leaders"], hide_index=True)
            with c:
                st.caption("Trending stocks within 3% of a breakout. Set an alert and wait for the volume.")
                st.dataframe(sc["watch"], hide_index=True)
            with d:
                if st.button("Load my paper positions"):
                    cl = clients()
                    if cl:
                        try:
                            st.session_state["pos"] = swing_positions(cl[1], cl[0], sp)
                        except Exception as e:
                            st.error(f"Could not load positions: {e}")
                pos = st.session_state.get("pos")
                if pos is not None:
                    st.dataframe(pos, hide_index=True) if not pos.empty else st.write("No open positions.")

    # ---------------- Day ----------------
    with T[2]:
        now = datetime.now(NY)
        flat = now.replace(hour=p.flatten_at[0], minute=p.flatten_at[1], second=0, microsecond=0)
        in_window = now.weekday() < 5 and p.start_after <= (now.hour, now.minute) < p.flatten_at
        c1, c2, c3 = st.columns(3)
        c1.metric("New York time", now.strftime("%H:%M:%S"))
        c2.metric("Minutes until flatten", f"{max((flat - now).total_seconds() / 60, 0):.0f}")
        c3.metric("Trading window", "OPEN" if in_window else "CLOSED")
        dyn = st.checkbox("Scan beyond my list: most-active + biggest movers", True)
        auto = st.checkbox("Auto-refresh every 60s", False)
        cl = clients()
        if cl:
            try:
                eq = float(cl[1].get_account().equity)
                st.write(f"Paper equity: **${eq:,.2f}**")
                if st.button("Scan now") or auto:
                    lst, note = dynamic_watchlist(key, secret, p.watchlist) if dyn else (p.watchlist, "")
                    avoid = set()
                    if fh:
                        e = earnings_calendar(fh, 2)
                        avoid = set(e.symbol) if not e.empty else set()
                    st.session_state["scan"] = scan(cl[0], p, eq, lst, avoid)
                    st.session_state["scan_note"] = note + (f" | skipping {len(avoid)} earnings names" if avoid else "")
                sc = st.session_state.get("scan")
                if sc is not None:
                    m = sc.attrs.get("mkt", 0)
                    st.info({1: "Market gate: SPY above VWAP (longs allowed)", -1: "Market gate: SPY below VWAP (shorts only)",
                             0: "Market gate: neutral/unknown"}[m])
                    st.caption(st.session_state.get("scan_note", ""))
                    if sc.empty:
                        st.write("No data (market closed or no bars yet).")
                    else:
                        st.dataframe(sc, hide_index=True)
                        for _, r in sc[sc.signal != ""].iterrows():
                            if st.button(f"Paper {r.signal} {int(r.qty)} {r.symbol} (stop {r.stop}, target {r.target})", key=f"o_{r.symbol}"):
                                place_order(cl[1], r.symbol, r.signal, r.qty, r.stop, r.target)
                                st.success(f"Paper bracket order sent for {r.symbol}")
                if auto:
                    time.sleep(60)
                    st.rerun()
            except Exception as e:
                st.error(f"Alpaca error: {e}")

    # ---------------- Backtest ----------------
    with T[3]:
        eng = st.radio("Engine", ["Swing (daily bars)", "Day trading (1-minute)"], horizontal=True)
        src = st.radio("Data", ["Synthetic (code test only)", "Alpaca history"] + (["Upload CSV"] if eng.startswith("Day") else []), horizontal=True)
        c1, c2, c3 = st.columns(3)
        years = 3 if eng.startswith("Swing") else 1
        start = c1.date_input("Start", datetime.now().date() - timedelta(days=365 * years if eng.startswith("Swing") else 60))
        end = c2.date_input("End", datetime.now().date())
        nsym = c3.slider("Stocks (swing)", 100, 1500, 300, 50) if eng.startswith("Swing") else 0
        up = st.file_uploader("CSV (timestamp,symbol,open,high,low,close,volume)", type="csv") if src == "Upload CSV" else None
        if st.button("Load data"):
            try:
                bar = st.progress(0.0, text="Loading...")
                if eng.startswith("Swing"):
                    P, F, note = load_swing_data("synthetic" if src.startswith("Synthetic") else "alpaca", start, end, nsym, key or None,
                                                 secret or None, lambda f: bar.progress(min(float(f), 1.0), text="Downloading..."))
                    st.session_state["swing"] = dict(P=P, F=F, note=note)
                else:
                    data = (synthetic(syms) if src.startswith("Synthetic") else
                            load_alpaca(syms, str(start), str(end), key, secret) if src == "Alpaca history" else load_csv(up))
                    st.session_state["prep"] = prep_data(data)
                    st.session_state["day_note"] = src
                bar.empty()
            except Exception as e:
                st.error(f"Could not load data: {e}")
        if eng.startswith("Swing") and st.session_state.get("swing"):
            sw = st.session_state["swing"]
            st.caption(sw["note"])
            if st.button("Run swing backtest"):
                with st.spinner("Simulating..."):
                    tr, cv, expo = run_swing(sw["P"], sw["F"], sp, eq0, slip, comm)
                    spy = sw["P"]["C"]["SPY"] if "SPY" in sw["P"]["C"] else None
                    s, bm = stats(tr, cv, eq0), bench_stats(spy, cv)
                    mc = monte_carlo(tr)
                    stress = stress_table(lambda sl: run_swing(sw["P"], sw["F"], sp, eq0, sl, comm)[:2], max(slip, 0.0005), eq0)
                m = st.columns(4)
                m[0].metric("Return", f"{s['ret']:+.1%}", f"SPY {bm['ret']:+.1%}" if bm else None)
                m[1].metric("Max drawdown", f"{s['max_dd']:.1%}", f"SPY {bm['dd']:.1%}" if bm else None, delta_color="off")
                m[2].metric("Profit factor", "-" if not s["trades"] else f"{s['profit_factor']:.2f}")
                m[3].metric("Trades", s["trades"])
                m = st.columns(4)
                m[0].metric("CAGR", f"{s['cagr']:+.1%}")
                m[1].metric("Win rate", "-" if not s["trades"] else f"{s['win_rate']:.0%}")
                m[2].metric("Avg R", "-" if not s["trades"] else f"{s['avg_R']:.2f}")
                m[3].metric("Time invested", f"{expo:.0%}")
                if s["trades"]:
                    chart = pd.DataFrame({"Strategy": cv / eq0})
                    if bm is not None:
                        chart["SPY buy & hold"] = bm["curve"]
                    st.line_chart(chart)
                    st.area_chart((cv / cv.cummax() - 1).rename("Drawdown"))
                    st.write("**Returns by year**")
                    st.dataframe(yearly_returns(cv, eq0).map(lambda x: f"{x:+.1%}").rename("return").to_frame())
                    st.write("**By setup**")
                    st.dataframe(tr.groupby("kind").agg(trades=("pnl", "count"), win_rate=("pnl", lambda x: (x > 0).mean()),
                                                         avg_R=("R", "mean"), total_pnl=("pnl", "sum")).round(2))
                    st.write("**Slippage stress test** (does it survive worse fills?)")
                    st.dataframe(stress, hide_index=True)
                    if mc:
                        st.write("**Monte Carlo** (range of outcomes from luck alone, same trades reshuffled)")
                        st.write(f"Typical return {mc['ret_p50']:+.0%} (bad luck {mc['ret_p5']:+.0%}, good luck {mc['ret_p95']:+.0%}). "
                                 f"Typical worst drawdown {mc['dd_p50']:.0%}, unlucky {mc['dd_p95']:.0%}. "
                                 f"Chance of losing money: {mc['prob_loss']:.0%}.")
                    if s["trades"] < 200:
                        st.warning("Under ~200 trades: treat as anecdotal.")
                    st.dataframe(tr)
                    st.download_button("Download trades CSV", tr.to_csv(index=False), "swing_trades.csv")
                else:
                    st.write("No trades. Loosen the filters or load more data.")
        if eng.startswith("Day") and st.session_state.get("prep"):
            prep = st.session_state["prep"]
            st.caption(f"{len(prep)} trading days loaded ({st.session_state.get('day_note')})")
            if st.button("Run day-trading backtest"):
                tr, cv = run(prep, p, eq0, slip, comm)
                s = stats(tr, cv, eq0)
                m = st.columns(5)
                m[0].metric("Trades", s["trades"])
                m[1].metric("Win rate", "-" if not s["trades"] else f"{s['win_rate']:.0%}")
                m[2].metric("Profit factor", "-" if not s["trades"] else f"{s['profit_factor']:.2f}")
                m[3].metric("Return", f"{s['ret']:+.1%}")
                m[4].metric("Max drawdown", f"{s['max_dd']:.1%}")
                if s["trades"]:
                    st.line_chart(cv)
                    st.write("**Slippage stress test**")
                    st.dataframe(stress_table(lambda sl: run(prep, p, eq0, sl, comm), max(slip, 0.0005), eq0), hide_index=True)
                    mc = monte_carlo(tr)
                    if mc:
                        st.write(f"**Monte Carlo:** typical {mc['ret_p50']:+.0%}, bad luck {mc['ret_p5']:+.0%}, "
                                 f"chance of losing money {mc['prob_loss']:.0%}.")
                    st.dataframe(tr)
                    st.download_button("Download trades CSV", tr.to_csv(index=False), "trades.csv")

    # ---------------- Optimizer ----------------
    with T[4]:
        st.subheader("Swing: walk-forward test")
        st.caption("Trains on the past, trades the next unseen stretch, repeats, then stitches the unseen results together.")
        if st.session_state.get("swing"):
            sw = st.session_state["swing"]
            folds = st.slider("Folds", 2, 5, 3)
            if st.button("Run walk-forward"):
                with st.spinner("Testing 12 settings per fold..."):
                    rows, s_all, cv_all, tr_all, verdict = walk_forward_swing(sw["P"], sw["F"], sp, eq0, slip, comm, folds)
                st.dataframe(rows, hide_index=True)
                if s_all:
                    st.write(f"**Stitched out-of-sample:** return {s_all['ret']:+.1%}, profit factor {s_all['profit_factor']:.2f}, "
                             f"max drawdown {s_all['max_dd']:.1%}, trades {s_all['trades']}")
                    spy = sw["P"]["C"]["SPY"] if "SPY" in sw["P"]["C"] else None
                    bm = bench_stats(spy, cv_all)
                    if bm:
                        st.write(f"SPY buy & hold over the same period: {bm['ret']:+.1%} (drawdown {bm['dd']:.1%})")
                    st.line_chart(cv_all / eq0)
                (st.success if verdict.startswith("Some") else st.error)(verdict)
        else:
            st.info("Load swing data in the Backtest tab first.")
        st.subheader("Day trading: train/test sweep")
        if st.session_state.get("prep"):
            frac = st.slider("Share of days used for tuning", 0.5, 0.9, 0.7, 0.05)
            if st.button("Run day sweep"):
                with st.spinner("Testing 27 combinations..."):
                    res, verdict = sweep(st.session_state["prep"], p, eq0, slip, comm, frac)
                if not res.empty:
                    st.dataframe(res, hide_index=True)
                (st.success if verdict.startswith("Some") else st.error)(verdict)
        else:
            st.info("Load day-trading data in the Backtest tab first.")

    with T[5]:
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
        sp.add_argument("--slippage", type=float, default=0.0005, help="per side, fraction (0.0005 = 5 bps)")
        sp.add_argument("--commission", type=float, default=0.0, help="$ per share per side")

    def day_common(sp):
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--csv")
        sp.add_argument("--start")
        sp.add_argument("--end", default=str(datetime.now().date()))
        sp.add_argument("--symbols", nargs="*")
        sp.add_argument("--side", choices=["long", "short", "both"], default="long")
        money(sp)
        for n in ("rr", "atr", "rvol", "risk"):
            sp.add_argument(f"--{n}", type=float)

    def swing_common(sp):
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--start", default=str((datetime.now() - timedelta(days=3 * 365)).date()))
        sp.add_argument("--end", default=str(datetime.now().date()))
        sp.add_argument("--max-symbols", type=int, default=300)
        sp.add_argument("--setup", choices=["both", "pullback", "breakout"], default="both")
        sp.add_argument("--stop-atr", type=float)
        sp.add_argument("--trail-atr", type=float)
        sp.add_argument("--target-rr", type=float)
        sp.add_argument("--no-regime", action="store_true")
        money(sp)

    day_common(sub.add_parser("backtest", help="1-minute day-trading backtest"))
    sw = sub.add_parser("sweep", help="day-trading train/test sweep"); day_common(sw)
    sw.add_argument("--train-frac", type=float, default=0.7)
    sw.add_argument("--min-trades", type=int, default=30)
    swing_common(sub.add_parser("swing-backtest", help="daily-bar swing backtest across many stocks"))
    swing_common(sub.add_parser("swing-walkforward", help="swing walk-forward (honest out-of-sample test)"))
    sub.add_parser("swing-scan", help="scan the whole market for swing setups now")
    lv = sub.add_parser("live", help="run the paper day-trading bot")
    lv.add_argument("--side", choices=["long", "short", "both"], default="long")
    lv.add_argument("--symbols", nargs="*")
    lv.add_argument("--no-dynamic", action="store_true")
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
        live_loop(build_params(a), paper, dynamic=not a.no_dynamic)
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
        sp = SwingParams()
        res = swing_scan(dc, tc, sp, float(tc.get_account().equity), 1000, os.environ.get("FINNHUB_KEY"),
                         lambda f: print(f"\rdownloading {f:.0%}", end=""))
        print("\n" + res["ov"]["text"])
        print(f"\nBuy candidates ({res['asof']}):")
        print(res["cands"].to_string(index=False) if not res["cands"].empty else "none today")
        print("\nTop momentum leaders:")
        print(res["leaders"].head(15).to_string(index=False))
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
