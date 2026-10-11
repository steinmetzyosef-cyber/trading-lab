#!/usr/bin/env python3
"""
DAY TRADING LAB  (single file)  -  paper trading by default. Not financial advice.
Most day traders lose money; prove an edge in backtests + weeks of paper trading first.

SETUP (once)
  pip install alpaca-py pandas numpy matplotlib streamlit
  Free paper account at alpaca.markets, then:
    export ALPACA_KEY=your_paper_key
    export ALPACA_SECRET=your_paper_secret       (or type them into the dashboard sidebar)

USE
  python day_trading_lab.py dashboard                  # web dashboard (scanner, backtest, optimizer)
  python day_trading_lab.py backtest --synthetic       # code test on random data (meaningless results)
  python day_trading_lab.py backtest --start 2026-01-02 --side both
  python day_trading_lab.py backtest --csv bars.csv    # columns: timestamp,symbol,open,high,low,close,volume
  python day_trading_lab.py sweep --start 2026-01-02   # optimizer with out-of-sample test
  python day_trading_lab.py live                       # paper-trading bot (Ctrl+C to stop)

STRATEGY (1-minute bars, one source of truth for backtest AND live)
  LONG : close > VWAP, close > prior 20-min high, rel-volume >= min, RSI in 55-75
  SHORT: close < VWAP, close < prior 20-min low,  rel-volume >= min, RSI in 25-45 (needs margin + shortable stock)
  Exit : bracket order. Stop = atr_mult x ATR, target = rr x risk. Everything flat at 3:50pm ET.
  Risk : 1% of equity per trade, max 3 open, 15-min cooldown per symbol, stop for the day at -3%.
"""
import os, sys, time, argparse, itertools, subprocess
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL", "SPY", "QQQ"]


# =====================================================================
# 1. STRATEGY
# =====================================================================
@dataclass
class Params:
    watchlist: list = field(default_factory=lambda: list(DEFAULT_WATCHLIST))
    risk_per_trade: float = 0.01
    daily_loss_limit: float = 0.03
    rr: float = 2.0                 # reward : risk
    atr_mult: float = 1.5           # stop distance in ATRs
    min_rvol: float = 1.5           # volume vs 30-min average
    rsi_long: tuple = (55, 75)
    rsi_short: tuple = (25, 45)
    max_open: int = 3
    max_pos_pct: float = 0.25       # cap one position at 25% of equity
    cooldown_min: int = 15
    start_after: tuple = (9, 45)    # skip the chaotic open
    flatten_at: tuple = (15, 50)
    allow_long: bool = True
    allow_short: bool = False


def indicators(df):
    """df = ONE trading day of 1-min bars. Uses past data only (no lookahead)."""
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
    """Indicators -> dict of numpy arrays (fast to loop over)."""
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
        ok = (m >= lo) & (m < hi) & (d["atr"] > 0) & (d["rvol"] >= p.min_rvol)
        if p.allow_long:
            sig[ok & (d["close"] > d["vwap"]) & (d["close"] > d["hi20"]) &
                (d["rsi"] >= p.rsi_long[0]) & (d["rsi"] <= p.rsi_long[1])] = 1
        if p.allow_short:
            sig[ok & (d["close"] < d["vwap"]) & (d["close"] < d["lo20"]) &
                (d["rsi"] >= p.rsi_short[0]) & (d["rsi"] <= p.rsi_short[1])] = -1
    return sig


def levels(entry, atr, side, p):
    risk = p.atr_mult * atr
    return (entry - side * risk, entry + side * p.rr * risk)   # (stop, target)


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
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
                           start=pd.Timestamp(start, tz=NY),
                           end=pd.Timestamp(end, tz=NY) + pd.Timedelta(days=1), feed=DataFeed.IEX)
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
    """Random-walk data to test the code. Results say NOTHING about profitability."""
    rng, out = np.random.default_rng(seed), {}
    bdays = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=days)
    for s in symbols:
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
    """{symbol: bars} -> {day: {symbol: arrays}} (indicators computed once, reused by sweeps)."""
    prep = {}
    for sym, df in data.items():
        for day, g in df.groupby(df.index.date):
            if len(g) > 60:
                prep.setdefault(day, {})[sym] = prep_day(g)
    return prep


# =====================================================================
# 3. BACKTESTER
# Realism: signal on bar close -> fill at NEXT bar open (+slippage). If stop and target
# both fall inside one bar, the STOP is assumed to hit first. Gaps fill at the open.
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
            open_.append(t); taken.append(t)
        equity += sum(x["pnl"] for x in taken)
        accepted += taken
        curve[pd.Timestamp(day)] = equity
    return pd.DataFrame(accepted), pd.Series(curve, dtype=float)


def stats(tr, cv, eq0):
    nan = float("nan")
    if tr.empty or cv.empty:
        return dict(days=len(cv), trades=0, win_rate=nan, avg_R=nan, profit_factor=nan,
                    ret=0.0, max_dd=0.0, sharpe=nan, end_equity=eq0)
    w, l = tr.loc[tr.pnl > 0, "pnl"].sum(), -tr.loc[tr.pnl <= 0, "pnl"].sum()
    full = pd.concat([pd.Series([eq0]), cv.reset_index(drop=True)])
    daily = full.pct_change().dropna()
    return dict(days=len(cv), trades=len(tr), win_rate=(tr.pnl > 0).mean(), avg_R=tr.R.mean(),
                profit_factor=w / l if l else float("inf"), ret=cv.iloc[-1] / eq0 - 1,
                max_dd=(full / full.cummax() - 1).min(), end_equity=cv.iloc[-1],
                sharpe=daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else nan)


def print_stats(s, tr, eq0):
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Days tested     : {s['days']}")
    print(f"Trades          : {s['trades']}")
    if s["trades"]:
        print(f"Win rate        : {s['win_rate']:.1%}")
        print(f"Avg R / trade   : {s['avg_R']:.2f}")
        print(f"Profit factor   : {s['profit_factor']:.2f}")
        print(f"Equity          : ${eq0:,.0f} -> ${s['end_equity']:,.0f} ({s['ret']:+.1%})")
        print(f"Max drawdown    : {s['max_dd']:.1%}")
        print(f"Sharpe (daily)  : {s['sharpe']:.2f}")
        print("Exits           :", tr.reason.value_counts().to_dict())
        print("Longs/Shorts    :", tr.side.value_counts().to_dict())
    print("======================================")
    print("Trust nothing under ~200 trades, nothing tuned and tested on the same data,")
    print("and nothing that dies when slippage doubles.")


# =====================================================================
# 4. OPTIMIZER (train on early days, TEST on later unseen days)
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
                    "train_avgR": round(s["avg_R"], 2), "test_trades": s2["trades"],
                    "test_PF": round(s2["profit_factor"], 2) if s2["trades"] else np.nan,
                    "test_avgR": round(s2["avg_R"], 2) if s2["trades"] else np.nan,
                    "test_return": f"{s2['ret']:+.1%}"})
    df = pd.DataFrame(out)
    good = (df.test_PF > 1.1) & (df.test_trades >= 15)
    verdict = ("Some settings held up on unseen data; still re-check with higher slippage and more history."
               if good.any() else
               "NO robust edge: the best in-sample settings did not hold up on unseen data. "
               "That is the normal outcome; do not trade this live.")
    return df, verdict


# =====================================================================
# 5. LIVE / PAPER TRADING
# =====================================================================
def get_keys():
    return os.environ.get("ALPACA_KEY"), os.environ.get("ALPACA_SECRET")


def live_clients(key, secret, paper=True):
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    return StockHistoricalDataClient(key, secret), TradingClient(key, secret, paper=paper)


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
    for s in symbols:
        if s in df.index.get_level_values(0):
            g = df.loc[s].copy()
            g.index = g.index.tz_convert(NY)
            g = g[g.index.date == datetime.now(NY).date()].between_time("09:30", "15:59")
            if len(g) >= 40:
                out[s] = g
    return out


def scan(dc, p, equity):
    """Scan the watchlist now. Returns a table incl. suggested stop/target/qty for live signals."""
    rows = []
    for s, g in fetch_today(dc, p.watchlist).items():
        d = prep_day(g)
        sig = int(signals(d, p)[-1])
        age = (datetime.now(NY) - d["ts"][-1]).total_seconds() / 60
        if age > 5:
            sig = 0   # stale data (illiquid on IEX feed or market closed)
        price = float(d["close"][-1])
        row = dict(symbol=s, price=round(price, 2), vwap=round(float(d["vwap"][-1]), 2),
                   rsi=round(float(d["rsi"][-1]), 1), rvol=round(float(d["rvol"][-1]), 2),
                   age_min=round(age, 1), signal={1: "LONG", -1: "SHORT", 0: ""}[sig])
        if sig:
            stop, target = levels(price, float(d["atr"][-1]), sig, p)
            row.update(stop=round(stop, 2), target=round(target, 2), qty=size(equity, price, stop, p))
        rows.append(row)
    return pd.DataFrame(rows)


def place_order(tc, sym, signal, qty, stop, target):
    from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
    tc.submit_order(MarketOrderRequest(
        symbol=sym, qty=int(qty), side=OrderSide.BUY if signal == "LONG" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(float(target), 2)),
        stop_loss=StopLossRequest(stop_price=round(float(stop), 2))))


def live_loop(p, paper=True, poll=60):
    key, secret = get_keys()
    if not (key and secret):
        sys.exit("Set ALPACA_KEY and ALPACA_SECRET first.")
    dc, tc = live_clients(key, secret, paper)
    start_eq, cool = float(tc.get_account().last_equity), {}
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
        held = {x.symbol for x in tc.get_all_positions()}
        if len(held) < p.max_open:
            df = scan(dc, p, eq)
            if not df.empty:
                for _, r in df[df.signal != ""].iterrows():
                    if r.symbol in held or cool.get(r.symbol, datetime.min) > datetime.now() or r.qty < 1:
                        continue
                    try:
                        place_order(tc, r.symbol, r.signal, r.qty, r.stop, r.target)
                        cool[r.symbol] = datetime.now() + timedelta(minutes=p.cooldown_min)
                        held.add(r.symbol)
                        print(f"{now:%H:%M:%S} {r.signal} {int(r.qty)} {r.symbol} @~{r.price} "
                              f"stop {r.stop} target {r.target}")
                    except Exception as e:
                        print("Order failed:", e)
                    if len(held) >= p.max_open:
                        break
        time.sleep(poll)


# =====================================================================
# 6. STREAMLIT DASHBOARD   (python day_trading_lab.py dashboard)
# =====================================================================
def dashboard():
    import streamlit as st
    st.set_page_config(page_title="Day Trading Lab", layout="wide")
    st.title("📈 Day Trading Lab")
    st.caption("Paper trading only. Educational tool, not financial advice. Most day traders lose money.")
    sb = st.sidebar
    sb.header("Alpaca paper keys")
    key = sb.text_input("API key", os.environ.get("ALPACA_KEY", ""), type="password")
    secret = sb.text_input("API secret", os.environ.get("ALPACA_SECRET", ""), type="password")
    sb.header("Strategy")
    syms = [s.strip().upper() for s in
            sb.text_area("Watchlist (comma separated)", ", ".join(DEFAULT_WATCHLIST)).split(",") if s.strip()]
    side = sb.radio("Direction", ["long", "short", "both"], horizontal=True)
    p = Params(watchlist=syms,
               risk_per_trade=sb.slider("Risk per trade (%)", 0.1, 3.0, 1.0, 0.1) / 100,
               rr=sb.slider("Reward : risk", 1.0, 4.0, 2.0, 0.5),
               atr_mult=sb.slider("Stop distance (x ATR)", 0.5, 3.0, 1.5, 0.1),
               min_rvol=sb.slider("Min relative volume", 1.0, 5.0, 1.5, 0.1),
               daily_loss_limit=sb.slider("Daily loss limit (%)", 1.0, 10.0, 3.0, 0.5) / 100,
               max_open=sb.slider("Max open trades", 1, 10, 3),
               cooldown_min=sb.slider("Cooldown per symbol (min)", 0, 60, 15),
               allow_long=side in ("long", "both"), allow_short=side in ("short", "both"))
    t1, t2, t3 = st.tabs(["Live scanner", "Backtest", "Optimizer"])

    # ---- tab 1: live scanner + timers ----
    with t1:
        now = datetime.now(NY)
        flat = now.replace(hour=p.flatten_at[0], minute=p.flatten_at[1], second=0, microsecond=0)
        in_window = now.weekday() < 5 and p.start_after <= (now.hour, now.minute) < p.flatten_at
        c1, c2, c3 = st.columns(3)
        c1.metric("New York time", now.strftime("%H:%M:%S"))
        c2.metric("Minutes until flatten", f"{max((flat - now).total_seconds() / 60, 0):.0f}")
        c3.metric("Trading window", "OPEN" if in_window else "CLOSED")
        if not (key and secret):
            st.info("Enter your Alpaca PAPER keys in the sidebar.")
        else:
            try:
                dc, tc = live_clients(key, secret, paper=True)
                eq = float(tc.get_account().equity)
                st.write(f"Paper account equity: **${eq:,.2f}**")
                auto = st.checkbox("Auto-refresh every 60s")
                if st.button("Scan now") or auto:
                    st.session_state["scan"] = scan(dc, p, eq)
                sc = st.session_state.get("scan")
                if sc is not None and not sc.empty:
                    st.dataframe(sc, use_container_width=True)
                    for _, r in sc[sc.signal != ""].iterrows():
                        if st.button(f"Paper {r.signal} {int(r.qty)} {r.symbol} "
                                     f"(stop {r.stop}, target {r.target})", key=f"o_{r.symbol}"):
                            place_order(tc, r.symbol, r.signal, r.qty, r.stop, r.target)
                            st.success(f"Paper bracket order sent for {r.symbol}")
                elif sc is not None:
                    st.write("No data returned (market closed or no bars yet).")
                if auto:
                    time.sleep(60)
                    st.rerun()
            except Exception as e:
                st.error(f"Alpaca error: {e}")

    # ---- tab 2: backtest ----
    with t2:
        src = st.radio("Data", ["Synthetic (code test only)", "Alpaca history", "Upload CSV"], horizontal=True)
        c1, c2, c3 = st.columns(3)
        start = c1.date_input("Start", datetime.now().date() - timedelta(days=120))
        end = c2.date_input("End", datetime.now().date())
        eq0 = c3.number_input("Start equity ($)", 1000.0, 10_000_000.0, 25000.0)
        slip = st.slider("Slippage per side (bps)", 0, 30, 5) / 10000
        comm = st.number_input("Commission ($/share/side)", 0.0, 0.1, 0.0, 0.001, format="%.3f")
        up = st.file_uploader("CSV (timestamp,symbol,open,high,low,close,volume)", type="csv") \
            if src == "Upload CSV" else None
        if st.button("Load data"):
            try:
                with st.spinner("Loading..."):
                    data = (synthetic(syms) if src.startswith("Synthetic") else
                            load_alpaca(syms, str(start), str(end), key, secret) if src == "Alpaca history"
                            else load_csv(up))
                    st.session_state["prep"], st.session_state["src"] = prep_data(data), src
            except Exception as e:
                st.error(f"Could not load data: {e}")
        prep = st.session_state.get("prep")
        if prep:
            st.caption(f"{len(prep)} trading days loaded ({st.session_state['src']})")
            if st.session_state["src"].startswith("Synthetic"):
                st.warning("Synthetic data: tests the code only, says nothing about profitability.")
            if st.button("Run backtest"):
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
                    st.dataframe(tr, use_container_width=True)
                    st.download_button("Download trades CSV", tr.to_csv(index=False), "trades.csv")
                    if s["trades"] < 200:
                        st.warning("Under ~200 trades: too few to trust.")
                else:
                    st.write("No trades. Loosen the filters or load more data.")

    # ---- tab 3: optimizer ----
    with t3:
        prep = st.session_state.get("prep")
        if not prep:
            st.info("Load data in the Backtest tab first.")
        else:
            frac = st.slider("Share of days used for tuning (rest = unseen test)", 0.5, 0.9, 0.7, 0.05)
            mint = st.number_input("Min trades in tuning period", 5, 500, 30)
            if st.button("Run optimizer"):
                with st.spinner("Testing 27 setting combinations..."):
                    res, verdict = sweep(prep, p, eq0, slip, comm, frac, int(mint))
                if not res.empty:
                    st.dataframe(res, use_container_width=True)
                (st.success if verdict.startswith("Some") else st.error)(verdict)


# =====================================================================
# 7. COMMAND LINE
# =====================================================================
def build_params(a):
    p = Params()
    if a.symbols: p.watchlist = a.symbols
    for arg, attr in [("rr", "rr"), ("atr", "atr_mult"), ("rvol", "min_rvol"), ("risk", "risk_per_trade")]:
        if getattr(a, arg) is not None:
            setattr(p, attr, getattr(a, arg))
    p.allow_long, p.allow_short = a.side in ("long", "both"), a.side in ("short", "both")
    return p


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

    def common(sp):
        sp.add_argument("--synthetic", action="store_true")
        sp.add_argument("--csv")
        sp.add_argument("--start")
        sp.add_argument("--end", default=str(datetime.now().date()))
        sp.add_argument("--symbols", nargs="*")
        sp.add_argument("--side", choices=["long", "short", "both"], default="long")
        sp.add_argument("--equity", type=float, default=25000)
        sp.add_argument("--slippage", type=float, default=0.0005, help="per side, fraction (0.0005 = 5 bps)")
        sp.add_argument("--commission", type=float, default=0.0, help="$ per share per side")
        for n in ("rr", "atr", "rvol", "risk"):
            sp.add_argument(f"--{n}", type=float)

    common(sub.add_parser("backtest", help="replay history through the strategy"))
    sw = sub.add_parser("sweep", help="optimizer with out-of-sample test"); common(sw)
    sw.add_argument("--train-frac", type=float, default=0.7)
    sw.add_argument("--min-trades", type=int, default=30)
    lv = sub.add_parser("live", help="run the paper-trading bot")
    lv.add_argument("--side", choices=["long", "short", "both"], default="long")
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
    elif a.cmd in ("backtest", "sweep"):
        p = build_params(a)
        prep = prep_data(get_data(a, p))
        if a.cmd == "backtest":
            tr, cv = run(prep, p, a.equity, a.slippage, a.commission)
            print_stats(stats(tr, cv, a.equity), tr, a.equity)
            if not tr.empty:
                tr.to_csv("backtest_trades.csv", index=False)
                cv.to_csv("backtest_equity.csv", header=["equity"])
                try:
                    import matplotlib; matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    cv.plot(title="Equity curve"); plt.tight_layout(); plt.savefig("backtest_equity.png", dpi=120)
                except ImportError:
                    pass
                print("Saved backtest_trades.csv, backtest_equity.csv, backtest_equity.png")
        else:
            df, verdict = sweep(prep, p, a.equity, a.slippage, a.commission, a.train_frac, a.min_trades)
            if not df.empty:
                print(df.to_string(index=False))
            print("\n" + verdict)
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
