import numpy as np
import pandas as pd
import yfinance as yf
import os
from pathlib import Path
from datetime import datetime

import stocks_q_learning as sql

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

# ----------------------------
# Vectorized backtest helpers
# ----------------------------
def backtest_stats(name: str, index: pd.Index, equity: np.ndarray, positions: np.ndarray, trades: int, max_position: int):
    # Be permissive about shapes coming from pandas/numpy; flatten (n,1) etc to (n,).
    equity = np.asarray(equity, dtype=np.float64).squeeze()
    positions = np.asarray(positions, dtype=np.float64).squeeze()
    equity = np.ravel(equity)
    positions = np.ravel(positions)
    if equity.size != index.size or positions.size != index.size:
        raise ValueError(f"length mismatch: equity={equity.size}, positions={positions.size}, index={index.size}")

    rets = equity[1:] / equity[:-1] - 1.0
    sharpe = sql._nan_safe_sharpe(rets)
    cummax = np.maximum.accumulate(equity)
    dd = equity / cummax - 1.0
    max_dd = float(np.min(dd)) if dd.size else 0.0
    steps = max(1, equity.size - 1)
    turnover = float(trades) / float(steps)
    avg_pos = float(np.mean(positions))
    exposure = avg_pos / float(max_position) if max_position > 0 else 0.0

    # CAGR (approx using calendar time)
    days = (index[-1] - index[0]).days if hasattr(index[-1], "to_pydatetime") else None
    years = (days / 365.25) if days and days > 0 else None
    total_return = float(equity[-1]) / float(equity[0]) if float(equity[0]) != 0.0 else None
    cagr = (float(total_return) ** (1.0 / years) - 1.0) if (years and total_return is not None) else None

    pos_counts = {int(p): int(np.sum(positions == p)) for p in sorted(set(map(int, positions)))}
    pos_frac = {k: v / float(positions.size) for k, v in pos_counts.items()}

    print(f"\n== {name} ==")
    print(f"period: {index[0].date()} -> {index[-1].date()}  (n={index.size})")
    print(f"final_equity: {equity[-1]:.4f}  sharpe: {sharpe:.3f}  max_dd: {max_dd:.3%}")
    if cagr is not None:
        print(f"cagr: {cagr:.2%}")
    print(f"trades: {trades}  turnover(trades/step): {turnover:.4f}")
    print(f"avg_position: {avg_pos:.3f} / {max_position}  exposure(norm): {exposure:.3f}")
    print("position_frac:", ", ".join([f"{k}:{pos_frac[k]:.2%}" for k in sorted(pos_frac)]))

    return {
        "final_equity": float(equity[-1]),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "turnover": float(turnover),
        "avg_position": float(avg_pos),
        "exposure": float(exposure),
        "cagr": float(cagr) if cagr is not None else None,
    }

# ----------------------------
# Load data
# ----------------------------
csv_path = os.environ.get("SPY_CSV")
if csv_path:
    df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
else:
    _cache_dir = (Path(__file__).resolve().parent / ".py-yfinance-cache")
    try:
        yf.cache.set_cache_location(str(_cache_dir))
    except Exception:
        # Cache location is a best-effort improvement; download can still work without it
        pass

    tickers = os.environ.get("TICKERS", "SPY,QQQ,TLT,GLD")
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    df = yf.download(tickers_list, start="2015-01-01", end="2024-01-01", group_by="ticker")
if df is None or df.empty:
    raise RuntimeError(
        "No data loaded for SPY. If you're behind a proxy / in a restricted environment, "
        "set SPY_CSV=/path/to/spy.csv (with columns including Date and Adj Close/Close)."
    )

def build_features(df_raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # Extract per-ticker OHLCV if needed
    if isinstance(df_raw.columns, pd.MultiIndex):
        if ticker not in df_raw.columns.get_level_values(0):
            raise KeyError(f"Ticker {ticker} not found in downloaded data")
        sub = df_raw[ticker].copy()
    else:
        sub = df_raw.copy()

    # yfinance single-ticker sometimes gives MultiIndex (field,ticker); normalize
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(0)

    price = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError(f"Expected single price column for {ticker}, got shape={price.shape}")
        price = price.iloc[:, 0]
    price = price.astype(float)

    out = pd.DataFrame(index=sub.index)
    out["price"] = price
    out["ret"] = price.pct_change()
    out["vol"] = out["ret"].rolling(20).std()
    out["ma_fast"] = price.rolling(10).mean()
    out["ma_slow"] = price.rolling(50).mean()
    out.dropna(inplace=True)

    out["trend"] = np.select(
        [out["ma_fast"] > out["ma_slow"] * 1.001, out["ma_fast"] < out["ma_slow"] * 0.999],
        [2, 0],
        default=1,
    ).astype(int)
    out["ticker"] = ticker
    return out


if csv_path:
    tickers_list = ["SPY"]

frames = [build_features(df, t) for t in tickers_list]

# ----------------------------
# Split + discretize vol per ticker
# ----------------------------
val_start = os.environ.get("VAL_START", "2021-01-01")
test_start = os.environ.get("TEST_START", "2022-01-01")


def split_and_bin(frame: pd.DataFrame):
    tr = frame.loc[frame.index < val_start].copy()
    va = frame.loc[(frame.index >= val_start) & (frame.index < test_start)].copy()
    te = frame.loc[frame.index >= test_start].copy()
    if len(tr) == 0:
        raise ValueError("Train split is empty; adjust VAL_START/TEST_START")
    vol_threshold = tr["vol"].median()
    tr["vol_bin"] = (tr["vol"] > vol_threshold).astype(int)
    if len(va) > 0:
        va["vol_bin"] = (va["vol"] > vol_threshold).astype(int)
    if len(te) > 0:
        te["vol_bin"] = (te["vol"] > vol_threshold).astype(int)
    return tr, va, te


splits = {f["ticker"].iloc[0]: split_and_bin(f) for f in frames}
dfs_train = [splits[t][0] for t in tickers_list]
dfs_val = [splits[t][1] for t in tickers_list if len(splits[t][1]) > 1]

eval_ticker = os.environ.get("EVAL_TICKER", tickers_list[0]).strip().upper()
if eval_ticker not in splits:
    raise ValueError(f"EVAL_TICKER={eval_ticker} not in TICKERS={tickers_list}")
df_train, df_val, df_test = splits[eval_ticker]

print(f"Training tickers: {tickers_list} | eval ticker: {eval_ticker}")

alpha = 0.1
gamma = 0.99
eps_start = float(os.environ.get("EPS_START", "0.2"))
eps_end = float(os.environ.get("EPS_END", "0.01"))
eps_decay_epochs = int(os.environ.get("EPS_DECAY_EPOCHS", "80"))
commission = float(os.environ.get("COMMISSION", "8.0"))
trade_penalty = float(os.environ.get("TRADE_PENALTY", "1.0"))
alpha_reward_scale = float(os.environ.get("ALPHA_REWARD_SCALE", "2.0"))
# Small penalty per step when the agent is flat (discourages "always flat").
# Tune via env var, e.g. FLAT_PENALTY=0 (disable) or 1e-5 (stronger).
flat_penalty = float(os.environ.get("FLAT_PENALTY", "0.25"))
# Optional penalty per step when the agent is long (discourages "always long"/pure buy&hold).
# Tune via env var, e.g. LONG_PENALTY=0 (disable) or 1e-5 (mild).
long_penalty = float(os.environ.get("LONG_PENALTY", "0"))
max_position = int(os.environ.get("MAX_POSITION", "2"))
initial_cash = float(os.environ.get("INITIAL_CASH", "10000"))
epochs = int(os.environ.get("EPOCHS", "400"))
mode = os.environ.get("MODE", "timing_vol_target")  # timing_vol_target | discrete

# ----------------------------
# Training
# ----------------------------
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = Path(__file__).resolve().parent / "runs" / "ch05_stocks" / run_name
writer = SummaryWriter(logdir=str(log_dir))

if mode == "timing_vol_target":
    target_ann_vol = float(os.environ.get("TARGET_ANN_VOL", "0.16"))
    max_leverage = float(os.environ.get("MAX_LEVERAGE", "1.0"))
    # Rebalance only when target shares move materially (reduces churn).
    rebalance_threshold = float(os.environ.get("REBALANCE_THRESHOLD", "0.5"))
    vol_target_penalty = float(os.environ.get("VOL_TARGET_PENALTY", "1000.0"))
    vol_ewma_beta = float(os.environ.get("VOL_EWMA_BETA", "0.94"))
    if len(dfs_train) > 1:
        Q = sql.train_q_table_timing_vol_target_multi(
            dfs_train,
            epochs=epochs,
            alpha=alpha,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_epochs=eps_decay_epochs,
            commission=commission,
            initial_cash=initial_cash,
            alpha_reward_scale=alpha_reward_scale,
            target_ann_vol=target_ann_vol,
            max_leverage=max_leverage,
            rebalance_threshold=rebalance_threshold,
            vol_target_penalty=vol_target_penalty,
            vol_ewma_beta=vol_ewma_beta,
            trade_penalty=trade_penalty,
            flat_penalty=flat_penalty,
            long_penalty=long_penalty,
            seed=42,
            writer=writer,
            dfs_val=dfs_val if len(dfs_val) > 0 else None,
            val_eval_ticker=eval_ticker,
            val_eval_every=int(os.environ.get("VAL_EVAL_EVERY", "1")),
            val_seed=123,
        )
    else:
        Q = sql.train_q_table_timing_vol_target(
            df_train,
            epochs=epochs,
            alpha=alpha,
            gamma=gamma,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_epochs=eps_decay_epochs,
            commission=commission,
            initial_cash=initial_cash,
            alpha_reward_scale=alpha_reward_scale,
            target_ann_vol=target_ann_vol,
            max_leverage=max_leverage,
            rebalance_threshold=rebalance_threshold,
            vol_target_penalty=vol_target_penalty,
            vol_ewma_beta=vol_ewma_beta,
            trade_penalty=trade_penalty,
            flat_penalty=flat_penalty,
            long_penalty=long_penalty,
            seed=42,
            writer=writer,
            df_val=df_val if len(df_val) > 1 else None,
            val_eval_every=int(os.environ.get("VAL_EVAL_EVERY", "1")),
            val_seed=123,
        )
else:
    # In MODE=discrete the agent is solely responsible for sizing (no auto-rebalancing).
    target_ann_vol = float(os.environ.get("TARGET_ANN_VOL", "0.16"))
    vol_target_penalty = float(os.environ.get("VOL_TARGET_PENALTY", "5000.0"))
    vol_ewma_beta = float(os.environ.get("VOL_EWMA_BETA", "0.94"))
    Q = sql.train_q_table(
        df_train,
        epochs=epochs,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_epochs=eps_decay_epochs,
        cost=commission,
        flat_penalty=flat_penalty,
        long_penalty=long_penalty,
        max_position=max_position,
        initial_cash=initial_cash,
        alpha_reward_scale=alpha_reward_scale,
        target_ann_vol=target_ann_vol,
        vol_target_penalty=vol_target_penalty,
        vol_ewma_beta=vol_ewma_beta,
        trade_penalty=trade_penalty,
        seed=42,
        writer=writer,
        writer_prefix="train",
        df_val=df_val if len(df_val) > 1 else None,
        val_cost=commission,
        val_eval_every=int(os.environ.get("VAL_EVAL_EVERY", "1")),
        val_seed=123,
    )

# ----------------------------
# Evaluate strategy
# ----------------------------
if mode == "timing_vol_target":
    test_stats = sql.eval_policy_timing_vol_target(
        df_test,
        Q,
        commission=commission,
        initial_cash=initial_cash,
        target_ann_vol=float(os.environ.get("TARGET_ANN_VOL", "0.16")),
        max_leverage=float(os.environ.get("MAX_LEVERAGE", "1.0")),
        rebalance_threshold=float(os.environ.get("REBALANCE_THRESHOLD", "0.5")),
        seed=123,
    )
    equity = test_stats["equity"]
    positions = test_stats["positions"]
    trades = int(test_stats["trades"])
    equity_np = np.asarray(equity, dtype=np.float64)
    # Log final test scalars
    writer.add_scalar("test/final_equity", test_stats["final_equity"], 0)
    writer.add_scalar("test/sharpe", test_stats["sharpe"], 0)
    writer.add_scalar("test/ann_vol", test_stats["ann_vol"], 0)
    writer.add_scalar("test/trades", test_stats["trades"], 0)
    writer.add_scalar("test/exposure", test_stats["exposure"], 0)
else:
    equity, positions, trades = sql.eval_policy(
        df_test, Q, cost=commission, seed=123, max_position=max_position, initial_cash=initial_cash
    )
    equity_np = np.asarray(equity, dtype=np.float64)
    if equity_np.size > 1:
        test_rets = equity_np[1:] / equity_np[:-1] - 1.0
        writer.add_scalar("test/sharpe", sql._nan_safe_sharpe(test_rets), 0)
        writer.add_scalar("test/final_equity", float(equity_np[-1]), 0)
        writer.add_scalar("test/trades", int(trades), 0)
        avg_pos = float(np.mean(np.asarray(positions, dtype=np.float64)))
        writer.add_scalar("test/avg_position", avg_pos, 0)
        writer.add_scalar("test/exposure", avg_pos / float(max_position) if max_position > 0 else 0.0, 0)
writer.close()

def baseline_always_in_vol_target(df_: pd.DataFrame):
    cash = float(initial_cash)
    shares = 0
    trades_ = 0
    equity_ = [cash]
    pos_ = [0]
    target_ann_vol_ = float(os.environ.get("TARGET_ANN_VOL", "0.16"))
    max_lev_ = float(os.environ.get("MAX_LEVERAGE", "1.0"))
    reb_thr_ = float(os.environ.get("REBALANCE_THRESHOLD", "0.5"))

    for i in range(len(df_) - 1):
        row = df_.iloc[i]
        next_row = df_.iloc[i + 1]
        price_t = float(row["price"])
        price_t1 = float(next_row["price"])
        asset_ann_vol = float(row["vol"]) * float(np.sqrt(252.0))
        equity_t = cash + shares * price_t

        w = sql._compute_target_weight(target_ann_vol_, asset_ann_vol, max_lev_)
        target_shares = int((w * equity_t) // price_t) if price_t > 0 else 0

        do_rebalance = True
        if shares > 0 and target_shares > 0:
            rel = abs(target_shares - shares) / float(shares)
            do_rebalance = rel >= reb_thr_
        if not do_rebalance:
            target_shares = shares

        cash, shares, did_trade = sql._rebalance_to_target_shares(
            cash=cash,
            shares=shares,
            price=price_t,
            target_shares=target_shares,
            commission=commission,
        )
        if did_trade:
            trades_ += 1

        equity_t1 = cash + shares * price_t1
        equity_.append(equity_t1)
        pos_.append(int(shares > 0))

    return np.asarray(equity_, dtype=np.float64), np.asarray(pos_, dtype=np.float64), int(trades_)


# Baseline
test_price = df_test["price"].astype(float)
if mode == "timing_vol_target":
    buy_hold, bh_positions, bh_trades = baseline_always_in_vol_target(df_test)
    baseline_maxpos = 1
else:
    # Buy and hold (same max_position shares, single entry commission)
    bh_cash = float(initial_cash) - float(max_position) * float(test_price.iloc[0]) - float(commission)
    bh_shares = float(max_position)
    buy_hold = (bh_cash + bh_shares * test_price).to_numpy(dtype=np.float64)
    bh_positions = np.full(df_test.index.size, float(max_position), dtype=np.float64)
    bh_trades = 1
    baseline_maxpos = int(max_position)

# Debug / summary output (vectorized)
q_norm_maxpos = 1 if mode == "timing_vol_target" else int(max_position)
q_stats = backtest_stats(
    "Q-learning (test)",
    df_test.index[: len(equity_np)],
    equity_np,
    np.asarray(positions, dtype=np.float64),
    int(trades),
    int(q_norm_maxpos),
)
bh_stats = backtest_stats(
    "Buy & Hold (test)",
    df_test.index,
    np.asarray(buy_hold, dtype=np.float64),
    bh_positions,
    int(bh_trades),
    int(baseline_maxpos),
)

# Log a couple more test scalars (makes run-to-run comparisons easier)
w2 = SummaryWriter(logdir=str(log_dir))
w2.add_scalar("test/max_dd", q_stats["max_dd"], 0)
if q_stats["cagr"] is not None:
    w2.add_scalar("test/cagr", q_stats["cagr"], 0)
w2.add_scalar("baseline/final_equity", bh_stats["final_equity"], 0)
w2.add_scalar("baseline/sharpe", bh_stats["sharpe"], 0)
w2.add_scalar("baseline/max_dd", bh_stats["max_dd"], 0)
w2.close()

# ----------------------------
# Plot
# ----------------------------
plt.plot(df_test.index[: len(equity_np)], equity_np, label="Q-learning (test)")
plt.plot(df_test.index, np.asarray(buy_hold, dtype=np.float64), label="Buy & Hold (test)")
plt.legend()
plt.title("Tabular Q-learning vs Buy & Hold")
plt.show()
