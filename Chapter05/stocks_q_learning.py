import numpy as np


def _to_scalar(x):
    """
    Convert numpy/pandas/list-like singletons to a Python scalar.

    This exists because pandas rows can yield 1-element Series when the DataFrame
    has MultiIndex/duplicate columns (which breaks plain int()/float() casting).
    """
    # pandas Series (and similar) with .iloc
    if hasattr(x, "iloc") and hasattr(x, "__len__"):
        if len(x) != 1:
            raise TypeError(f"Expected scalar or 1-element value, got length={len(x)}: {type(x)}")
        return x.iloc[0]

    # numpy arrays / list-like
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.size != 1:
            raise TypeError(f"Expected scalar or 1-element value, got size={arr.size}: {type(x)}")
        return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]

    # numpy scalar
    if isinstance(x, np.generic):
        return x.item()

    return x


def _to_int(x):
    return int(_to_scalar(x))


def _to_float(x):
    return float(_to_scalar(x))


def _nan_safe_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio of a return series using mean/std * sqrt(periods_per_year).
    Returns 0.0 if variance is 0 or series is empty.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=0))
    if sigma == 0.0 or not np.isfinite(sigma):
        return 0.0
    return mu / sigma * float(np.sqrt(periods_per_year))


def _nan_safe_ann_vol(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    sigma = float(np.std(r, ddof=0))
    if sigma == 0.0 or not np.isfinite(sigma):
        return 0.0
    return sigma * float(np.sqrt(periods_per_year))


def _compute_target_weight(target_ann_vol: float, asset_ann_vol: float, max_leverage: float) -> float:
    """
    Volatility targeting weight. If asset_ann_vol is ~0, returns 0.
    """
    if asset_ann_vol is None or not np.isfinite(asset_ann_vol) or asset_ann_vol <= 0.0:
        return 0.0
    w = float(target_ann_vol) / float(asset_ann_vol)
    return float(np.clip(w, 0.0, float(max_leverage)))


def _rebalance_to_target_shares(
    *,
    cash: float,
    shares: int,
    price: float,
    target_shares: int,
    commission: float,
) -> tuple[float, int, bool]:
    """
    Rebalance shares toward target_shares at price, paying commission once if we trade.
    """
    cash = float(cash)
    shares = int(shares)
    target_shares = int(target_shares)
    commission = float(commission)
    price = float(price)

    delta = target_shares - shares
    if delta == 0:
        return cash, shares, False

    # Buy
    if delta > 0:
        # Pay commission once per rebalance/order
        if cash <= commission or price <= 0.0:
            return cash, shares, False
        affordable = int((cash - commission) // price)
        buy_qty = min(delta, max(0, affordable))
        if buy_qty <= 0:
            return cash, shares, False
        cash = cash - commission - buy_qty * price
        shares = shares + buy_qty
        return cash, shares, True

    # Sell
    sell_qty = min(-delta, shares)
    if sell_qty <= 0:
        return cash, shares, False
    cash = cash - commission + sell_qty * price
    shares = shares - sell_qty
    return cash, shares, True


def train_q_table_timing_vol_target(
    df_train,
    *,
    epochs=50,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    eps_start: float | None = None,
    eps_end: float = 0.01,
    eps_decay_epochs: int = 0,
    commission: float = 5.0,
    initial_cash: float = 10000.0,
    alpha_reward_scale: float = 1.0,
    target_ann_vol: float = 0.16,
    max_leverage: float = 1.0,
    rebalance_threshold: float = 0.2,
    vol_target_penalty: float = 0.0,
    vol_ewma_beta: float = 0.94,
    trade_penalty: float = 0.0,
    flat_penalty: float = 0.0,
    long_penalty: float = 0.0,
    seed: int = 42,
    writer=None,
    df_val=None,
    val_eval_every: int = 1,
    val_seed: int = 123,
):
    """
    Option B: RL learns timing (in/out), deterministic volatility targeting sizes exposure.

    Required columns:
    - trend (int), vol_bin (int): discrete state features
    - price (float): trade price
    - vol (float): rolling daily return std (used to estimate asset annual vol = vol*sqrt(252))
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((12, 3), dtype=np.float64)

    if eps_start is None:
        eps_start = float(epsilon)
    eps_start = float(eps_start)
    eps_end = float(eps_end)
    eps_decay_epochs = int(eps_decay_epochs)
    val_eval_every = int(val_eval_every)

    for epoch_no in range(int(epochs)):
        if eps_decay_epochs > 0:
            frac = min(1.0, epoch_no / float(eps_decay_epochs))
            eps = eps_start + frac * (eps_end - eps_start)
        else:
            eps = eps_start

        cash = float(initial_cash)
        shares = 0
        pos_flag = 0  # 0=flat, 1=in-market
        ewma_var = 0.0  # EWMA of portfolio return variance

        eq_curve = [cash]
        trades = 0
        pos_sum = 0.0
        reward_sum = 0.0

        for i in range(len(df_train) - 1):
            row = df_train.iloc[i]
            next_row = df_train.iloc[i + 1]
            price_t = _to_float(row["price"])
            price_t1 = _to_float(next_row["price"])
            asset_ann_vol = _to_float(row["vol"]) * float(np.sqrt(252.0))

            equity_t = cash + shares * price_t

            s = encode_state(row["trend"], pos_flag, row["vol_bin"])
            acts = valid_actions(pos_flag)
            if rng.random() < eps:
                a = int(rng.choice(acts))
            else:
                a = _select_greedy_action(Q, s, acts, rng)

            # Timing action updates pos_flag only
            if a == 1 and pos_flag == 0:
                pos_flag = 1
            elif a == 2 and pos_flag == 1:
                pos_flag = 0

            # Determine target shares from vol targeting if in-market
            if pos_flag == 1:
                w = _compute_target_weight(target_ann_vol, asset_ann_vol, max_leverage)
                target_shares = int((w * equity_t) // price_t) if price_t > 0 else 0
            else:
                target_shares = 0

            # Optional rebalance band to reduce churn
            do_rebalance = True
            if shares > 0 and target_shares > 0:
                rel = abs(target_shares - shares) / float(shares)
                do_rebalance = rel >= float(rebalance_threshold)
            if not do_rebalance:
                target_shares = shares

            cash, shares, did_trade = _rebalance_to_target_shares(
                cash=cash,
                shares=shares,
                price=price_t,
                target_shares=target_shares,
                commission=commission,
            )
            if did_trade:
                trades += 1

            equity_t1 = cash + shares * price_t1
            commission_paid = float(commission) if did_trade else 0.0
            market_pnl = (equity_t1 - equity_t) + commission_paid
            step_reward = float(alpha_reward_scale) * market_pnl - commission_paid
            if did_trade:
                step_reward -= float(trade_penalty)
            if shares == 0:
                step_reward -= float(flat_penalty)
            else:
                step_reward -= float(long_penalty)
            reward_sum += step_reward

            # Percent return for sharpe/vol logging
            step_ret = (equity_t1 / equity_t - 1.0) if equity_t > 0 else 0.0
            # EWMA volatility tracking + penalty to encourage hitting target vol
            beta = float(vol_ewma_beta)
            ewma_var = beta * ewma_var + (1.0 - beta) * float(step_ret) ** 2
            ann_vol_ewma = float(np.sqrt(252.0 * ewma_var)) if ewma_var > 0 else 0.0
            if float(vol_target_penalty) > 0.0:
                step_reward -= float(vol_target_penalty) * (ann_vol_ewma - float(target_ann_vol)) ** 2

            eq_curve.append(equity_t1)
            pos_sum += float(shares > 0)

            s2 = encode_state(next_row["trend"], pos_flag, next_row["vol_bin"])
            next_acts = valid_actions(pos_flag)
            max_q = float(np.max(Q[s2, next_acts]))
            Q[s, a] += float(alpha) * (step_reward + float(gamma) * max_q - Q[s, a])

        # Logging
        if writer is not None:
            eq_np = np.asarray(eq_curve, dtype=np.float64)
            rets = eq_np[1:] / eq_np[:-1] - 1.0
            writer.add_scalar("train/final_equity", float(eq_np[-1]), epoch_no)
            writer.add_scalar("train/avg_equity", float(np.mean(eq_np)), epoch_no)
            writer.add_scalar("train/sharpe", _nan_safe_sharpe(rets), epoch_no)
            writer.add_scalar("train/ann_vol", _nan_safe_ann_vol(rets), epoch_no)
            writer.add_scalar("train/ann_vol_ewma", ann_vol_ewma, epoch_no)
            writer.add_scalar("train/target_ann_vol", float(target_ann_vol), epoch_no)
            writer.add_scalar("train/vol_target_penalty", float(vol_target_penalty), epoch_no)
            writer.add_scalar("train/alpha_reward_scale", float(alpha_reward_scale), epoch_no)
            writer.add_scalar("train/trades", int(trades), epoch_no)
            writer.add_scalar("train/exposure", float(pos_sum) / max(1.0, float(len(df_train) - 1)), epoch_no)
            writer.add_scalar("train/reward_sum", float(reward_sum), epoch_no)
            writer.add_scalar("train/epsilon", float(eps), epoch_no)

            if df_val is not None and val_eval_every > 0 and (epoch_no % val_eval_every == 0):
                val_stats = eval_policy_timing_vol_target(
                    df_val,
                    Q,
                    commission=float(commission),
                    initial_cash=float(initial_cash),
                    target_ann_vol=float(target_ann_vol),
                    max_leverage=float(max_leverage),
                    rebalance_threshold=float(rebalance_threshold),
                    seed=int(val_seed),
                )
                writer.add_scalar("val/final_equity", val_stats["final_equity"], epoch_no)
                writer.add_scalar("val/avg_equity", val_stats["avg_equity"], epoch_no)
                writer.add_scalar("val/sharpe", val_stats["sharpe"], epoch_no)
                writer.add_scalar("val/ann_vol", val_stats["ann_vol"], epoch_no)
                writer.add_scalar("val/trades", val_stats["trades"], epoch_no)
                writer.add_scalar("val/exposure", val_stats["exposure"], epoch_no)

    return Q


def train_q_table_timing_vol_target_multi(
    dfs_train: list,
    *,
    epochs=50,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    eps_start: float | None = None,
    eps_end: float = 0.01,
    eps_decay_epochs: int = 0,
    commission: float = 5.0,
    initial_cash: float = 10000.0,
    alpha_reward_scale: float = 1.0,
    target_ann_vol: float = 0.16,
    max_leverage: float = 1.0,
    rebalance_threshold: float = 0.2,
    vol_target_penalty: float = 0.0,
    vol_ewma_beta: float = 0.94,
    trade_penalty: float = 0.0,
    flat_penalty: float = 0.0,
    long_penalty: float = 0.0,
    seed: int = 42,
    writer=None,
    dfs_val: list | None = None,
    val_eval_ticker: str | None = None,
    val_eval_every: int = 1,
    val_seed: int = 123,
):
    """
    Multi-instrument version of timing+vol-target training.

    Each epoch, we run one episode per instrument (cash/shares reset each episode),
    updating a shared Q-table.
    """
    if not isinstance(dfs_train, (list, tuple)) or len(dfs_train) == 0:
        raise ValueError("dfs_train must be a non-empty list of DataFrames")

    rng = np.random.default_rng(seed)
    Q = np.zeros((12, 3), dtype=np.float64)

    if eps_start is None:
        eps_start = float(epsilon)
    eps_start = float(eps_start)
    eps_end = float(eps_end)
    eps_decay_epochs = int(eps_decay_epochs)
    val_eval_every = int(val_eval_every)

    def run_one(df_ep, *, eps: float):
        cash = float(initial_cash)
        shares = 0
        pos_flag = 0
        ewma_var = 0.0

        eq_curve = [cash]
        trades = 0
        exposure_steps = 0.0
        reward_sum = 0.0
        step_rets = []

        for i in range(len(df_ep) - 1):
            row = df_ep.iloc[i]
            next_row = df_ep.iloc[i + 1]
            price_t = _to_float(row["price"])
            price_t1 = _to_float(next_row["price"])
            asset_ann_vol = _to_float(row["vol"]) * float(np.sqrt(252.0))

            equity_t = cash + shares * price_t

            s = encode_state(row["trend"], pos_flag, row["vol_bin"])
            acts = valid_actions(pos_flag)
            if rng.random() < eps:
                a = int(rng.choice(acts))
            else:
                a = _select_greedy_action(Q, s, acts, rng)

            if a == 1 and pos_flag == 0:
                pos_flag = 1
            elif a == 2 and pos_flag == 1:
                pos_flag = 0

            if pos_flag == 1:
                w = _compute_target_weight(target_ann_vol, asset_ann_vol, max_leverage)
                target_shares = int((w * equity_t) // price_t) if price_t > 0 else 0
            else:
                target_shares = 0

            do_rebalance = True
            if shares > 0 and target_shares > 0:
                rel = abs(target_shares - shares) / float(shares)
                do_rebalance = rel >= float(rebalance_threshold)
            if not do_rebalance:
                target_shares = shares

            cash, shares, did_trade = _rebalance_to_target_shares(
                cash=cash,
                shares=shares,
                price=price_t,
                target_shares=target_shares,
                commission=commission,
            )
            if did_trade:
                trades += 1

            equity_t1 = cash + shares * price_t1
            commission_paid = float(commission) if did_trade else 0.0
            market_pnl = (equity_t1 - equity_t) + commission_paid
            step_reward = float(alpha_reward_scale) * market_pnl - commission_paid
            if did_trade:
                step_reward -= float(trade_penalty)
            if shares == 0:
                step_reward -= float(flat_penalty)
            else:
                step_reward -= float(long_penalty)

            step_ret = (equity_t1 / equity_t - 1.0) if equity_t > 0 else 0.0
            beta = float(vol_ewma_beta)
            ewma_var = beta * ewma_var + (1.0 - beta) * float(step_ret) ** 2
            ann_vol_ewma = float(np.sqrt(252.0 * ewma_var)) if ewma_var > 0 else 0.0
            if float(vol_target_penalty) > 0.0:
                step_reward -= float(vol_target_penalty) * (ann_vol_ewma - float(target_ann_vol)) ** 2

            reward_sum += step_reward
            step_rets.append(step_ret)
            eq_curve.append(equity_t1)
            exposure_steps += float(shares > 0)

            s2 = encode_state(next_row["trend"], pos_flag, next_row["vol_bin"])
            next_acts = valid_actions(pos_flag)
            max_q = float(np.max(Q[s2, next_acts]))
            Q[s, a] += float(alpha) * (step_reward + float(gamma) * max_q - Q[s, a])

        eq_np = np.asarray(eq_curve, dtype=np.float64)
        rets = np.asarray(step_rets, dtype=np.float64)
        ann_vol = _nan_safe_ann_vol(rets)
        sharpe = _nan_safe_sharpe(rets)
        exposure = float(exposure_steps) / float(max(1, len(df_ep) - 1))
        return {
            "final_equity": float(eq_np[-1]),
            "avg_equity": float(np.mean(eq_np)),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "ann_vol_ewma": float(ann_vol_ewma),
            "trades": int(trades),
            "exposure": float(exposure),
            "reward_sum": float(reward_sum),
            "rets": rets,
        }

    for epoch_no in range(int(epochs)):
        if eps_decay_epochs > 0:
            frac = min(1.0, epoch_no / float(eps_decay_epochs))
            eps = eps_start + frac * (eps_end - eps_start)
        else:
            eps = eps_start

        order = list(range(len(dfs_train)))
        rng.shuffle(order)
        stats = [run_one(dfs_train[i], eps=float(eps)) for i in order]

        all_rets = np.concatenate([s["rets"] for s in stats]) if stats else np.asarray([], dtype=np.float64)
        if writer is not None:
            writer.add_scalar("train/final_equity", float(np.mean([s["final_equity"] for s in stats])), epoch_no)
            writer.add_scalar("train/avg_equity", float(np.mean([s["avg_equity"] for s in stats])), epoch_no)
            writer.add_scalar("train/sharpe", _nan_safe_sharpe(all_rets), epoch_no)
            writer.add_scalar("train/ann_vol", _nan_safe_ann_vol(all_rets), epoch_no)
            writer.add_scalar("train/ann_vol_ewma", float(np.mean([s["ann_vol_ewma"] for s in stats])), epoch_no)
            writer.add_scalar("train/target_ann_vol", float(target_ann_vol), epoch_no)
            writer.add_scalar("train/vol_target_penalty", float(vol_target_penalty), epoch_no)
            writer.add_scalar("train/alpha_reward_scale", float(alpha_reward_scale), epoch_no)
            writer.add_scalar("train/trades", int(np.sum([s["trades"] for s in stats])), epoch_no)
            writer.add_scalar("train/exposure", float(np.mean([s["exposure"] for s in stats])), epoch_no)
            writer.add_scalar("train/reward_sum", float(np.sum([s["reward_sum"] for s in stats])), epoch_no)
            writer.add_scalar("train/epsilon", float(eps), epoch_no)

            if dfs_val is not None and val_eval_every > 0 and (epoch_no % val_eval_every == 0):
                vstats_by_ticker = {}
                for dfv in dfs_val:
                    tkr = str(dfv["ticker"].iloc[0]) if "ticker" in dfv.columns and len(dfv) else "UNK"
                    vstats_by_ticker[tkr] = eval_policy_timing_vol_target(
                        dfv,
                        Q,
                        commission=float(commission),
                        initial_cash=float(initial_cash),
                        target_ann_vol=float(target_ann_vol),
                        max_leverage=float(max_leverage),
                        rebalance_threshold=float(rebalance_threshold),
                        seed=int(val_seed),
                    )

                vstats = list(vstats_by_ticker.values())
                # Mean across instruments
                writer.add_scalar("val_mean/final_equity", float(np.mean([s["final_equity"] for s in vstats])), epoch_no)
                writer.add_scalar("val_mean/avg_equity", float(np.mean([s["avg_equity"] for s in vstats])), epoch_no)
                writer.add_scalar("val_mean/sharpe", float(np.mean([s["sharpe"] for s in vstats])), epoch_no)
                writer.add_scalar("val_mean/ann_vol", float(np.mean([s["ann_vol"] for s in vstats])), epoch_no)
                writer.add_scalar("val_mean/trades", int(np.sum([s["trades"] for s in vstats])), epoch_no)
                writer.add_scalar("val_mean/exposure", float(np.mean([s["exposure"] for s in vstats])), epoch_no)

                # Optional: also log the chosen eval ticker separately (e.g., SPY)
                if val_eval_ticker is not None:
                    key = str(val_eval_ticker).upper()
                    if key in vstats_by_ticker:
                        s = vstats_by_ticker[key]
                        writer.add_scalar("val_eval/final_equity", s["final_equity"], epoch_no)
                        writer.add_scalar("val_eval/avg_equity", s["avg_equity"], epoch_no)
                        writer.add_scalar("val_eval/sharpe", s["sharpe"], epoch_no)
                        writer.add_scalar("val_eval/ann_vol", s["ann_vol"], epoch_no)
                        writer.add_scalar("val_eval/trades", s["trades"], epoch_no)
                        writer.add_scalar("val_eval/exposure", s["exposure"], epoch_no)

    return Q


def eval_policy_timing_vol_target(
    df,
    Q,
    *,
    commission: float = 5.0,
    initial_cash: float = 10000.0,
    target_ann_vol: float = 0.16,
    max_leverage: float = 1.0,
    rebalance_threshold: float = 0.2,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    cash = float(initial_cash)
    shares = 0
    pos_flag = 0
    trades = 0
    pos_sum = 0.0
    equity = [cash]
    pos_flags = [0]

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        price_t = _to_float(row["price"])
        price_t1 = _to_float(next_row["price"])
        asset_ann_vol = _to_float(row["vol"]) * float(np.sqrt(252.0))

        equity_t = cash + shares * price_t
        s = encode_state(row["trend"], pos_flag, row["vol_bin"])
        acts = valid_actions(pos_flag)
        a = _select_greedy_action(Q, s, acts, rng)

        if a == 1 and pos_flag == 0:
            pos_flag = 1
        elif a == 2 and pos_flag == 1:
            pos_flag = 0

        if pos_flag == 1:
            w = _compute_target_weight(target_ann_vol, asset_ann_vol, max_leverage)
            target_shares = int((w * equity_t) // price_t) if price_t > 0 else 0
        else:
            target_shares = 0

        do_rebalance = True
        if shares > 0 and target_shares > 0:
            rel = abs(target_shares - shares) / float(shares)
            do_rebalance = rel >= float(rebalance_threshold)
        if not do_rebalance:
            target_shares = shares

        cash, shares, did_trade = _rebalance_to_target_shares(
            cash=cash, shares=shares, price=price_t, target_shares=target_shares, commission=commission
        )
        if did_trade:
            trades += 1

        equity_t1 = cash + shares * price_t1
        equity.append(equity_t1)
        pos_flags.append(int(shares > 0))
        pos_sum += float(shares > 0)

    eq_np = np.asarray(equity, dtype=np.float64)
    rets = eq_np[1:] / eq_np[:-1] - 1.0
    steps = max(1, len(df) - 1)
    return {
        "equity": equity,
        "positions": pos_flags,
        "final_equity": float(eq_np[-1]),
        "avg_equity": float(np.mean(eq_np)),
        "sharpe": float(_nan_safe_sharpe(rets)),
        "ann_vol": float(_nan_safe_ann_vol(rets)),
        "trades": int(trades),
        "exposure": float(pos_sum) / float(steps),
    }
def _select_greedy_action(Q: np.ndarray, s: int, acts: np.ndarray, rng: np.random.Generator) -> int:
    """
    Greedy action with tie-breaking.

    Plain argmax will systematically pick the first action on ties (e.g. hold=0),
    which can make the policy look artificially "flat" when Q-values are near-equal.
    """
    q = Q[s, acts]
    max_q = float(np.max(q))
    best = np.flatnonzero(q == max_q)
    # deterministic given rng seed
    idx = int(rng.choice(best))
    return int(acts[idx])


def encode_state(trend, position, vol):
    """
    State encoding for (trend in {0,1,2}, position in {0..max_position}, vol in {0,1})
    with max_position=1 by default -> integer in [0, 11].
    """
    return encode_state_with_maxpos(trend, position, vol, max_position=1)


def encode_state_with_maxpos(trend, position, vol, max_position: int):
    max_position = int(max_position)
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position}")
    trend_i = _to_int(trend)
    pos_i = _to_int(position)
    vol_i = _to_int(vol)
    if not (0 <= trend_i <= 2):
        raise ValueError(f"trend out of range: {trend_i}")
    if not (0 <= vol_i <= 1):
        raise ValueError(f"vol out of range: {vol_i}")
    if not (0 <= pos_i <= max_position):
        raise ValueError(f"position out of range: {pos_i} (max_position={max_position})")

    pos_bins = max_position + 1
    stride = pos_bins * 2
    return trend_i * stride + pos_i * 2 + vol_i


def valid_actions(position):
    """
    Action space: 0=hold, 1=buy, 2=sell.
    We mask invalid actions to avoid learning artifacts:
    - If flat: can hold or buy
    - If long: can hold or sell
    """
    return valid_actions_with_maxpos(position, max_position=1)


def valid_actions_with_maxpos(position, max_position: int):
    """
    Discrete position sizing with integer units.
    position in [0..max_position].

    Action space: 0=hold, 1=buy (+1 unit), 2=sell (-1 unit).
    """
    max_position = int(max_position)
    if max_position < 1:
        raise ValueError(f"max_position must be >= 1, got {max_position}")
    pos = _to_int(position)
    if not (0 <= pos <= max_position):
        raise ValueError(f"position out of range: {pos} (max_position={max_position})")
    acts = [0]
    if pos < max_position:
        acts.append(1)
    if pos > 0:
        acts.append(2)
    return np.asarray(acts, dtype=np.int64)


def train_q_table(
    df_train,
    epochs=20,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1,
    eps_start: float | None = None,
    eps_end: float = 0.01,
    eps_decay_epochs: int = 0,
    cost=0.0005,
    flat_penalty: float = 0.0,
    long_penalty: float = 0.0,
    max_position: int = 1,
    initial_cash: float = 10000.0,
    alpha_reward_scale: float = 1.0,
    target_ann_vol: float | None = None,
    vol_target_penalty: float = 0.0,
    vol_ewma_beta: float = 0.94,
    trade_penalty: float = 0.0,
    seed=42,
    writer=None,
    writer_prefix: str = "train",
    df_val=None,
    val_cost: float | None = None,
    val_eval_every: int = 1,
    val_seed: int = 123,
):
    """
    Tabular Q-learning over a single passable historical sequence.

    Expects df_train to have columns: trend (int), vol_bin (int), price (float).
    Reward is computed in *dollars* as change in portfolio equity between t and t+1,
    after executing the action at price_t.
    """
    rng = np.random.default_rng(seed)
    max_position = int(max_position)
    pos_bins = max_position + 1
    n_states = 3 * pos_bins * 2
    n_actions = 3
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    if eps_start is None:
        eps_start = float(epsilon)
    eps_start = float(eps_start)
    eps_end = float(eps_end)
    eps_decay_epochs = int(eps_decay_epochs)

    val_eval_every = int(val_eval_every)
    if val_cost is None:
        val_cost = float(cost)

    for epoch_no in range(int(epochs)):
        if eps_decay_epochs > 0:
            frac = min(1.0, epoch_no / float(eps_decay_epochs))
            eps = eps_start + frac * (eps_end - eps_start)
        else:
            eps = eps_start

        epoch_reward_sum = 0.0  # dollars
        epoch_trades = 0
        epoch_returns = []  # pct returns for sharpe/logging
        epoch_pos_sum = 0.0
        epoch_equity_sum = 0.0
        epoch_equity_last = None
        seg_lens_flat = []
        seg_lens_long = []
        prev_pos = None
        seg_len = 0

        cash = float(initial_cash)
        position = 0  # integer shares in [0..max_position]
        ewma_var = 0.0  # EWMA of portfolio return variance
        ann_vol_ewma = 0.0
        for i in range(len(df_train) - 1):
            row = df_train.iloc[i]
            next_row = df_train.iloc[i + 1]
            price_t = _to_float(row["price"])
            price_t1 = _to_float(next_row["price"])

            equity_t_pre = cash + float(position) * price_t

            # Use [] access to avoid pandas attribute edge-cases
            s = encode_state_with_maxpos(row["trend"], position, row["vol_bin"], max_position=max_position)

            acts = valid_actions_with_maxpos(position, max_position=max_position)
            if rng.random() < float(eps):
                a = int(rng.choice(acts))
            else:
                a = _select_greedy_action(Q, s, acts, rng)

            reward = 0.0
            traded = False

            # Execute action (position changes happen "now")
            if a == 1 and position < max_position:
                # Buy 1 share if we have enough cash for price + commission
                if cash >= price_t + float(cost):
                    cash -= price_t + float(cost)
                    position += 1
                    traded = True
            elif a == 2 and position > 0:
                # Sell 1 share, receive price minus commission
                cash += price_t - float(cost)
                position -= 1
                traded = True

            if traded:
                epoch_trades += 1

            # Equity at t+1
            equity_t1 = cash + float(position) * price_t1
            commission_paid = float(cost) if traded else 0.0
            market_pnl = (equity_t1 - equity_t_pre) + commission_paid
            step_reward_dollars = float(alpha_reward_scale) * market_pnl - commission_paid
            if traded:
                # Extra fixed penalty per trade (in dollars), on top of commissions.
                step_reward_dollars -= float(trade_penalty)
            epoch_equity_sum += 0.5 * (equity_t_pre + equity_t1)
            epoch_equity_last = equity_t1

            # Reward shaping in dollars
            if position == 0:
                step_reward_dollars -= float(flat_penalty)
            else:
                step_reward_dollars -= float(long_penalty)

            # Percent return for diagnostics/sharpe
            step_ret_pct = (equity_t1 / equity_t_pre - 1.0) if equity_t_pre > 0 else 0.0

            # Optional: volatility-targeting penalty (agent controls sizing; no auto-rebalancing).
            beta = float(vol_ewma_beta)
            ewma_var = beta * ewma_var + (1.0 - beta) * float(step_ret_pct) ** 2
            ann_vol_ewma = float(np.sqrt(252.0 * ewma_var)) if ewma_var > 0 else 0.0
            if target_ann_vol is not None and float(vol_target_penalty) > 0.0:
                step_reward_dollars -= float(vol_target_penalty) * (ann_vol_ewma - float(target_ann_vol)) ** 2

            epoch_returns.append(step_ret_pct)
            reward += step_reward_dollars
            epoch_pos_sum += float(position)

            # Segment lengths (how long we stay flat vs long)
            if prev_pos is None:
                prev_pos = int(position)
                seg_len = 1
            elif int(position) == prev_pos:
                seg_len += 1
            else:
                if prev_pos == 0:
                    seg_lens_flat.append(seg_len)
                else:
                    seg_lens_long.append(seg_len)
                prev_pos = int(position)
                seg_len = 1

            s2 = encode_state_with_maxpos(next_row["trend"], position, next_row["vol_bin"], max_position=max_position)
            next_acts = valid_actions_with_maxpos(position, max_position=max_position)
            max_q_s2 = float(np.max(Q[s2, next_acts]))

            Q[s, a] += float(alpha) * (reward + float(gamma) * max_q_s2 - Q[s, a])
            epoch_reward_sum += step_reward_dollars

        # close last segment
        if prev_pos is not None and seg_len > 0:
            if prev_pos == 0:
                seg_lens_flat.append(seg_len)
            else:
                seg_lens_long.append(seg_len)

        if writer is not None:
            step = epoch_no
            epoch_returns_np = np.asarray(epoch_returns, dtype=np.float64)
            sharpe = _nan_safe_sharpe(epoch_returns_np)
            steps = max(1, len(df_train) - 1)
            avg_position = float(epoch_pos_sum) / float(steps)
            exposure = avg_position / float(max_position) if max_position > 0 else 0.0
            turnover = float(epoch_trades) / float(steps)
            avg_equity = float(epoch_equity_sum) / float(steps)
            writer.add_scalar(f"{writer_prefix}/reward_sum", epoch_reward_sum, step)
            writer.add_scalar(f"{writer_prefix}/reward_mean", float(np.mean(epoch_returns_np)) if epoch_returns_np.size else 0.0, step)
            writer.add_scalar(f"{writer_prefix}/reward_std", float(np.std(epoch_returns_np)) if epoch_returns_np.size else 0.0, step)
            writer.add_scalar(f"{writer_prefix}/sharpe", sharpe, step)
            writer.add_scalar(f"{writer_prefix}/ann_vol", _nan_safe_ann_vol(epoch_returns_np), step)
            writer.add_scalar(f"{writer_prefix}/ann_vol_ewma", float(ann_vol_ewma), step)
            if target_ann_vol is not None:
                writer.add_scalar(f"{writer_prefix}/target_ann_vol", float(target_ann_vol), step)
                writer.add_scalar(f"{writer_prefix}/vol_target_penalty", float(vol_target_penalty), step)
            writer.add_scalar(f"{writer_prefix}/alpha_reward_scale", float(alpha_reward_scale), step)
            writer.add_scalar(f"{writer_prefix}/trades", epoch_trades, step)
            writer.add_scalar(f"{writer_prefix}/turnover", turnover, step)
            writer.add_scalar(f"{writer_prefix}/exposure", exposure, step)
            writer.add_scalar(f"{writer_prefix}/avg_position", avg_position, step)
            writer.add_scalar(f"{writer_prefix}/avg_equity", avg_equity, step)
            if epoch_equity_last is not None:
                writer.add_scalar(f"{writer_prefix}/final_equity", float(epoch_equity_last), step)
            writer.add_scalar(f"{writer_prefix}/avg_flat_steps", float(np.mean(seg_lens_flat)) if seg_lens_flat else 0.0, step)
            writer.add_scalar(f"{writer_prefix}/avg_long_steps", float(np.mean(seg_lens_long)) if seg_lens_long else 0.0, step)
            writer.add_scalar(f"{writer_prefix}/q_abs_mean", float(np.mean(np.abs(Q))), step)
            writer.add_scalar(f"{writer_prefix}/epsilon", float(eps), step)

            if df_val is not None and val_eval_every > 0 and (epoch_no % val_eval_every == 0):
                val_equity, val_positions, val_trades = eval_policy(
                    df_val,
                    Q,
                    cost=float(val_cost),
                    seed=int(val_seed),
                    max_position=max_position,
                    initial_cash=float(initial_cash),
                )
                val_equity_np = np.asarray(val_equity, dtype=np.float64)
                if val_equity_np.size > 1:
                    val_rets = val_equity_np[1:] / val_equity_np[:-1] - 1.0
                    writer.add_scalar("val/sharpe", _nan_safe_sharpe(val_rets), step)
                    writer.add_scalar("val/final_equity", float(val_equity_np[-1]), step)
                    writer.add_scalar("val/avg_equity", float(np.mean(val_equity_np)), step)
                    writer.add_scalar("val/trades", int(val_trades), step)
                    val_avg_pos = float(np.mean(np.asarray(val_positions, dtype=np.float64)))
                    writer.add_scalar("val/avg_position", val_avg_pos, step)
                    writer.add_scalar("val/exposure", val_avg_pos / float(max_position) if max_position > 0 else 0.0, step)

    return Q


def eval_policy(df_test, Q, cost=0.0005, seed: int = 123, max_position: int = 1, initial_cash: float = 10000.0):
    """
    Deterministic greedy evaluation on df_test.

    Expects df_test to have columns: trend (int), vol_bin (int), price (float).
    Returns: equity list of floats (length len(df_test)), positions list of ints.
    """
    rng = np.random.default_rng(seed)
    max_position = int(max_position)
    cash = float(initial_cash)
    position = 0
    equity = [cash]
    positions = [0]
    trades = 0

    for i in range(len(df_test) - 1):
        row = df_test.iloc[i]
        next_row = df_test.iloc[i + 1]
        price_t = _to_float(row["price"])
        price_t1 = _to_float(next_row["price"])

        s = encode_state_with_maxpos(row["trend"], position, row["vol_bin"], max_position=max_position)
        acts = valid_actions_with_maxpos(position, max_position=max_position)
        a = _select_greedy_action(Q, s, acts, rng)

        traded = False
        if a == 1 and position < max_position:
            if cash >= price_t + float(cost):
                cash -= price_t + float(cost)
                position += 1
                traded = True
        elif a == 2 and position > 0:
            cash += price_t - float(cost)
            position -= 1
            traded = True
        if traded:
            trades += 1

        eq = cash + float(position) * price_t1
        equity.append(eq)
        positions.append(int(position))

    return equity, positions, trades

