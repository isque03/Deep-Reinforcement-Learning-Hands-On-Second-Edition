import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing sibling module when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import stocks_q_learning as sql  # noqa: E402


class TestStocksQLearning(unittest.TestCase):
    def test_encode_state_range_and_type(self):
        for trend in (0, 1, 2):
            for position in (0, 1):
                for vol in (0, 1):
                    s = sql.encode_state(trend, position, vol)
                    self.assertIsInstance(s, int)
                    self.assertGreaterEqual(s, 0)
                    self.assertLess(s, 12)

    def test_valid_actions_mask(self):
        a0 = sql.valid_actions(0)
        a1 = sql.valid_actions(1)
        self.assertTrue(isinstance(a0, np.ndarray))
        self.assertTrue(isinstance(a1, np.ndarray))
        self.assertEqual(set(map(int, a0.tolist())), {0, 1})
        self.assertEqual(set(map(int, a1.tolist())), {0, 2})

    def test_max_position_support(self):
        # With max_position=3, we have 3*(3+1)*2 = 24 states
        max_pos = 3
        self.assertEqual(sql.encode_state_with_maxpos(2, 3, 1, max_position=max_pos), 2 * (max_pos + 1) * 2 + 3 * 2 + 1)

        # Action masking at boundaries
        self.assertEqual(set(map(int, sql.valid_actions_with_maxpos(0, max_position=max_pos).tolist())), {0, 1})
        self.assertEqual(set(map(int, sql.valid_actions_with_maxpos(max_pos, max_position=max_pos).tolist())), {0, 2})
        self.assertEqual(set(map(int, sql.valid_actions_with_maxpos(2, max_position=max_pos).tolist())), {0, 1, 2})

        rng = np.random.default_rng(10)
        n = 120
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        df = pd.DataFrame(
            {
                "trend": rng.integers(0, 3, size=n, dtype=np.int64),
                "vol_bin": rng.integers(0, 2, size=n, dtype=np.int64),
                "price": price,
            }
        )
        Q = sql.train_q_table(df, epochs=2, epsilon=0.5, seed=0, max_position=max_pos)
        self.assertEqual(Q.shape, (3 * (max_pos + 1) * 2, 3))

    def test_train_q_table_no_indexing_errors(self):
        rng = np.random.default_rng(0)
        n = 300
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        df = pd.DataFrame(
            {
                "trend": rng.integers(0, 3, size=n, dtype=np.int64),
                "vol_bin": rng.integers(0, 2, size=n, dtype=np.int64),
                "price": price,
            }
        )

        Q = sql.train_q_table(df, epochs=2, epsilon=0.2, seed=123)
        self.assertEqual(Q.shape, (12, 3))
        self.assertTrue(np.isfinite(Q).all())

        # Should learn *something* on random data: not all Q-values remain exactly zero
        self.assertGreater(float(np.abs(Q).sum()), 0.0)

    def test_train_handles_one_element_series_values(self):
        # Reproduces the failure mode: row['trend'] is a 1-element Series
        rng = np.random.default_rng(1)
        n = 100

        cols = pd.MultiIndex.from_tuples(
            [("trend", "x"), ("vol_bin", "x"), ("price", "x")], names=["feat", "sub"]
        )
        arr = np.zeros((n, 3), dtype=float)
        arr[:, 0] = rng.integers(0, 3, size=n)
        arr[:, 1] = rng.integers(0, 2, size=n)
        arr[:, 2] = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        df = pd.DataFrame(arr, columns=cols)

        # train_q_table should not crash even though row['trend'] etc are Series
        Q = sql.train_q_table(df, epochs=1, epsilon=0.5, seed=7)
        self.assertEqual(Q.shape, (12, 3))

    def test_train_tensorboard_writer_hook(self):
        class DummyWriter:
            def __init__(self):
                self.scalars = []

            def add_scalar(self, tag, scalar_value, global_step=None):
                self.scalars.append((str(tag), float(scalar_value), int(global_step)))

        rng = np.random.default_rng(2)
        n = 50
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        df = pd.DataFrame(
            {
                "trend": rng.integers(0, 3, size=n, dtype=np.int64),
                "vol_bin": rng.integers(0, 2, size=n, dtype=np.int64),
                "price": price,
            }
        )
        df_val = df.copy()
        w = DummyWriter()
        _Q = sql.train_q_table(
            df,
            df_val=df_val,
            val_eval_every=1,
            epochs=3,
            epsilon=0.5,
            seed=0,
            writer=w,
            writer_prefix="train",
        )
        tags = {t for (t, _v, _s) in w.scalars}
        self.assertIn("train/reward_sum", tags)
        self.assertIn("train/sharpe", tags)
        self.assertIn("train/epsilon", tags)
        self.assertIn("train/exposure", tags)
        self.assertIn("train/turnover", tags)
        self.assertIn("train/avg_equity", tags)
        self.assertIn("train/final_equity", tags)
        self.assertIn("train/ann_vol", tags)
        self.assertIn("train/ann_vol_ewma", tags)
        self.assertIn("val/sharpe", tags)
        self.assertIn("val/final_equity", tags)
        self.assertIn("val/avg_equity", tags)
        self.assertIn("val/trades", tags)
        self.assertIn("val/exposure", tags)
        steps = {s for (_t, _v, s) in w.scalars if _t == "train/reward_sum"}
        self.assertEqual(steps, {0, 1, 2})

        eps_by_step = {s: v for (t, v, s) in w.scalars if t == "train/epsilon"}
        self.assertEqual(set(eps_by_step.keys()), {0, 1, 2})

    def test_eval_policy_returns_trade_count(self):
        rng = np.random.default_rng(3)
        n = 80
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        df = pd.DataFrame(
            {
                "trend": rng.integers(0, 3, size=n, dtype=np.int64),
                "vol_bin": rng.integers(0, 2, size=n, dtype=np.int64),
                "price": price,
            }
        )
        Q = sql.train_q_table(df, epochs=2, epsilon=0.8, seed=0)
        equity, positions, trades = sql.eval_policy(df, Q, cost=0.0, seed=0, initial_cash=10000.0)
        self.assertEqual(len(equity), len(df))
        self.assertEqual(len(positions), len(df))
        self.assertIsInstance(trades, int)
        self.assertGreaterEqual(trades, 0)

    def test_eval_policy_applies_cost_on_trades_only(self):
        # Construct a Q-table that forces buy on first step then hold.
        max_pos = 1
        Q = np.zeros((3 * (max_pos + 1) * 2, 3), dtype=np.float64)
        # For state (trend=0, pos=0, vol=0) prefer buy
        s0 = sql.encode_state_with_maxpos(0, 0, 0, max_position=max_pos)
        Q[s0, 1] = 1.0
        # For state (trend=0, pos=1, vol=0) prefer hold
        s1 = sql.encode_state_with_maxpos(0, 1, 0, max_position=max_pos)
        Q[s1, 0] = 1.0

        df = pd.DataFrame({"trend": [0, 0, 0], "vol_bin": [0, 0, 0], "price": [100.0, 100.0, 100.0]})
        equity, positions, trades = sql.eval_policy(df, Q, cost=5.0, seed=0, max_position=max_pos, initial_cash=1000.0)
        # One trade (buy) then hold
        self.assertEqual(trades, 1)
        self.assertEqual(positions, [0, 1, 1])
        # No price change, so only commission reduces equity.
        self.assertAlmostEqual(equity[-1], 995.0, places=10)

    def test_flat_penalty_changes_reward_signal(self):
        # If flat_penalty is enabled, staying flat should be slightly worse than without it.
        rng = np.random.default_rng(4)
        n = 60
        df = pd.DataFrame(
            {
                "trend": np.zeros(n, dtype=np.int64),
                "vol_bin": np.zeros(n, dtype=np.int64),
                "price": np.full(n, 100.0, dtype=np.float64),
            }
        )
        # With zero returns, the only learning signal is penalties/costs.
        Q0 = sql.train_q_table(df, epochs=1, epsilon=0.0, seed=0, flat_penalty=0.0)
        Qp = sql.train_q_table(df, epochs=1, epsilon=0.0, seed=0, flat_penalty=1.0)
        self.assertGreater(float(np.abs(Qp).sum()), float(np.abs(Q0).sum()))

    def test_timing_vol_target_logs_ann_vol_ewma(self):
        class DummyWriter:
            def __init__(self):
                self.tags = set()

            def add_scalar(self, tag, scalar_value, global_step=None):
                self.tags.add(str(tag))

        rng = np.random.default_rng(5)
        n = 120
        price = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, size=n))
        ret = pd.Series(price).pct_change().fillna(0.0).to_numpy()
        vol = pd.Series(ret).rolling(20).std().fillna(0.0).to_numpy()
        df = pd.DataFrame(
            {
                "trend": rng.integers(0, 3, size=n, dtype=np.int64),
                "vol_bin": rng.integers(0, 2, size=n, dtype=np.int64),
                "price": price,
                "vol": vol,
            }
        )
        w = DummyWriter()
        _Q = sql.train_q_table_timing_vol_target(
            df,
            epochs=3,
            eps_start=0.5,
            eps_end=0.1,
            eps_decay_epochs=2,
            commission=5.0,
            initial_cash=10000.0,
            target_ann_vol=0.16,
            max_leverage=1.0,
            rebalance_threshold=0.5,
            vol_target_penalty=1.0,
            vol_ewma_beta=0.9,
            writer=w,
        )
        self.assertIn("train/ann_vol_ewma", w.tags)
        self.assertIn("train/target_ann_vol", w.tags)


if __name__ == "__main__":
    unittest.main()

