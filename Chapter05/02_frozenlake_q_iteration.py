#!/usr/bin/env python3
import collections
import os
import random
import sys

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import RecordVideo
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"
# Map configuration
# - Gymnasium provides built-in presets "4x4" and "8x8" via map_name.
# - For "16x16" we generate a random valid map once and reuse it.
MAP_NAME = "8x8"            # "4x4", "8x8", or "16x16"
IS_SLIPPERY = True
MAP_RANDOM_P = 0.8          # probability of frozen tile when generating random maps
MAP_RANDOM_SEED = 12345     # set to None for a new random map each run


def build_env_kwargs():
    if MAP_NAME in ("4x4", "8x8"):
        return {"map_name": MAP_NAME, "is_slippery": IS_SLIPPERY}
    if MAP_NAME == "16x16":
        desc = generate_random_map(size=16, p=MAP_RANDOM_P, seed=MAP_RANDOM_SEED)
        return {"desc": desc, "is_slippery": IS_SLIPPERY}
    raise ValueError(f"Unsupported MAP_NAME={MAP_NAME!r}")


ENV_KWARGS = build_env_kwargs()
GAMMA = 0.9
TEST_EPISODES = 20
MAX_ITERATIONS = 3000

# Video recording (mp4)
RECORD_VIDEO_EVERY = 0        # set to e.g. 50 to record every 50 iterations
RECORD_VIDEO_ON_SOLVE = False
POST_SOLVE_VIDEO_EPISODES = 5
VIDEO_DIR = "videos"

# ASCII value/policy dashboard
PRINT_VALUES_EVERY = 5        # set to e.g. 10 to print dashboard every 10 iterations
PRINT_POLICY = True
LIVE_ASCII = True             # refresh in-place (TTY only)
COLOR_ASCII = True            # colorize cells (TTY only; disabled if NO_COLOR is set)

# Optional random initialization for Q(s,a)
RANDOM_INIT_Q = False
RANDOM_INIT_SEED = 12345      # set to None for non-deterministic
RANDOM_INIT_LOW = 0.0
RANDOM_INIT_HIGH = 1.0
PRINT_INITIAL_VALUES = True

# Experience collection / exploration
USE_EPSILON_GREEDY_STEPS = False
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY_ITERS = 1000        # iterations to decay from start to end (linear)
STEPS_PER_ITER = 100


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, **ENV_KWARGS)
        print('Observation space:', self.env.observation_space)
        print('Action space:', self.env.action_space)
        self.state, _ = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)  # Q(s,a)

        if RANDOM_INIT_Q:
            rng = random.Random(RANDOM_INIT_SEED) if RANDOM_INIT_SEED is not None else random
            for s in range(self.env.observation_space.n):
                for a in range(self.env.action_space.n):
                    self.values[(s, a)] = rng.uniform(RANDOM_INIT_LOW, RANDOM_INIT_HIGH)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def select_action_epsilon_greedy(self, state, epsilon: float):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        return self.select_action(state)

    def play_n_epsilon_greedy_steps(self, count, epsilon: float):
        for _ in range(count):
            action = self.select_action_epsilon_greedy(self.state, epsilon)
            new_state, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            if is_done:
                self.state, _ = self.env.reset()
            else:
                self.state = new_state

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            is_done = terminated or truncated
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                if total == 0:
                    # No data yet for this (state, action): keep existing Q(s,a).
                    continue
                for tgt_state, count in target_counts.items():
                    key = (state, action, tgt_state)
                    reward = self.rewards[key]
                    best_action = self.select_action(tgt_state)
                    val = reward + GAMMA * \
                          self.values[(tgt_state, best_action)]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME, **ENV_KWARGS)
    video_env = {"env": None}
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    def epsilon_by_iter(iteration: int) -> float:
        if EPS_DECAY_ITERS <= 0:
            return float(EPS_END)
        frac = min(max(iteration / EPS_DECAY_ITERS, 0.0), 1.0)
        return float(EPS_START + frac * (EPS_END - EPS_START))

    def get_video_env():
        if video_env["env"] is None:
            base_video_env = gym.make(ENV_NAME, **ENV_KWARGS, render_mode="rgb_array")
            video_env["env"] = RecordVideo(
                base_video_env,
                video_folder=VIDEO_DIR,
                name_prefix="frozenlake-q",
                episode_trigger=lambda ep: True,
            )
        return video_env["env"]

    def q_to_v_and_policy(q_values):
        def v(s: int) -> float:
            return max(q_values[(s, a)] for a in range(agent.env.action_space.n))

        def pi(s: int) -> int:
            best_a, best_q = 0, q_values[(s, 0)]
            for a in range(1, agent.env.action_space.n):
                q = q_values[(s, a)]
                if q > best_q:
                    best_q = q
                    best_a = a
            return best_a

        return v, pi

    def format_values_grid(env, value_fn, policy_fn=None, color=False):
        desc = env.unwrapped.desc
        nrow, ncol = desc.shape
        action_to_char = {0: "<", 1: "v", 2: ">", 3: "^"}

        vals = [float(value_fn(s)) for s in range(nrow * ncol)]
        vmin = min(vals)
        vmax = max(vals)
        span = (vmax - vmin) if vmax > vmin else 1.0

        def colorize(text, bg_256=None, fg_256=None):
            if bg_256 is None and fg_256 is None:
                return text
            seq = ""
            if fg_256 is not None:
                seq += f"\x1b[38;5;{fg_256}m"
            if bg_256 is not None:
                seq += f"\x1b[48;5;{bg_256}m"
            return f"{seq}{text}\x1b[0m"

        def value_bg(v):
            palette = [196, 202, 208, 220, 190, 46]  # red -> green
            norm = (v - vmin) / span
            idx = int(norm * (len(palette) - 1) + 1e-9)
            idx = max(0, min(len(palette) - 1, idx))
            bg = palette[idx]
            fg = 97 if bg in (196, 202, 208) else 30
            return bg, fg

        lines = []
        for r in range(nrow):
            row_parts = []
            for c in range(ncol):
                s = r * ncol + c
                ch = desc[r, c]
                if isinstance(ch, (bytes, bytearray)):
                    ch = ch.decode("utf-8")
                v = float(value_fn(s))
                if policy_fn is None:
                    cell = f"{ch}:{v:5.2f}"
                else:
                    a = int(policy_fn(s))
                    cell = f"{ch}{action_to_char.get(a, '?')}:{v:5.2f}"
                if color:
                    if ch == "H":
                        cell = colorize(cell, bg_256=240, fg_256=97)
                    elif ch == "G":
                        cell = colorize(cell, bg_256=27, fg_256=97)
                    else:
                        bg, fg = value_bg(v)
                        cell = colorize(cell, bg_256=bg, fg_256=fg)
                row_parts.append(cell)
            lines.append("  ".join(row_parts))
        return "\n".join(lines)

    def print_values_grid(env, value_fn, policy_fn=None, header=None, live=False, color=False):
        body = format_values_grid(env, value_fn, policy_fn=policy_fn, color=color)
        if header:
            body = f"{header}\n{body}"
        if live and sys.stdout.isatty():
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(body)
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            print(body)

    def status_bar(iteration, avg_reward, best_reward, epsilon_value):
        eps_text = "off" if epsilon_value is None else f"{epsilon_value:.3f}"
        return f"iter={iteration} | avg_reward={avg_reward:.3f} | best_reward={best_reward:.3f} | epsilon={eps_text}"

    iter_no = 0
    best_reward = 0.0

    if RANDOM_INIT_Q and PRINT_INITIAL_VALUES and PRINT_VALUES_EVERY > 0:
        v_fn, pi_fn = q_to_v_and_policy(agent.values)
        print_values_grid(
            test_env,
            v_fn,
            policy_fn=pi_fn if PRINT_POLICY else None,
            header="--- initial V(s)=max_a Q(s,a) ---",
            live=LIVE_ASCII,
            color=COLOR_ASCII and sys.stdout.isatty() and not os.environ.get("NO_COLOR"),
        )

    while True:
        if iter_no >= MAX_ITERATIONS:
            print(f"Stopped after {MAX_ITERATIONS} iterations (best_reward={best_reward:.3f})")
            break
        iter_no += 1
        cur_eps = None
        if USE_EPSILON_GREEDY_STEPS:
            cur_eps = epsilon_by_iter(iter_no)
            agent.play_n_epsilon_greedy_steps(STEPS_PER_ITER, cur_eps)
        else:
            agent.play_n_random_steps(STEPS_PER_ITER)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if PRINT_VALUES_EVERY > 0 and iter_no % PRINT_VALUES_EVERY == 0:
            v_fn, pi_fn = q_to_v_and_policy(agent.values)
            print_values_grid(
                test_env,
                v_fn,
                policy_fn=pi_fn if PRINT_POLICY else None,
                header=status_bar(iter_no, reward, best_reward, cur_eps),
                live=LIVE_ASCII,
                color=COLOR_ASCII and sys.stdout.isatty() and not os.environ.get("NO_COLOR"),
            )
        if RECORD_VIDEO_EVERY > 0 and iter_no % RECORD_VIDEO_EVERY == 0:
            agent.play_episode(get_video_env())
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            if RECORD_VIDEO_ON_SOLVE:
                for _ in range(POST_SOLVE_VIDEO_EPISODES):
                    agent.play_episode(get_video_env())
            break
    writer.close()
    test_env.close()
    agent.env.close()
    if video_env["env"] is not None:
        video_env["env"].close()
