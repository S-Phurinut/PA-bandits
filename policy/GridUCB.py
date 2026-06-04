import numpy as np
import math
import itertools


class GridUCB():
    """
    Baseline theoretical algorithm:
        discretise incentive space and run UCB over grid contracts.

    This class follows the same interface style as TS_EU:

        alg = UCB_Discretised_EU(type_arm, **alg_config)
        alg.update_data(player)
        cost = alg.run(**info)

    Each grid point is a full incentive vector beta in [beta_min,beta_max]^N.

    Theoretical grid:
        h_T = (beta_max-beta_min) T^{-1/(N+2)}
        grid_intervals = ceil(T^{1/(N+2)})
        grid_per_dim = grid_intervals + 1

    Regret:
        R_T = O~(T^((N+1)/(N+2))).
    """

    def __init__(self, type_arm, **alg):
        self.type_arm = type_arm
        self.alg = alg

        self.beta_min = self.alg.get("beta_min", 0.0)
        self.beta_max = self.alg.get("beta_max", 1.0)

        self.sigma = self.alg.get("sigma", 1.0)

        self.reward_is_net = self.alg.get("reward_is_net", False)

        self.grid_rule = self.alg.get("grid_rule", "theoretical")
        self.grid_per_dim = self.alg.get("grid_per_dim", None)

        self.max_grid_size = self.alg.get("max_grid_size", 200000)

        self.verbose = self.alg.get("verbose", True)

        self.reset = True

    def update_data(self, player):
        self.player = player

    def run(self, **info):
        self.curr_round = info["curr_round"]

        if info["curr_round"] == 1 and self.reset:
            self.reset = False
            self.previous_arm_idx = None
            self.build_theoretical_grid(max_round=info["max_round"])

            self.num_pull = np.zeros((self.num_grid_arm,), dtype=int)
            self.sum_reward = np.zeros((self.num_grid_arm,), dtype=float)
            self.mean_reward = np.zeros((self.num_grid_arm,), dtype=float)

        else:
            # update previous grid arm using realised reward
            if info["curr_round"] > 1:
                self.update_grid_reward(info)

        best_arm_idx = self.UCB_grid_arm(
            curr_round=info["curr_round"],
            max_round=info["max_round"]
        )

        self.previous_arm_idx = best_arm_idx

        best_cost = np.array(self.grid[best_arm_idx], dtype=float)

        if info["curr_round"] % 1000 == 0:
            print(
                "[UCB-Discretised-EU]",
                "round=", info["curr_round"],
                "played arms=", int(np.sum(self.num_pull > 0)),
                "best empirical mean=", float(np.max(self.mean_reward)),
            )

        if info["curr_round"] == info["max_round"]:
            print("[UCB-Discretised-EU] final incentive=", np.round(best_cost, 4))
            print("[UCB-Discretised-EU] theoretical h_T=", self.theoretical_h)
            print("[UCB-Discretised-EU] actual h=", self.actual_h)
            print("[UCB-Discretised-EU] grid_per_dim=", self.grid_per_dim_actual)
            print("[UCB-Discretised-EU] num_grid_arm=", self.num_grid_arm)

        return best_cost

    def build_theoretical_grid(self, max_round):
        N = self.player.num_agent

        if self.grid_rule == "theoretical":
            grid_intervals = int(math.ceil(max_round ** (1.0 / (N + 2))))
            grid_intervals = max(grid_intervals, 1)
            grid_per_dim = grid_intervals + 1

        elif self.grid_rule == "manual":
            if self.grid_per_dim is None:
                raise ValueError("Provide grid_per_dim when grid_rule='manual'.")
            grid_per_dim = int(self.grid_per_dim)
            grid_per_dim = max(grid_per_dim, 2)
            grid_intervals = grid_per_dim - 1

        else:
            raise ValueError("grid_rule must be 'theoretical' or 'manual'.")

        total_grid_size = grid_per_dim ** N

        if total_grid_size > self.max_grid_size:
            raise ValueError(
                f"Grid too large: grid_per_dim={grid_per_dim}, "
                f"num_agent={N}, total arms={total_grid_size}. "
                f"Use grid_rule='manual' with smaller grid_per_dim, "
                f"or increase max_grid_size."
            )

        self.theoretical_h = (
            (self.beta_max - self.beta_min)
            * max_round ** (-1.0 / (N + 2))
        )

        self.actual_h = (
            (self.beta_max - self.beta_min)
            / float(grid_intervals)
        )

        one_dim_grid = np.linspace(
            self.beta_min,
            self.beta_max,
            grid_per_dim
        )

        self.grid = np.array(
            list(itertools.product(one_dim_grid, repeat=N)),
            dtype=float
        )

        self.grid_per_dim_actual = grid_per_dim
        self.grid_intervals_actual = grid_intervals
        self.num_grid_arm = self.grid.shape[0]

        if self.verbose:
            print("[UCB-Discretised-EU] theoretical grid")
            print("num_agent=", N)
            print("max_round=", max_round)
            print("theoretical h_T=", self.theoretical_h)
            print("actual h=", self.actual_h)
            print("grid_per_dim=", self.grid_per_dim_actual)
            print("num_grid_arm=", self.num_grid_arm)

    def update_grid_reward(self, info):
        if self.previous_arm_idx is None:
            return

        reward = float(info["previous_reward"])

        if self.reward_is_net:
            net_reward = reward
        else:
            previous_response = np.array(
                info["previous_agent_response"],
                dtype=float
            )
            previous_cost = self.grid[self.previous_arm_idx]
            net_reward = reward - float(np.dot(previous_cost, previous_response))

        arm = self.previous_arm_idx

        self.num_pull[arm] += 1
        self.sum_reward[arm] += net_reward
        self.mean_reward[arm] = self.sum_reward[arm] / self.num_pull[arm]

    def UCB_grid_arm(self, curr_round, max_round):
        unplayed = np.where(self.num_pull == 0)[0]

        if len(unplayed) > 0:
            return int(np.random.choice(unplayed))

        bonus = np.sqrt(
            2.0 * (self.sigma ** 2) * np.log(max_round)
            / self.num_pull
        )

        UCB = self.mean_reward + bonus

        best_value = np.max(UCB)
        best_arms = np.where(np.isclose(UCB, best_value))[0]

        return int(np.random.choice(best_arms))