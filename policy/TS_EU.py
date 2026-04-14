import numpy as np
import scipy.signal
import scipy.signal.windows

# Patch older import name for PyMC compatibility
if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian
import pymc as pm
import pytensor.tensor as pt

import scipy as sc
from scipy.stats import beta as beta_dist
import cvxpy as cp
import math
from poibin import PoiBin


class TS_EU():  # EU with agent approx model
    def __init__(self, type_arm, **alg):
        self.type_arm = type_arm
        self.alg = alg
        self.num_cost_learning = self.alg['num_cost_learning']
        if "cost_alg" in self.alg:
            self.cost_alg = self.alg['cost_alg']
        else:
            self.cost_alg = None
            self.num_cost_learning = 0

        if 'is_reward_known' in alg:
            self.is_reward_known = alg['is_reward_known']
        else:
            self.is_reward_known = False

        if 'is_model_known' in alg:
            self.is_model_known = alg['is_model_known']
        else:
            self.is_model_known = False

        self.num_optimiser = alg['num_optimiser']
        self.is_cost_learning_done = False
        self.is_model_training_done = False
        self.need_model_training = True
        self.reset = True

        # -------- KL smoothing config --------
        self.kl_lambda = self.alg.get("kl_lambda", 0.0)
        self.kl_mode = self.alg.get("kl_mode", "forward")   # "forward", "reverse", "symmetric"
        self.kl_decay = self.alg.get("kl_decay", None)      # None, "1/t", "1/sqrt_t"

    def update_data(self, player):
        self.player = player

    def run(self, **info):
        self.curr_round = info['curr_round']

        if info['curr_round'] == 1 and self.reset:
            self.need_model_training = True
            self.is_cost_learning_done = False
            self.is_model_training_done = False
            self.previous_c = np.ones((self.player.num_agent,)) * 0.5
            self.previous_p = np.ones((self.player.num_agent,)) * 0.5
            self.sum_reward = np.zeros((self.player.num_agent,))
            self.num_reward = np.zeros((self.player.num_agent,))

            if self.num_cost_learning == 'log2T':
                self.num_cost_learning = math.ceil(np.log2(info['max_round']))
            elif self.num_cost_learning == 'logT':
                self.num_cost_learning = math.ceil(np.log(info['max_round']))
            elif self.num_cost_learning == 'T1/2':
                self.num_cost_learning = math.ceil(np.sqrt(info['max_round']))

            if self.cost_alg == "uniformly-space":
                self.cost_list = list(np.linspace(1E-12, 1 - (1E-12), int(self.num_cost_learning)))

            if self.alg['est_reward'] == 'TS' or self.alg['est_reward'] == 'TS-Gibbs-monotone' or self.alg['est_reward'] == 'posterior-mean' and self.type_arm == 'participation-based':
                if self.alg['prior'] is not None:
                    if self.alg['prior'][0] == 'beta':
                        if self.alg['prior'][1][0] == 'fixed':
                            self.alpha = np.ones((self.player.num_agent,)) * self.alg['prior'][1][1]
                        elif self.alg['prior'][1][0] == 'linear':
                            self.alpha = np.linspace(self.alg['prior'][1][1], self.alg['prior'][1][2], num=self.player.num_agent)

                        if self.alg['prior'][2][0] == 'fixed':
                            self.beta = np.ones((self.player.num_agent,)) * self.alg['prior'][2][1]
                        elif self.alg['prior'][2][0] == 'linear':
                            self.alpha = np.linspace(self.alg['prior'][2][1], self.alg['prior'][2][2], num=self.player.num_agent)
                else:
                    self.alpha = np.ones((self.player.num_agent,)) * 1
                    self.beta = np.ones((self.player.num_agent,))

            elif self.alg['est_reward'] == "increasing-TS" or self.alg['est_reward'] == "concave-TS":
                self.input = []
                self.output = []
        else:
            self.reset = False
            self.alg['model'].update_data(
                X=self.player.incentive_array[:(info['curr_round'] - 1), :],
                Y=self.player.agent_response_array[:(info['curr_round'] - 1), :]
            )

            # ---------- Update agent model learning ----------
            if info['curr_round'] > 1 and self.is_model_known == False and self.need_model_training == True:
                train_model = False
                if type(self.alg['model_training_appr']) == int:
                    if (info['curr_round'] - 1) % self.alg['model_training_appr'] == 0 or info['curr_round'] <= self.alg['model_training_max_round_1step']:
                        train_model = True
                else:
                    if self.alg['model_training_appr'] == 'once' and self.is_model_training_done == False:
                        train_model = True
                    elif self.alg['model_training_appr'] == 'T':
                        train_model = True
                    elif self.alg['model_training_appr'] == "adaptive-log10":
                        refit_step = max(int(10 ** (math.floor(np.log10(info['curr_round'])))), 1)
                        if info['curr_round'] % min(refit_step, self.alg['model_training_max_round_step']) == 0 or info['curr_round'] <= self.alg['model_training_max_round_1step']:
                            train_model = True
                        else:
                            train_model = False

                if train_model:
                    self.alg['model'].fit()

                if info['curr_round'] == self.num_cost_learning:
                    self.is_model_training_done = True

            # ----------- Update Reward Data ---------
            n = np.sum(info['previous_agent_response']) - 1
            if n >= 0:
                self.sum_reward[n] += info['previous_reward']
                self.num_reward[n] += 1

                if self.alg['est_reward'] == 'TS' or self.alg['est_reward'] == 'posterior-mean' or self.alg['est_reward'] == 'TS-Gibbs-monotone':
                    if info['previous_reward'] > 0:
                        self.alpha[n] += 1
                    else:
                        self.beta[n] += 1

                elif self.alg['est_reward'] == "increasing-TS" or self.alg['est_reward'] == "concave-TS":
                    self.input.append(n + 1)
                    self.output.append(info['previous_reward'])

        if info['curr_round'] <= self.num_cost_learning:
            if self.cost_alg == "uniformly-space":
                best_cost = np.ones((self.player.num_agent,)) * self.cost_list[int(info['curr_round'] - 1)]
                self.need_model_training = False
            elif self.cost_alg == "approx-D-optimal":
                best_cost = np.clip(self.approx_D_optimal(curr_round=info['curr_round']), 0, 1)
                self.need_model_training = False
            elif self.cost_alg == "A-optimal":
                pass
            elif self.cost_alg == "ex-BinSearch":
                best_cost = np.clip(self.exBinSearch(info['curr_round']), 0, 1)
                self.need_model_training = False
            elif self.cost_alg == "ex-BinSearch2":
                best_cost = np.clip(self.exBinSearch2(info['curr_round'], self.num_cost_learning), 0, 1)
                self.need_model_training = False

            if info['curr_round'] == self.num_cost_learning:
                self.is_cost_learning_done = True
                self.need_model_training = True
        else:
            self.need_model_training = True
            self.is_cost_learning_done = True

            # ==================== Contracting Part ============================
            if self.type_arm == 'participation-based':

                # --------------- Reward estimator -----------------
                if self.is_reward_known:
                    est_reward = np.array(self.player.reward_generator.mean)
                else:
                    if self.alg['est_reward'] == 'UCB-lattimore':
                        est_reward = self.UCBlat_value(info['max_round'])
                    elif self.alg['est_reward'] == 'UCB1':
                        est_reward = self.UCB1_value(info['curr_round'], info['max_round'])
                    elif self.alg['est_reward'] == 'BayesUCB':
                        est_reward = self.BayesUCB_value(info['curr_round'], info['max_round'])
                    elif self.alg['est_reward'] == 'TS':
                        est_reward = self.TS_value()
                    elif self.alg['est_reward'] == 'posterior-mean':
                        est_reward = self.alpha / (self.alpha + self.beta)

                eps = 0
                bnds = [(float(0 - eps), float(1 + eps)) for _ in range(self.player.num_agent)]
                best_EU = -math.inf
                best_cost = np.ones((self.player.num_agent,))

                self.model_para_sample = self.alg['model'].get_sample()
                print(self.model_para_sample)

                for i in range(0, self.num_optimiser):
                    if i == 0:
                        x0 = np.clip(self.previous_c, 1e-6, 1 - (1e-6))
                    else:
                        x0 = np.clip(np.random.rand(self.player.num_agent,), 0.01, 0.9)

                    opt = sc.optimize.minimize(
                        self.EU_value,
                        x0=x0,
                        bounds=bnds,
                        args=(est_reward),
                        tol=1E-12
                    )
                    cost = opt.x
                    EU = -opt.fun
                    if cost is not None and EU > best_EU:
                        best_EU = float(EU)
                        best_cost = np.array(cost)

                self.previous_c = np.array(best_cost)

                if self.is_model_known:
                    self.previous_p = np.array(self.player.agent_policy.prob_accept(best_cost))
                else:
                    self.previous_p = np.array(self.alg['model'].prob_accept(best_cost, **self.model_para_sample))

            if info['curr_round'] % 1000 == 0:
                print("num reward=", self.num_reward)

            if info['curr_round'] == info['max_round']:
                print("final incentive=", np.round(best_cost, 4))
                print("num reward=", self.num_reward)

        return best_cost

    def EU_value(self, cost, reward):
        if self.is_model_known:
            p = np.array(self.player.agent_policy.prob_accept(cost))
        else:
            p = np.array(self.alg['model'].prob_accept(cost, **self.model_para_sample))

        EU = 0.0

        if np.sum(p <= 1E-4) >= self.player.num_agent:
            EU += 0.0
        elif np.sum(p >= 1 - 1E-12) >= self.player.num_agent:
            EU += reward[self.player.num_agent - 1] - np.dot(p, cost)
        else:
            pb = PoiBin(p)
            num_offered_agent = int(np.sum(p > 1E-6))
            num_guaranteed_agent = int(np.sum(p >= 1 - 1E-6))
            for arm in range(max(num_guaranteed_agent, 1), num_offered_agent + 1):
                EU += reward[arm - 1] * np.clip(pb.pmf(arm), 0, 1)
            EU += -np.dot(p, cost)

        # -------- KL smoothing in acceptance-probability space --------
        lam = self.get_kl_lambda()
        if lam > 0:
            smooth_penalty = self.kl_smoothing_penalty(
                p_current=p,
                p_reference=self.previous_p
            )
            EU = EU - lam * smooth_penalty

        return -EU

    def get_kl_lambda(self):
        lam = float(self.kl_lambda)
        if self.kl_decay is None:
            return lam

        t = max(int(getattr(self, "curr_round", 1)), 1)

        if self.kl_decay == "1/t":
            return lam / t
        elif self.kl_decay == "1/sqrt_t":
            return lam / np.sqrt(t)
        else:
            return lam

    def bernoulli_kl(self, p, q):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
        q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def kl_smoothing_penalty(self, p_current, p_reference):
        p_current = np.clip(np.asarray(p_current, dtype=float), 1e-12, 1 - 1e-12)
        p_reference = np.clip(np.asarray(p_reference, dtype=float), 1e-12, 1 - 1e-12)

        if self.kl_mode == "forward":
            # KL(current || previous)
            return float(np.sum(self.bernoulli_kl(p_current, p_reference)))
        elif self.kl_mode == "reverse":
            # KL(previous || current)
            return float(np.sum(self.bernoulli_kl(p_reference, p_current)))
        elif self.kl_mode == "symmetric":
            return float(
                np.sum(self.bernoulli_kl(p_current, p_reference)) +
                np.sum(self.bernoulli_kl(p_reference, p_current))
            )
        else:
            raise ValueError("kl_mode must be 'forward', 'reverse', or 'symmetric'")

    def UCBlat_value(self, max_round):
        UCB = np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n] == 0:
                UCB[n] = 1 + 2 * np.sqrt(np.log(max_round))
            else:
                UCB[n] = self.sum_reward[n] / self.num_reward[n] + 2 * np.sqrt(np.log(max_round) / self.num_reward[n])
        return UCB

    def UCB1_value(self, round, max_round):
        UCB = np.zeros((self.player.num_agent,))
        for n in range(self.player.num_agent):
            if self.num_reward[n] == 0:
                UCB[n] = 1 + 2 * np.sqrt(np.log(max_round))
            else:
                UCB[n] = self.sum_reward[n] / self.num_reward[n] + np.sqrt(2 * np.log(round) / self.num_reward[n])
        return UCB

    def TS_value(self):
        sampled_thetas = []
        for n in range(self.player.num_agent):
            sample = np.random.beta(self.alpha[n], self.beta[n])
            sampled_thetas.append(sample)
        return np.array(sampled_thetas)

    def BayesUCB_value(self, round, max_round):
        BayesUCB = np.zeros((self.player.num_agent,))

        if self.alg['conf_bound'] == '1/t' or self.alg['conf_bound'] is None:
            q = 1.0 - 1.0 / max(2, round)

        for n in range(self.player.num_agent):
            BayesUCB[n] = sc.stats.beta.ppf(q, self.alpha[n], self.beta[n])

        return BayesUCB

    def approx_D_optimal(self, curr_round):
        if self.reset:
            self.count = 0

        if curr_round == 0:
            return np.ones((self.player.num_agent,))
        else:
            if self.count % 2 == 0:
                if self.alg['model'].name == "logit":
                    if self.alg['model'].para_type == "loc-shape":
                        self.x_mid = self.alg['model'].para_loc
                        self.b = self.alg['model'].para_shape
                elif self.alg['model'].name == "bayes-logit":
                    if self.alg['model'].para_type == "loc-shape":
                        self.x_mid = self.alg['model'].u_mean
                        self.b = self.alg['model'].s_mean

                        for i in range(self.player.num_agent):
                            self.x_mid[i], self.b[i] = self.alg['model'].get_MAP_estimator(agent_id=i)

                x = self.x_mid + (1.543 * self.b)
            else:
                x = self.x_mid - (1.543 * self.b)
            self.count += 1
        return x

    def A_optimal(self):
        return 1

    def exBinSearch(self, curr_round):
        x = np.ones((self.player.num_agent,)) / self.player.num_agent
        if curr_round > 1:
            for i in range(self.player.num_agent):
                if self.player.agent_response_array[curr_round - 2, i] == 0:
                    x[i] = (self.player.incentive_array[curr_round - 2, i]) + 0.5 / self.player.num_agent
                else:
                    x[i] = (self.player.incentive_array[curr_round - 2, i]) / 2
        print("exBinSearch at cost=", x)
        return x

    def exBinSearch2(self, curr_round, max_cost_round):
        x = np.ones((self.player.num_agent,)) / self.player.num_agent
        if curr_round > 1:
            for i in range(self.player.num_agent):
                if self.player.agent_response_array[curr_round - 2, i] == 0:
                    x[i] = (self.player.incentive_array[curr_round - 2, i]) + (1 - (1 / self.player.num_agent)) / (max_cost_round - 1)
                else:
                    x[i] = (self.player.incentive_array[curr_round - 2, i]) - (1 - (1 / self.player.num_agent)) / (max_cost_round - 1)
        print("exBinSearch at cost=", x)
        return x