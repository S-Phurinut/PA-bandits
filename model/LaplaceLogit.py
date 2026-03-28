import numpy as np
import scipy as sc
import math
from scipy.special import expit
from scipy.stats import chi2


class Laplace_Logit:
    def __init__(self, num_agent, **model):
        self.num_agent = num_agent
        self.para_loc = np.ones((self.num_agent,)) / num_agent
        self.para_shape = np.ones((self.num_agent,)) * 0.01

        self.reset = True
        self.name = "logit"
        self.para_type = "loc-shape"
        self.model = model

        # ---- NEW: UCB parameters ----
        self.c_mode = self.model.get("c_mode", "chisq")   # "chisq" or "fixed"
        self.c_value = self.model.get("c_value", 0.95)    # confidence level or constant
        self.ridge = self.model.get("ridge", 1e-6)

        # covariance (per agent, 2x2)
        self.cov = np.array([np.eye(2) for _ in range(self.num_agent)])

    # -------------------------------
    # FIT (same optimization + add covariance)
    # -------------------------------
    def fit(self, X, Y):
        if self.reset:
            self.para_loc = np.ones((self.num_agent,)) / self.num_agent
            self.para_shape = np.ones((self.num_agent,)) * 0.01
            self.reset = False

        for agent in range(self.num_agent):
            best_loss = math.inf

            for i in range(4):
                if i == 0:
                    x0 = np.array([self.para_loc[agent], self.para_shape[agent]])
                else:
                    x0 = np.random.random(2,) / 10

                eps = 1e-12
                x0 = np.clip(x0, eps, 1 - eps)
                bnds = [(eps, 1 - eps), (eps, 1.0)]

                opt = sc.optimize.minimize(
                    self.CE_loss,
                    x0=x0,
                    bounds=bnds,
                    args=(np.array(X[:, agent], dtype=float),
                          np.array(Y[:, agent], dtype=float))
                )

                if opt.x is not None and opt.fun < best_loss:
                    best_loss = opt.fun
                    self.para_loc[agent] = opt.x[0]
                    self.para_shape[agent] = opt.x[1]

            # ---- NEW: compute Laplace covariance ----
            self.cov[agent] = self.compute_covariance(
                self.para_loc[agent],
                self.para_shape[agent],
                np.array(X[:, agent], dtype=float),
                np.array(Y[:, agent], dtype=float)
            )

        self.para_loc = np.clip(self.para_loc, 1e-8, 1 - 1e-8)
        self.para_shape = np.clip(self.para_shape, 1e-8, 1)


    def CE_loss(self, para, X, y):
        y_pred = expit((X - para[0]) / para[1])
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        if self.model['est_appr'] == 'MLE':
            penalty = 0
        elif self.model['est_appr'] == 'MAP':
            if self.model['shape_prior'][0] == "gamma":
                penalty = (
                    self.model['shape_prior'][2] * para[1]
                    - (self.model['shape_prior'][1] - 1.0) * np.log(para[1])
                )

        return -loss + penalty


    def compute_covariance(self, u, s, X, y):
        """
        Laplace covariance for (u, s)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        eps = 1e-8

        z = (X - u) / s
        p = expit(z)
        w = p * (1 - p)

        # derivatives
        du = -1.0 / s
        ds = -(X - u) / (s**2)

        J = np.vstack([du * np.ones_like(X), ds]).T  # Jacobian

        H = J.T @ (J * w[:, None]) + self.ridge * np.eye(2)

        return np.linalg.inv(H + eps * np.eye(2))

    def prob_accept(self, incentive):
        p = np.zeros((self.num_agent,))
        for agent in range(self.num_agent):
            p[agent] = expit(
                (incentive[agent] - self.para_loc[agent]) / self.para_shape[agent]
            )
        return p


    def prob_accept_ucb(self, incentive, curr_round=None, delta=None):
        p = np.zeros((self.num_agent,))
        c_t = self.get_c_t(curr_round, delta)

        for agent in range(self.num_agent):
            u = self.para_loc[agent]
            s = self.para_shape[agent]

            x = incentive[agent]

            mean_logit = (x - u) / s

            # gradient wrt (u, s)
            grad = np.array([
                -1.0 / s,
                -(x - u) / (s**2)
            ])

            var = grad @ self.cov[agent] @ grad
            std = np.sqrt(max(var, 1e-12))

            p[agent] = expit(mean_logit + c_t * std)

        print("agent", agent, "x", x, "u", u, "s", s,
      "mean_logit", mean_logit, "std", std, "bonus", c_t * std)
        
        return np.clip(p, 1e-12, 1 - 1e-12)


    def get_c_t(self, curr_round=None, delta=None):
        if self.c_mode == "fixed":
            return float(self.c_value)

        # chi-square based
        if delta is not None:
            conf = 1.0 - delta
        else:
            conf = self.c_value

        conf = np.clip(conf, 1e-6, 1 - 1e-6)
        gamma = chi2.ppf(conf, df=2)

        return float(np.sqrt(gamma))