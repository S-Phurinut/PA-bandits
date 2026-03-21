import numpy as np
import scipy as sc
import math
from scipy.special import expit


class Logit:
    def __init__(
        self,
        num_agent,
        lambda_grid=None,
        fallback_shape_tol=1e-5,
        max_restarts=4
    ):
        self.num_agent = num_agent
        self.para_loc = np.ones(num_agent) * 0.5
        self.para_shape = np.ones(num_agent) * 0.001
        self.reset = True
        self.name = "logit"
        self.para_type = "loc-shape"

        self.lambda_grid = (
            lambda_grid
            if lambda_grid is not None
            else [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        )
        self.fallback_shape_tol = fallback_shape_tol
        self.max_restarts = max_restarts

    def fit(self, X, Y):
        if self.reset:
            self.para_loc = np.ones(self.num_agent) * 0.5
            self.para_shape = np.ones(self.num_agent) * 0.001
            self.reset = False

        for agent in range(self.num_agent):
            x = np.asarray(X[:, agent], dtype=float)
            y = np.asarray(Y[:, agent], dtype=float)

            result = self._fit_single_agent_adaptive(agent, x, y)

            if result is None:
                # emergency fallback
                self.para_loc[agent] = 0.5
                self.para_shape[agent] = 0.1
            else:
                self.para_loc[agent] = result["loc"]
                self.para_shape[agent] = result["shape"]

        self.para_loc = np.clip(self.para_loc, 1e-8, 1 - 1e-8)
        self.para_shape = np.clip(self.para_shape, 1e-8, 1.0)

    def _fit_single_agent_adaptive(self, agent, x, y):
        # 1) try ordinary MLE
        mle = self._fit_single_agent_mle(agent, x, y)

        if mle is not None and self._is_good_fit(mle["opt"], mle["shape"]):
            mle["method"] = "MLE"
            mle["lambda"] = 0.0
            return mle

        # 2) adaptive penalized fallback
        for lam in self.lambda_grid:
            pmle = self._fit_single_agent_penalized(agent, x, y, lam)
            if pmle is not None and self._is_good_fit(pmle["opt"], pmle["shape"]):
                pmle["method"] = "PMLE"
                pmle["lambda"] = lam
                return pmle

        # 3) if nothing is "good", return best available penalized fit with largest lambda
        best = None
        for lam in self.lambda_grid:
            pmle = self._fit_single_agent_penalized(agent, x, y, lam)
            if pmle is None:
                continue
            if best is None or pmle["loss"] < best["loss"]:
                best = pmle
                best["method"] = "PMLE"
                best["lambda"] = lam

        if best is not None:
            return best

        if mle is not None:
            mle["method"] = "MLE_bad"
            mle["lambda"] = 0.0
            return mle

        return None

    def _is_good_fit(self, opt, shape_hat):
        if opt is None:
            return False
        if not np.isfinite(opt.fun):
            return False
        if not opt.success:
            return False
        if shape_hat < self.fallback_shape_tol:
            return False
        return True

    def _fit_single_agent_mle(self, agent, x, y):
        best_opt = None
        best_loss = math.inf

        eps = 1e-8
        bounds = [(eps, 1 - eps), (eps, 1.0)]

        for i in range(self.max_restarts):
            if i == 0:
                x0 = np.array([self.para_loc[agent], self.para_shape[agent]], dtype=float)
            else:
                x0 = np.array([
                    np.random.uniform(eps, 1 - eps),
                    np.random.uniform(1e-3, 0.2)
                ])

            opt = sc.optimize.minimize(
                self.CE_loss,
                x0=x0,
                args=(x, y),
                bounds=bounds,
                method="L-BFGS-B"
            )

            if opt.x is not None and np.isfinite(opt.fun) and opt.fun < best_loss:
                best_loss = opt.fun
                best_opt = opt

        if best_opt is None:
            return None

        return {
            "loc": float(best_opt.x[0]),
            "shape": float(best_opt.x[1]),
            "loss": float(best_opt.fun),
            "opt": best_opt
        }

    def _fit_single_agent_penalized(self, agent, x, y, lam):
        """
        Fit in beta-space:
            p = expit(beta0 + beta1 * x), beta1 > 0
        with ridge penalty lam * (beta0^2 + beta1^2)
        """
        best_opt = None
        best_loss = math.inf

        eps = 1e-8
        loc0 = np.clip(self.para_loc[agent], eps, 1 - eps)
        shape0 = max(self.para_shape[agent], 1e-3)

        beta1_init = 1.0 / shape0
        beta0_init = -loc0 / shape0

        for i in range(self.max_restarts):
            if i == 0:
                theta0 = np.array([beta0_init, np.log(beta1_init)], dtype=float)
            else:
                theta0 = np.array([
                    np.random.normal(0.0, 1.0),
                    np.random.normal(0.0, 1.0)
                ])

            opt = sc.optimize.minimize(
                self.penalized_loss_beta,
                x0=theta0,
                args=(x, y, lam),
                method="L-BFGS-B"
            )

            if opt.x is not None and np.isfinite(opt.fun) and opt.fun < best_loss:
                best_loss = opt.fun
                best_opt = opt

        if best_opt is None:
            return None

        beta0 = float(best_opt.x[0])
        beta1 = float(np.exp(best_opt.x[1]))

        shape_hat = 1.0 / beta1
        loc_hat = -beta0 / beta1

        loc_hat = np.clip(loc_hat, eps, 1 - eps)
        shape_hat = np.clip(shape_hat, eps, 1.0)

        return {
            "loc": loc_hat,
            "shape": shape_hat,
            "loss": float(best_opt.fun),
            "opt": best_opt
        }

    def CE_loss(self, para, X, y):
        loc, shape = para
        y_pred = expit((X - loc) / shape)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def penalized_loss_beta(self, theta, X, y, lam):
        beta0 = theta[0]
        beta1 = np.exp(theta[1])  # enforce beta1 > 0

        z = beta0 + beta1 * X
        y_pred = expit(z)

        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)

        nll = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        penalty = lam * (beta0**2 + beta1**2)

        return nll + penalty

    def prob_accept(self, incentive):
        incentive = np.asarray(incentive, dtype=float)
        return expit((incentive - self.para_loc) / self.para_shape)