"""Quadratic (and Linear) Reasoning-Aware 3PL

Model
-----
  X_ij = log(1 + B_ij)
  theta_eff_ij = theta_base_j + gamma_j X_ij + kappa_j X_ij^2
  eta_ij = a_i (theta_eff_ij - b_i)
  p_ij = c_i + (1 - c_i) * sigmoid(eta_ij)

Linear special case: kappa_j = 0.

Notes
-----
This file mirrors the structural style of `gem_item_gate.py`:
  - E-step via Gauss–Hermite posterior weights over theta
  - M-steps for (a_i,b_i) and (gamma_j,kappa_j)
  - Post-hoc identifiability pin (re-center/rescale theta; co-transform params)
  - Bracket-style uncertainty summaries (tau + probabilistic truncated bracket)
  - Optional post-hoc Louis-method SE utilities (observed information via Louis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Iterable

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit
from scipy.optimize import minimize

EPS = 1e-6


# ------------------------------
# User-facing configuration/data/result
# ------------------------------


@dataclass
class CalibrationConfig:
    gh_nodes: int = 17
    max_iter: int = 50
    tol: float = 1e-4
    seed: Optional[int] = 0
    verbose: bool = True

    use_quadratic: bool = True
    optimal_kappa: bool = False
    tau: float = 0.75

    ridge_a: float = 1e-3
    ridge_b: float = 1e-3
    ridge_gamma: float = 0.1
    ridge_kappa: float = 0.1

    mu_a: float = 1.0
    mu_b: float = 0.0
    mu_gamma: float = 0.2
    mu_kappa: float = -0.02

    a_min: float = 0.05
    a_max: float = 5.00

    gamma_min: float = -0.5
    gamma_max: float = 2.0

    kappa_min: float = -2.0
    kappa_max: float = -EPS

    x0_j: Optional[np.ndarray] = None # shape (N,), mu of optimal log1p budget
    ridge_x0: float = 0.1

    def __post_init__(self) -> None:
        if self.use_quadratic:
            # concavity: gamma >= 0, kappa < 0
            if self.gamma_min < 0.0:
                self.gamma_min = 0.0
            if self.kappa_max >= 0.0:
                self.kappa_max = -EPS


@dataclass
class CalibrationData:
    Y_ij: NDArray
    B_ij: NDArray
    c_i: NDArray | float


@dataclass
class CalibrationResult:
    a_i: NDArray
    b_i: NDArray
    gamma_j: NDArray
    kappa_j: NDArray

    theta_j_mean: NDArray
    theta_j_se: NDArray

    # Bracket-style uncertainties (half-widths)
    theta_bracket_half_tau: NDArray     # (N,) tau-bracket for theta
    theta_bracket_half_ideal: NDArray   # (N,) probabilistic truncated bracket (replaces "ideal")

    b_bracket_half_tau: NDArray         # (Q,) tau-bracket for b
    b_bracket_half_ideal: NDArray       # (Q,) probabilistic truncated bracket (replaces "ideal")

    gamma_bracket_half_tau: NDArray     # (N,) tau-bracket for gamma
    gamma_bracket_half_ideal: NDArray   # (N,) probabilistic truncated bracket (replaces "ideal")

    kappa_bracket_half_tau: NDArray     # (N,) tau-bracket for kappa (NaN in linear mode)
    kappa_bracket_half_ideal: NDArray   # (N,) probabilistic truncated bracket (replaces "ideal")

    w_post_jk: NDArray
    theta_k: NDArray
    info: Dict[str, float | list[float] | object]


# ------------------------------
# Calibrator (GEM orchestrator)
# ------------------------------


class ReasoningAware3PLQuadraticCalibrator:
    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def fit(self, data: CalibrationData, strict_Y_int: bool = True) -> CalibrationResult:
        Y_ij, X_ij, c_i = self._prepare_inputs(data, strict_Y_int=strict_Y_int)
        Q, N = Y_ij.shape

        a_i, b_i, gamma_j, kappa_j = self._init_params(Y_ij, c_i, 
                                                       use_quadratic=self.cfg.use_quadratic,
                                                       optimal_kappa=self.cfg.optimal_kappa)
        theta_k, w_k = self._make_gh(self.cfg.gh_nodes)

        w_post_jk = np.tile(w_k[None, :], (N, 1))  # (N,K)
        ll_trace: list[float] = []
        prev_ll = -np.inf

        for t in range(self.cfg.max_iter):
            # 1) E-step
            w_post_jk, ll_obs = self._estep(
                Y_ij=Y_ij,
                X_ij=X_ij,
                a_i=a_i,
                b_i=b_i,
                gamma_j=gamma_j,
                kappa_j=kappa_j,
                c_i=c_i,
                theta_k=theta_k,
                w_k=w_k,
                use_quadratic=self.cfg.use_quadratic,
            )
            ll_trace.append(ll_obs)

            # 2) M-step: items (a_i, b_i)
            a_i, b_i = self._mstep_items(
                Y_ij=Y_ij,
                X_ij=X_ij,
                a_i=a_i,
                b_i=b_i,
                gamma_j=gamma_j,
                kappa_j=kappa_j,
                c_i=c_i,
                theta_k=theta_k,
                w_post_jk=w_post_jk,
                use_quadratic=self.cfg.use_quadratic,
            )

            # 3) M-step: model gain (gamma_j, kappa_j)
            gamma_j, kappa_j = self._mstep_model_gain(
                Y_ij=Y_ij,
                X_ij=X_ij,
                a_i=a_i,
                b_i=b_i,
                gamma_j=gamma_j,
                kappa_j=kappa_j,
                c_i=c_i,
                theta_k=theta_k,
                w_post_jk=w_post_jk,
                use_quadratic=self.cfg.use_quadratic,
            )

            # 4) Pin identifiability
            a_i, b_i, gamma_j, kappa_j = self._pin_identifiability(
                a_i=a_i,
                b_i=b_i,
                gamma_j=gamma_j,
                kappa_j=kappa_j,
                w_post_jk=w_post_jk,
                theta_k=theta_k,
            )

            if self.cfg.verbose:
                d = ll_trace[-1] - (ll_trace[-2] if t > 1 else np.nan)
                print(f"iter {t+1:3d}  ll={ll_obs:.6f}   dLL={d:.3e}")

            if abs(ll_obs - prev_ll) < self.cfg.tol:
                if self.cfg.verbose:
                    print(f"converged at iter {t+1}: ll={ll_trace[-1]:.6f}")
                break
            prev_ll = ll_obs

        theta_j_mean, theta_j_se = self._summarize_theta(w_post_jk, theta_k)

        info: Dict[str, float | list[float] | object] = {
            "iters": float(t + 1),
            "final_loglik": float(ll_trace[-1]) if ll_trace else float("nan"),
            "ll_trace": ll_trace,
            "theta_k": theta_k,
            "use_quadratic": bool(self.cfg.use_quadratic),
            "tau": float(self.cfg.tau),
            "input_data": data,
        }

        # Bracket-style uncertainties
        tau = float(self.cfg.tau)
        prob_K = 3

        theta_br_tau, theta_br_prob = self._compute_theta_brackets(
            a_i=a_i,
            b_i=b_i,
            gamma_j=gamma_j,
            kappa_j=kappa_j,
            c_i=c_i,
            X_ij=X_ij,
            w_post_jk=w_post_jk,
            theta_k=theta_k,
            theta_j_mean=theta_j_mean,
            tau=tau,
            prob_K=prob_K,
        )
        b_br_tau, b_br_prob = self._compute_b_brackets(
            a_i=a_i,
            b_i=b_i,
            gamma_j=gamma_j,
            kappa_j=kappa_j,
            c_i=c_i,
            X_ij=X_ij,
            w_post_jk=w_post_jk,
            theta_k=theta_k,
            theta_j_mean=theta_j_mean,
            tau=tau,
            prob_K=prob_K,
        )
        gamma_br_tau, gamma_br_prob = self._compute_gamma_brackets(
            a_i=a_i,
            b_i=b_i,
            gamma_j=gamma_j,
            kappa_j=kappa_j,
            c_i=c_i,
            X_ij=X_ij,
            w_post_jk=w_post_jk,
            theta_k=theta_k,
            theta_j_mean=theta_j_mean,
            tau=tau,
            prob_K=prob_K,
        )
        kappa_br_tau, kappa_br_prob = self._compute_kappa_brackets(
            a_i=a_i,
            b_i=b_i,
            gamma_j=gamma_j,
            kappa_j=kappa_j,
            c_i=c_i,
            X_ij=X_ij,
            w_post_jk=w_post_jk,
            theta_k=theta_k,
            theta_j_mean=theta_j_mean,
            tau=tau,
            prob_K=prob_K,
            use_quadratic=self.cfg.use_quadratic,
        )

        info["prob_K"] = float(prob_K)

        return CalibrationResult(
            a_i=a_i,
            b_i=b_i,
            gamma_j=gamma_j,
            kappa_j=kappa_j,
            theta_j_mean=theta_j_mean,
            theta_j_se=theta_j_se,
            theta_bracket_half_tau=theta_br_tau,
            theta_bracket_half_ideal=theta_br_prob,
            b_bracket_half_tau=b_br_tau,
            b_bracket_half_ideal=b_br_prob,
            gamma_bracket_half_tau=gamma_br_tau,
            gamma_bracket_half_ideal=gamma_br_prob,
            kappa_bracket_half_tau=kappa_br_tau,
            kappa_bracket_half_ideal=kappa_br_prob,
            w_post_jk=w_post_jk,
            theta_k=theta_k,
            info=info,
        )

    # ---------- E-step ----------
    def _estep(
        self,
        *,
        Y_ij: NDArray,
        X_ij: NDArray,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        theta_k: NDArray,
        w_k: NDArray,
        use_quadratic: bool,
    ) -> Tuple[NDArray, float]:
        Y = np.asarray(Y_ij, float)   # (Q,N)
        X = np.asarray(X_ij, float)   # (Q,N)
        Q, N = Y.shape                # scalars
        K = theta_k.shape[0]          # scalar

        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        w_post_jk = np.empty((N, K), dtype=float)  # (N,K)
        ll_obs = 0.0                               # scalar
        log_wk = np.log(w_k + EPS)                 # (K,)

        if not use_quadratic:
            kappa_j = np.zeros_like(kappa_j)       # (N,)

        for j in range(N):
            x = X[:, j]                            # (Q,)
            x2 = x * x                             # (Q,)
            budget_i = a_i * (gamma_j[j] * x + kappa_j[j] * x2)  # (Q,)

            eta_ik = base_ik + budget_i[:, None]    # (Q,K)
            s_ik = expit(eta_ik)                    # (Q,K)
            p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik  # (Q,K)
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)    # (Q,K)

            y_i = Y[:, j][:, None]                  # (Q,1)
            ll_items_k = (y_i * np.log(p_ik) + (1.0 - y_i) * np.log(1.0 - p_ik)).sum(axis=0)  # (K,)

            log_post_k = log_wk + ll_items_k         # (K,)
            m = np.max(log_post_k)                   # scalar
            w = np.exp(log_post_k - m)               # (K,)
            z = np.sum(w)                            # scalar
            w_post_jk[j, :] = w / z                  # (K,)

            ll_obs += m + np.log(z)                  # scalar

        return w_post_jk, float(ll_obs)

    # ---------- M-step: items (a_i, b_i) ----------
    def _mstep_items(
        self,
        *,
        Y_ij: NDArray,
        X_ij: NDArray,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        theta_k: NDArray,
        w_post_jk: NDArray,
        use_quadratic: bool,
    ) -> Tuple[NDArray, NDArray]:
        Q, N = Y_ij.shape

        if not use_quadratic:
            kappa_j = np.zeros_like(kappa_j)  # (N,)

        for i in range(Q):
            y_j = Y_ij[i, :]                 # (N,)
            x_j = X_ij[i, :]                 # (N,)
            c = float(c_i[i])                # scalar

            x2_j = x_j * x_j                 # (N,)
            u_j = gamma_j * x_j + kappa_j * x2_j  # (N,)

            def obj_grad(params: NDArray) -> Tuple[float, NDArray]:
                a = float(params[0])         # scalar
                b = float(params[1])         # scalar

                base_k = a * (theta_k - b)                 # (K,)
                budget_j = a * u_j                         # (N,)
                eta_jk = base_k[None, :] + budget_j[:, None]  # (N,K)

                s_jk = expit(eta_jk)                        # (N,K)
                p_jk = c + (1.0 - c) * s_jk                 # (N,K)
                p_jk = np.clip(p_jk, EPS, 1.0 - EPS)        # (N,K)

                ll_jk = y_j[:, None] * np.log(p_jk) + (1.0 - y_j)[:, None] * np.log(1.0 - p_jk)  # (N,K)
                nll = -np.sum(w_post_jk * ll_jk)            # scalar

                g_jk = (1.0 - c) * s_jk * (1.0 - s_jk)      # (N,K)
                r_jk = (y_j[:, None] - p_jk) / (p_jk * (1.0 - p_jk))  # (N,K)

                deta_da_jk = (theta_k - b)[None, :] + u_j[:, None]    # (N,K)
                deta_db_jk = -a                                       # scalar

                wg = w_post_jk * r_jk * g_jk                          # (N,K)
                grad_a = -np.sum(wg * deta_da_jk) + 2.0 * self.cfg.ridge_a * (a - self.cfg.mu_a)  # scalar
                grad_b = -np.sum(wg * deta_db_jk) + 2.0 * self.cfg.ridge_b * (b - self.cfg.mu_b)  # scalar

                obj = nll + self.cfg.ridge_a * (a - self.cfg.mu_a) ** 2 + self.cfg.ridge_b * (b - self.cfg.mu_b) ** 2
                return float(obj), np.array([grad_a, grad_b], dtype=float)

            x0 = np.array([a_i[i], b_i[i]], dtype=float)
            bounds = [(self.cfg.a_min, self.cfg.a_max), (None, None)]
            res = minimize(
                obj_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 40, "ftol": 1e-6},
            )
            a_i[i], b_i[i] = res.x

        return a_i, b_i

    # ---------- M-step: model gain (gamma_j, kappa_j) ----------
    def _mstep_model_gain(
        self,
        *,
        Y_ij: NDArray,
        X_ij: NDArray,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        theta_k: NDArray,
        w_post_jk: NDArray,
        use_quadratic: bool,
    ) -> Tuple[NDArray, NDArray]:
        Y = np.asarray(Y_ij, float)  # (Q,N)
        X = np.asarray(X_ij, float)  # (Q,N)
        Q, N = Y.shape               # scalars

        mu_gamma = float(self.cfg.mu_gamma)   # scalar
        mu_kappa = float(self.cfg.mu_kappa)   # scalar
        rg = float(self.cfg.ridge_gamma)      # scalar
        rk = float(self.cfg.ridge_kappa)      # scalar

        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        kappa_bounds = (self.cfg.kappa_min, self.cfg.kappa_max) if use_quadratic else (0.0, 0.0)

        for j in range(N):
            y_i = Y[:, j]          # (Q,)
            x_i = X[:, j]          # (Q,)
            w_k = w_post_jk[j, :]  # (K,)

            x2_i = x_i * x_i       # (Q,)
            v1_i = a_i * x_i       # (Q,)
            v2_i = a_i * x2_i      # (Q,)

            if np.allclose(v1_i, 0.0) and np.allclose(v2_i, 0.0):
                continue
            
            if self.cfg.x0_j is not None:
                x0_j = self.cfg.x0_j[j] 
            else:
                x0_j = np.median(X[:, j][X[:, j] > 0])

            def obj_grad(params: NDArray) -> Tuple[float, NDArray]:
                gamma = float(params[0])      # scalar
                kappa = float(params[1])      # scalar

                eta_ik = base_ik + (gamma * v1_i + kappa * v2_i)[:, None]  # (Q,K)
                s_ik = expit(eta_ik)                                     # (Q,K)
                p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik         # (Q,K)
                p_ik = np.clip(p_ik, EPS, 1.0 - EPS)                      # (Q,K)

                ll_ik = y_i[:, None] * np.log(p_ik) + (1.0 - y_i)[:, None] * np.log(1.0 - p_ik)  # (Q,K)
                nll = -np.sum(w_k[None, :] * ll_ik)                         # scalar

                obj = nll + rg * (gamma - mu_gamma) ** 2 + rk * (kappa - mu_kappa) ** 2          # scalar

                g_ik = (1.0 - c_i)[:, None] * s_ik * (1.0 - s_ik)           # (Q,K)
                r_ik = (y_i[:, None] - p_ik) / (p_ik * (1.0 - p_ik))        # (Q,K)
                wg = w_k[None, :] * r_ik * g_ik                             # (Q,K)

                grad_gamma = -np.sum(wg * v1_i[:, None]) + 2.0 * rg * (gamma - mu_gamma)         # scalar
                grad_kappa = -np.sum(wg * v2_i[:, None]) + 2.0 * rk * (kappa - mu_kappa)         # scalar

                # simple modification to regularize optimal log-budget
                t = (gamma + 2.0*kappa*x0_j)
                alpha_x = self.cfg.ridge_x0
                obj += alpha_x * (t*t)
                grad_gamma += 2.0*alpha_x * t
                grad_kappa += 4.0*alpha_x * x0_j * t

                return float(obj), np.array([grad_gamma, grad_kappa], dtype=float)

            x0 = np.array([gamma_j[j], kappa_j[j] if use_quadratic else 0.0], dtype=float)
            bounds = [(self.cfg.gamma_min, self.cfg.gamma_max), kappa_bounds]
            res = minimize(
                obj_grad,
                x0,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 60, "ftol": 1e-6},
            )

            gamma_j[j] = float(np.clip(res.x[0], self.cfg.gamma_min, self.cfg.gamma_max))
            kappa_j[j] = float(np.clip(res.x[1], kappa_bounds[0], kappa_bounds[1]))

        return gamma_j, kappa_j

    # ---------- Identifiability pin ----------
    def _pin_identifiability(
        self,
        *,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        w_post_jk: NDArray,
        theta_k: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Standardize the latent theta scale and co-transform parameters."""
        theta_j_mean = w_post_jk @ theta_k            # (N,)
        theta2_j_mean = w_post_jk @ (theta_k ** 2)    # (N,)

        mu_theta = float(theta_j_mean.mean())
        var_theta = float(theta2_j_mean.mean() - mu_theta ** 2)
        sigma_theta = float(np.sqrt(max(var_theta, 1e-12)))

        a_i = a_i * sigma_theta
        b_i = (b_i - mu_theta) / sigma_theta
        gamma_j = gamma_j / sigma_theta
        if self.cfg.use_quadratic:
            kappa_j = kappa_j / sigma_theta
        else:
            kappa_j = np.zeros_like(kappa_j)

        if self.cfg.verbose:
            print(f"mu_theta = {mu_theta: .3f}, sigma_theta = {sigma_theta: .5f}")

        a_i = np.clip(a_i, self.cfg.a_min, self.cfg.a_max)
        gamma_j = np.clip(gamma_j, self.cfg.gamma_min, self.cfg.gamma_max)
        if self.cfg.use_quadratic:
            kappa_j = np.clip(kappa_j, self.cfg.kappa_min, self.cfg.kappa_max)

        return a_i, b_i, gamma_j, kappa_j

    # ---------- Bracket-style uncertainties ----------
    @staticmethod
    def _alpha_first_success(p: NDArray) -> NDArray:
        """alpha_k = P(first success at k), scanning from closest -> farthest."""
        p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
        K = p.shape[0]
        alpha = np.zeros(K, dtype=float)
        surv = 1.0
        for k in range(K):
            alpha[k] = surv * p[k]
            surv *= (1.0 - p[k])
        alpha[-1] += surv
        return alpha

    @staticmethod
    def _alpha_first_fail(p: NDArray) -> NDArray:
        """alpha_k = P(first failure at k), scanning from closest -> farthest."""
        p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
        K = p.shape[0]
        alpha = np.zeros(K, dtype=float)
        surv = 1.0
        for k in range(K):
            alpha[k] = surv * (1.0 - p[k])
            surv *= p[k]
        alpha[-1] += surv
        return alpha

    @staticmethod
    def _select_k_closest(
        vals: NDArray,
        probs: NDArray,
        *,
        center: float,
        side: str,
        k: int,
    ) -> Tuple[NDArray, NDArray]:
        """Pick k closest candidates on a given side, ordered from closest -> farthest."""
        vals = np.asarray(vals, float)
        probs = np.asarray(probs, float)
        if vals.size != probs.size:
            raise ValueError("vals and probs must have the same length")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        if k < 1:
            raise ValueError("k must be >= 1")

        if side == "left":
            mask = vals <= center
            dist = center - vals
        else:
            mask = vals >= center
            dist = vals - center

        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=float)

        v = vals[mask]
        p = probs[mask]
        d = dist[mask]

        k_eff = int(min(k, v.size))
        idx = np.argpartition(d, k_eff - 1)[:k_eff]
        ord_idx = idx[np.argsort(d[idx])]
        return v[ord_idx], p[ord_idx]

    @classmethod
    def _expected_halfwidth(
        cls,
        *,
        left_vals: NDArray,
        left_p: NDArray,
        right_vals: NDArray,
        right_p: NDArray,
        left_event: str,
        right_event: str,
    ) -> float:
        """Return 0.5*(E[R] - E[L]) using truncated probabilistic brackets."""
        if left_vals.size == 0 or right_vals.size == 0:
            return float("nan")

        if left_event == "success":
            aL = cls._alpha_first_success(left_p)
        elif left_event == "fail":
            aL = cls._alpha_first_fail(left_p)
        else:
            raise ValueError("left_event must be 'success' or 'fail'")

        if right_event == "success":
            aR = cls._alpha_first_success(right_p)
        elif right_event == "fail":
            aR = cls._alpha_first_fail(right_p)
        else:
            raise ValueError("right_event must be 'success' or 'fail'")

        EL = float(np.sum(aL * left_vals))
        ER = float(np.sum(aR * right_vals))
        hw = 0.5 * (ER - EL)
        return float(max(hw, 0.0))

    def _compute_theta_brackets(
        self,
        *,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        X_ij: NDArray,
        w_post_jk: NDArray,
        theta_k: NDArray,
        theta_j_mean: NDArray,
        tau: float,
        prob_K: int,
    ) -> Tuple[NDArray, NDArray]:
        Q, N = X_ij.shape
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        theta_half_tau = np.full(N, np.nan, dtype=float)
        theta_half_prob = np.full(N, np.nan, dtype=float)

        if not self.cfg.use_quadratic:
            kappa_j = np.zeros_like(kappa_j)

        for j in range(N):
            theta_hat = float(theta_j_mean[j])
            x = X_ij[:, j]
            x2 = x * x

            d_i = b_i - gamma_j[j] * x - kappa_j[j] * x2

            budget_i = a_i * (gamma_j[j] * x + kappa_j[j] * x2)         # (Q,)
            eta_ik = base_ik + budget_i[:, None]                        # (Q,K)
            s_ik = expit(eta_ik)
            p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            p_avg_i = p_ik @ w_post_jk[j, :]                            # (Q,)

            left_mask = d_i <= theta_hat
            right_mask = d_i >= theta_hat
            left_tau_mask = left_mask & (p_avg_i >= tau)
            right_tau_mask = right_mask & (p_avg_i <= 1.0 - tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L = float(np.max(d_i[left_tau_mask]))
                R = float(np.min(d_i[right_tau_mask]))
                theta_half_tau[j] = 0.5 * (R - L)

            L_vals, L_p = self._select_k_closest(d_i, p_avg_i, center=theta_hat, side="left", k=prob_K)
            R_vals, R_p = self._select_k_closest(d_i, p_avg_i, center=theta_hat, side="right", k=prob_K)
            theta_half_prob[j] = self._expected_halfwidth(
                left_vals=L_vals,
                left_p=L_p,
                right_vals=R_vals,
                right_p=R_p,
                left_event="success",
                right_event="fail",
            )

        return theta_half_tau, theta_half_prob

    def _compute_b_brackets(
        self,
        *,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        X_ij: NDArray,
        w_post_jk: NDArray,
        theta_k: NDArray,
        theta_j_mean: NDArray,
        tau: float,
        prob_K: int,
    ) -> Tuple[NDArray, NDArray]:
        Q, N = X_ij.shape
        b_half_tau = np.full(Q, np.nan, dtype=float)
        b_half_prob = np.full(Q, np.nan, dtype=float)

        if not self.cfg.use_quadratic:
            kappa_j = np.zeros_like(kappa_j)

        for i in range(Q):
            b_hat = float(b_i[i])
            x_j = X_ij[i, :]
            x2_j = x_j * x_j

            tilde_theta_j = theta_j_mean + gamma_j * x_j + kappa_j * x2_j

            a = float(a_i[i]); b = float(b_i[i]); c = float(c_i[i])
            base_k = a * (theta_k - b)                                        # (K,)
            budget_j = a * (gamma_j * x_j + kappa_j * x2_j)                   # (N,)
            eta_jk = base_k[None, :] + budget_j[:, None]                      # (N,K)
            s_jk = expit(eta_jk)
            p_jk = c + (1.0 - c) * s_jk
            p_jk = np.clip(p_jk, EPS, 1.0 - EPS)
            p_avg_j = np.sum(w_post_jk * p_jk, axis=1)                        # (N,)

            left_mask = tilde_theta_j <= b_hat
            right_mask = tilde_theta_j >= b_hat
            left_tau_mask = left_mask & (p_avg_j <= 1.0 - tau)
            right_tau_mask = right_mask & (p_avg_j >= tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L = float(np.max(tilde_theta_j[left_tau_mask]))
                R = float(np.min(tilde_theta_j[right_tau_mask]))
                b_half_tau[i] = 0.5 * (R - L)

            L_vals, L_p = self._select_k_closest(tilde_theta_j, p_avg_j, center=b_hat, side="left", k=prob_K)
            R_vals, R_p = self._select_k_closest(tilde_theta_j, p_avg_j, center=b_hat, side="right", k=prob_K)
            b_half_prob[i] = self._expected_halfwidth(
                left_vals=L_vals,
                left_p=L_p,
                right_vals=R_vals,
                right_p=R_p,
                left_event="fail",
                right_event="success",
            )

        return b_half_tau, b_half_prob

    def _compute_gamma_brackets(
        self,
        *,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        X_ij: NDArray,
        w_post_jk: NDArray,
        theta_k: NDArray,
        theta_j_mean: NDArray,
        tau: float,
        prob_K: int,
    ) -> Tuple[NDArray, NDArray]:
        Q, N = X_ij.shape
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        gamma_half_tau = np.full(N, np.nan, dtype=float)
        gamma_half_prob = np.full(N, np.nan, dtype=float)

        if not self.cfg.use_quadratic:
            kappa_j = np.zeros_like(kappa_j)

        for j in range(N):
            theta_hat = float(theta_j_mean[j])
            ghat = float(gamma_j[j])
            x = X_ij[:, j]
            x2 = x * x
            valid = x > 0.0
            if not np.any(valid):
                continue

            gamma_star = (b_i[valid] - theta_hat - kappa_j[j] * x2[valid]) / x[valid]

            budget_i = a_i[valid] * (ghat * x[valid] + kappa_j[j] * x2[valid])
            eta_ik = base_ik[valid, :] + budget_i[:, None]
            s_ik = expit(eta_ik)
            p_ik = c_i[valid, None] + (1.0 - c_i[valid, None]) * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            p_avg_i = p_ik @ w_post_jk[j, :]

            left_mask = gamma_star <= ghat
            right_mask = gamma_star >= ghat

            left_tau_mask = left_mask & (p_avg_i >= tau)
            right_tau_mask = right_mask & (p_avg_i <= 1.0 - tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L = float(np.max(gamma_star[left_tau_mask]))
                R = float(np.min(gamma_star[right_tau_mask]))
                gamma_half_tau[j] = 0.5 * (R - L)

            L_vals, L_p = self._select_k_closest(gamma_star, p_avg_i, center=ghat, side="left", k=prob_K)
            R_vals, R_p = self._select_k_closest(gamma_star, p_avg_i, center=ghat, side="right", k=prob_K)
            gamma_half_prob[j] = self._expected_halfwidth(
                left_vals=L_vals,
                left_p=L_p,
                right_vals=R_vals,
                right_p=R_p,
                left_event="success",
                right_event="fail",
            )

        return gamma_half_tau, gamma_half_prob

    def _compute_kappa_brackets(
        self,
        *,
        a_i: NDArray,
        b_i: NDArray,
        gamma_j: NDArray,
        kappa_j: NDArray,
        c_i: NDArray,
        X_ij: NDArray,
        w_post_jk: NDArray,
        theta_k: NDArray,
        theta_j_mean: NDArray,
        tau: float,
        prob_K: int,
        use_quadratic: bool,
    ) -> Tuple[NDArray, NDArray]:
        Q, N = X_ij.shape
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        kappa_half_tau = np.full(N, np.nan, dtype=float)
        kappa_half_prob = np.full(N, np.nan, dtype=float)

        if not use_quadratic:
            return kappa_half_tau, kappa_half_prob

        for j in range(N):
            theta_hat = float(theta_j_mean[j])
            khat = float(kappa_j[j])
            x = X_ij[:, j]
            x2 = x * x
            valid = x2 > 0.0
            if not np.any(valid):
                continue

            kappa_star = (b_i[valid] - theta_hat - gamma_j[j] * x[valid]) / x2[valid]

            budget_i = a_i[valid] * (gamma_j[j] * x[valid] + khat * x2[valid])
            eta_ik = base_ik[valid, :] + budget_i[:, None]
            s_ik = expit(eta_ik)
            p_ik = c_i[valid, None] + (1.0 - c_i[valid, None]) * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            p_avg_i = p_ik @ w_post_jk[j, :]

            left_mask = kappa_star <= khat
            right_mask = kappa_star >= khat

            left_tau_mask = left_mask & (p_avg_i >= tau)
            right_tau_mask = right_mask & (p_avg_i <= 1.0 - tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L = float(np.max(kappa_star[left_tau_mask]))
                R = float(np.min(kappa_star[right_tau_mask]))
                kappa_half_tau[j] = 0.5 * (R - L)

            L_vals, L_p = self._select_k_closest(kappa_star, p_avg_i, center=khat, side="left", k=prob_K)
            R_vals, R_p = self._select_k_closest(kappa_star, p_avg_i, center=khat, side="right", k=prob_K)
            kappa_half_prob[j] = self._expected_halfwidth(
                left_vals=L_vals,
                left_p=L_p,
                right_vals=R_vals,
                right_p=R_p,
                left_event="success",
                right_event="fail",
            )

        return kappa_half_tau, kappa_half_prob

    # ---------- Utilities ----------
    def _prepare_inputs(self, data: CalibrationData, strict_Y_int: bool = True) -> Tuple[NDArray, NDArray, NDArray]:
        Y = np.asarray(data.Y_ij)
        B = np.asarray(data.B_ij)

        if Y.ndim != 2 or B.ndim != 2:
            raise ValueError(f"Y_ij and B_ij must be 2D arrays; got shapes {Y.shape} and {B.shape}")
        if Y.shape != B.shape:
            raise ValueError(f"Y_ij and B_ij must have the same shape; got {Y.shape} vs {B.shape}")

        if not np.isfinite(Y).all():
            raise ValueError("Y_ij contains non-finite values")
        if not np.isfinite(B).all():
            raise ValueError("B_ij contains non-finite values")

        if strict_Y_int:
            if not np.isin(Y, [0, 1]).all():
                bad = np.unique(Y[~np.isin(Y, [0, 1])])
                raise ValueError(f"Y_ij must be binary in {{0,1}}; found {bad[:10]}")

        Y_ij = Y.astype(float)

        if (B < 0).any():
            raise ValueError("B_ij must be nonnegative")
        X_ij = np.log1p(B.astype(float))
        if not np.isfinite(X_ij).all():
            raise ValueError("X_ij = log1p(B_ij) produced non-finite values")

        Q, _ = Y_ij.shape
        c_raw = data.c_i
        if np.isscalar(c_raw):
            c_i = np.full(Q, float(c_raw), dtype=float)
        else:
            c_i = np.asarray(c_raw, dtype=float).reshape(-1)
            if c_i.shape[0] != Q:
                raise ValueError(f"c_i must be scalar or length Q={Q}; got shape {c_i.shape}")

        if not np.isfinite(c_i).all():
            raise ValueError("c_i contains non-finite values")
        c_i = np.clip(c_i, 0.0, 1.0 - EPS)

        return Y_ij, X_ij, c_i

    def _make_gh(self, K: int) -> Tuple[NDArray, NDArray]:
        x_k, w_k = hermgauss(K)
        theta_k = np.sqrt(2.0) * x_k
        w_k = w_k / np.sqrt(np.pi)
        return theta_k.astype(float), w_k.astype(float)

    def _init_params(
        self,
        Y_ij: NDArray,
        c_i: NDArray,
        *,
        use_quadratic: bool,
        optimal_kappa: bool = False,
        eps: float = 1e-3 # local EPS to avoid extreme init of b_i
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        Y = np.asarray(Y_ij, float)          # (Q,N)
        Q, N = Y.shape                       # scalars

        # Model ref: p_ij ≈ c_i + (1−c_i)·σ(theta − b_i)        
        p_i = np.mean(Y, axis=1)             # (Q,)
        denom = np.clip(1.0 - c_i, eps, 1.0) # (Q,)
        s_i = (p_i - c_i) / denom            # (Q,)

        # fix cases where c_i is abnormally large
        s_i[s_i <= 0] = p_i[s_i <= 0]

        s_i = np.clip(s_i, eps, 1.0 - eps)   # (Q,)
        logit_s = np.log(s_i) - np.log(1.0 - s_i)  # (Q,)

        a_i = np.full(Q, float(self.cfg.mu_a), dtype=float)
        a_i = np.clip(a_i, self.cfg.a_min, self.cfg.a_max)

        b_i = (-logit_s / a_i).astype(float)

        gamma_j = np.full(N, float(self.cfg.mu_gamma), dtype=float)
        gamma_j = np.clip(gamma_j, self.cfg.gamma_min, self.cfg.gamma_max)

        if use_quadratic:
            if (not optimal_kappa) or (self.cfg.x0_j is None):
                kappa_j = np.full(N, float(self.cfg.mu_kappa), dtype=float)
            else:
                kappa_j = -gamma_j / (2*self.cfg.x0_j)
            kappa_j = np.clip(kappa_j, self.cfg.kappa_min, self.cfg.kappa_max)
        else:
            kappa_j = np.zeros(N, dtype=float)

        return a_i, b_i, gamma_j, kappa_j

    def _summarize_theta(self, w_post_jk: NDArray, theta_k: NDArray) -> Tuple[NDArray, NDArray]:
        w = np.asarray(w_post_jk, float)
        theta = np.asarray(theta_k, float)

        mean = (w * theta[None, :]).sum(axis=1)
        second = (w * (theta[None, :] ** 2)).sum(axis=1)
        var = np.maximum(second - mean * mean, 0.0)
        se = np.sqrt(var)
        return mean.astype(float), se.astype(float)


# ---------- Post-hoc External Utility ----------

def compute_brackets(
    *,
    result: CalibrationResult,
    data: CalibrationData | None = None,
    tau: float | None = None,
    prob_K: int = 3,
    which: Iterable[str] = ("theta", "b", "gamma", "kappa"),
    return_result: bool = True,
) -> CalibrationResult | Dict[str, NDArray]:
    """Compute bracket arrays post-hoc."""
    if data is None:
        if getattr(result, "info", None) and "input_data" in result.info:
            data = result.info["input_data"]
        else:
            raise ValueError("data must be provided (or stored in result.info['input_data'])")

    Y = np.asarray(data.Y_ij, float)
    X = np.log1p(np.asarray(data.B_ij, float))

    a_i = np.asarray(result.a_i, float)
    b_i = np.asarray(result.b_i, float)
    gamma_j = np.asarray(result.gamma_j, float)
    kappa_j = np.asarray(result.kappa_j, float)
    theta_k = np.asarray(result.theta_k, float)
    w_post_jk = np.asarray(result.w_post_jk, float)
    theta_j_mean = np.asarray(result.theta_j_mean, float)

    Q = Y.shape[0]
    if np.isscalar(data.c_i):
        c_i = np.full(Q, float(data.c_i), dtype=float)
    else:
        c_i = np.asarray(data.c_i, float).reshape(-1)

    if tau is None:
        tau = float(result.info.get("tau", 0.75)) if getattr(result, "info", None) else 0.75

    use_quadratic = bool(result.info.get("use_quadratic", True)) if getattr(result, "info", None) else True

    cfg = CalibrationConfig(use_quadratic=use_quadratic, tau=float(tau), verbose=False)
    cal = ReasoningAware3PLQuadraticCalibrator(cfg)

    out: Dict[str, NDArray] = {}

    if "theta" in which:
        t_tau, t_prob = cal._compute_theta_brackets(
            a_i=a_i, b_i=b_i, gamma_j=gamma_j, kappa_j=kappa_j, c_i=c_i,
            X_ij=X, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean,
            tau=float(tau), prob_K=int(prob_K),
        )
        out["theta_half_tau"] = t_tau
        out["theta_half_prob"] = t_prob

    if "b" in which:
        b_tau, b_prob = cal._compute_b_brackets(
            a_i=a_i, b_i=b_i, gamma_j=gamma_j, kappa_j=kappa_j, c_i=c_i,
            X_ij=X, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean,
            tau=float(tau), prob_K=int(prob_K),
        )
        out["b_half_tau"] = b_tau
        out["b_half_prob"] = b_prob

    if "gamma" in which:
        g_tau, g_prob = cal._compute_gamma_brackets(
            a_i=a_i, b_i=b_i, gamma_j=gamma_j, kappa_j=kappa_j, c_i=c_i,
            X_ij=X, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean,
            tau=float(tau), prob_K=int(prob_K),
        )
        out["gamma_half_tau"] = g_tau
        out["gamma_half_prob"] = g_prob

    if "kappa" in which:
        k_tau, k_prob = cal._compute_kappa_brackets(
            a_i=a_i, b_i=b_i, gamma_j=gamma_j, kappa_j=kappa_j, c_i=c_i,
            X_ij=X, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean,
            tau=float(tau), prob_K=int(prob_K), use_quadratic=use_quadratic,
        )
        out["kappa_half_tau"] = k_tau
        out["kappa_half_prob"] = k_prob

    if not return_result:
        return out

    if "theta_half_tau" in out:
        result.theta_bracket_half_tau = out["theta_half_tau"]
        result.theta_bracket_half_ideal = out["theta_half_prob"]
    if "b_half_tau" in out:
        result.b_bracket_half_tau = out["b_half_tau"]
        result.b_bracket_half_ideal = out["b_half_prob"]
    if "gamma_half_tau" in out:
        result.gamma_bracket_half_tau = out["gamma_half_tau"]
        result.gamma_bracket_half_ideal = out["gamma_half_prob"]
    if "kappa_half_tau" in out:
        result.kappa_bracket_half_tau = out["kappa_half_tau"]
        result.kappa_bracket_half_ideal = out["kappa_half_prob"]

    if getattr(result, "info", None) is not None:
        result.info["tau"] = float(tau)
        result.info["prob_K"] = float(prob_K)

    return result


def compute_louis_se(
    *,
    result: CalibrationResult,
    data: CalibrationData | None = None,
    which: tuple[str, ...] = ("a", "b", "gamma", "kappa"),
    return_result: bool = False,
    jitter: float = 1e-9,
) -> Dict[str, NDArray] | CalibrationResult:
    """Louis-method observed-information SEs (post-hoc).

    Computes SEs for:
      - b_i (via the 2x2 (a_i,b_i) block, then extracts SE for b_i)
      - gamma_j (via 2x2 (gamma_j,kappa_j) block if quadratic; scalar if linear)

    Notes
    -----
    - This uses the likelihood only (ignores ridge penalties used in M-steps).
    - `return_result=True` stores arrays into result.info and returns `result`.
    """
    if data is None:
        if getattr(result, "info", None) and "input_data" in result.info:
            data = result.info["input_data"]
        else:
            raise ValueError("data must be provided (or stored in result.info['input_data'])")

    Y = np.asarray(data.Y_ij, float)                  # (Q,N)
    X = np.log1p(np.asarray(data.B_ij, float))        # (Q,N)

    a_i = np.asarray(result.a_i, float)              # (Q,)
    b_i = np.asarray(result.b_i, float)              # (Q,)
    gamma_j = np.asarray(result.gamma_j, float)      # (N,)
    kappa_j = np.asarray(result.kappa_j, float)      # (N,)

    theta_k = np.asarray(result.theta_k, float)      # (K,)
    w_post_jk = np.asarray(result.w_post_jk, float)  # (N,K)

    Q, N = Y.shape
    K = theta_k.shape[0]

    if np.isscalar(data.c_i):
        c_i = np.full(Q, float(data.c_i), dtype=float)
    else:
        c_i = np.asarray(data.c_i, float).reshape(-1)

    use_quadratic = bool(result.info.get("use_quadratic", True)) if getattr(result, "info", None) else True

    def _t_u(y: NDArray, s: NDArray, p: NDArray, c: float) -> Tuple[NDArray, NDArray]:
        """Return (t,u) where t = d l / d eta, u = d^2 l / d eta^2."""
        p = np.clip(p, EPS, 1.0 - EPS)
        g = p * (1.0 - p)
        A = (1.0 - c) * s * (1.0 - s)
        B = (1.0 - c) * s * (1.0 - s) * (1.0 - 2.0 * s)

        D = (y - p) / g
        dD_dp = (-g - (y - p) * (1.0 - 2.0 * p)) / (g * g)
        t = D * A
        u = dD_dp * (A * A) + D * B
        return t, u

    out: Dict[str, NDArray] = {}

    # -------- b_i SEs (via (a_i,b_i) 2x2 block) --------
    if "b" in which or "a" in which:
        b_se = np.full(Q, np.nan, dtype=float)
        a_se = np.full(Q, np.nan, dtype=float)

        for i in range(Q):
            a = float(a_i[i])
            b = float(b_i[i])
            c = float(c_i[i])

            y_j = Y[i, :]                 # (N,)
            x_j = X[i, :]                 # (N,)
            x2_j = x_j * x_j              # (N,)

            if not use_quadratic:
                u_j = gamma_j * x_j
            else:
                u_j = gamma_j * x_j + kappa_j * x2_j

            q_jk = (theta_k[None, :] - b) + u_j[:, None]       # (N,K)
            eta_jk = a * q_jk                                   # (N,K)

            s_jk = expit(eta_jk)                                # (N,K)
            p_jk = c + (1.0 - c) * s_jk                         # (N,K)

            y_jk = y_j[:, None]                                 # (N,1)
            t_jk, u_jk = _t_u(y_jk, s_jk, p_jk, c)              # (N,K), (N,K)

            ga_jk = t_jk * q_jk                                 # (N,K)
            gb_jk = t_jk * (-a)                                 # (N,K)

            H_aa = u_jk * (q_jk * q_jk)                          # (N,K)
            H_ab = u_jk * (q_jk * (-a)) + t_jk * (-1.0)          # (N,K)
            H_bb = u_jk * (a * a)                                # (N,K)

            # Aggregate observed information across respondents (latent variables are independent across j)
            I = np.zeros((2, 2), dtype=float)

            for j in range(N):
                w = w_post_jk[j, :]                              # (K,)

                Es_a = float(np.dot(w, ga_jk[j, :]))
                Es_b = float(np.dot(w, gb_jk[j, :]))

                Ess_aa = float(np.dot(w, ga_jk[j, :] * ga_jk[j, :]))
                Ess_ab = float(np.dot(w, ga_jk[j, :] * gb_jk[j, :]))
                Ess_bb = float(np.dot(w, gb_jk[j, :] * gb_jk[j, :]))

                EH_aa = float(np.dot(w, H_aa[j, :]))
                EH_ab = float(np.dot(w, H_ab[j, :]))
                EH_bb = float(np.dot(w, H_bb[j, :]))

                Cov_aa = Ess_aa - Es_a * Es_a
                Cov_ab = Ess_ab - Es_a * Es_b
                Cov_bb = Ess_bb - Es_b * Es_b

                I[0, 0] += -EH_aa - Cov_aa
                I[0, 1] += -EH_ab - Cov_ab
                I[1, 0] += -EH_ab - Cov_ab
                I[1, 1] += -EH_bb - Cov_bb

            # Invert 2x2 block
            I[0, 0] += jitter
            I[1, 1] += jitter
            try:
                cov = np.linalg.inv(I)
                a_se[i] = float(np.sqrt(max(cov[0, 0], 0.0)))
                b_se[i] = float(np.sqrt(max(cov[1, 1], 0.0)))
            except np.linalg.LinAlgError:
                a_se[i] = float("nan")
                b_se[i] = float("nan")

        if "a" in which:
            out["a_se_louis"] = a_se
        if "b" in which:
            out["b_se_louis"] = b_se

    # -------- gamma_j SEs --------
    if "gamma" in which or "kappa" in which:
        gamma_se = np.full(N, np.nan, dtype=float)
        kappa_se = np.full(N, np.nan, dtype=float)

        for j in range(N):
            y_i = Y[:, j]              # (Q,)
            x_i = X[:, j]              # (Q,)
            x2_i = x_i * x_i           # (Q,)

            # If there is no budget variation, gamma is not identified
            if np.allclose(x_i, 0.0):
                continue

            g = float(gamma_j[j])
            k = float(kappa_j[j]) if use_quadratic else 0.0

            q_ik = (theta_k[None, :] - b_i[:, None]) + g * x_i[:, None] + k * x2_i[:, None]  # (Q,K)
            eta_ik = a_i[:, None] * q_ik                                                     # (Q,K)

            s_ik = expit(eta_ik)
            # c_i broadcast: (Q,1)
            p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik

            y_ik = y_i[:, None]
            # Here we pass a vector c_i? _t_u expects scalar c; do it elementwise via broadcasting:
            # Implement t,u directly with c_i[:,None]
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            gden = p_ik * (1.0 - p_ik)
            A = (1.0 - c_i)[:, None] * s_ik * (1.0 - s_ik)
            B = (1.0 - c_i)[:, None] * s_ik * (1.0 - s_ik) * (1.0 - 2.0 * s_ik)
            D = (y_ik - p_ik) / gden
            dD_dp = (-gden - (y_ik - p_ik) * (1.0 - 2.0 * p_ik)) / (gden * gden)
            t_ik = D * A
            u_ik = dD_dp * (A * A) + D * B

            v1_i = a_i * x_i

            w = w_post_jk[j, :]  # (K,)

            if not use_quadratic:
                # scalar gamma
                s_k = np.sum(t_ik * v1_i[:, None], axis=0)               # (K,)
                h_k = np.sum(u_ik * (v1_i[:, None] ** 2), axis=0)         # (K,)

                Es = float(np.dot(w, s_k))
                Ess = float(np.dot(w, s_k * s_k))
                EH = float(np.dot(w, h_k))

                Cov = Ess - Es * Es
                I = -EH - Cov
                if I > 0:
                    gamma_se[j] = float(np.sqrt(1.0 / I))
                else:
                    gamma_se[j] = float("nan")
                kappa_se[j] = float("nan") # no kappa_se in linear-budget model

            else:
                v2_i = a_i * x2_i

                s1_k = np.sum(t_ik * v1_i[:, None], axis=0)               # (K,)
                s2_k = np.sum(t_ik * v2_i[:, None], axis=0)               # (K,)

                h11_k = np.sum(u_ik * (v1_i[:, None] ** 2), axis=0)        # (K,)
                h12_k = np.sum(u_ik * (v1_i * v2_i)[:, None], axis=0)      # (K,)
                h22_k = np.sum(u_ik * (v2_i[:, None] ** 2), axis=0)        # (K,)

                Es1 = float(np.dot(w, s1_k))
                Es2 = float(np.dot(w, s2_k))
 
                Ess11 = float(np.dot(w, s1_k * s1_k))
                Ess12 = float(np.dot(w, s1_k * s2_k))
                Ess22 = float(np.dot(w, s2_k * s2_k))

                EH11 = float(np.dot(w, h11_k))
                EH12 = float(np.dot(w, h12_k))
                EH22 = float(np.dot(w, h22_k))

                Cov11 = Ess11 - Es1 * Es1
                Cov12 = Ess12 - Es1 * Es2
                Cov22 = Ess22 - Es2 * Es2

                I = np.array(
                    [[-EH11 - Cov11, -EH12 - Cov12],
                     [-EH12 - Cov12, -EH22 - Cov22]],
                    dtype=float,
                )
                I[0, 0] += jitter
                I[1, 1] += jitter

                try:
                    cov = np.linalg.inv(I)
                    gamma_se[j] = float(np.sqrt(max(cov[0, 0], 0.0)))
                    kappa_se[j] = float(np.sqrt(max(cov[1, 1], 0.0)))
                except np.linalg.LinAlgError:
                    gamma_se[j] = float("nan")
                    kappa_se[j] = float("nan")
        if "gamma" in which:
            out["gamma_se_louis"] = gamma_se
        if "kappa" in which:
            out["kappa_se_louis"] = kappa_se

    if not return_result:
        return out

    if getattr(result, "info", None) is not None:
        for k, v in out.items():
            result.info[k] = v
        result.info["louis_se_ignores_ridge"] = True

    return result
