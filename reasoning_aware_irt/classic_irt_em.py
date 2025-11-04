"""
Classic 3PL IRT — EM Calibrator (skeleton)

Model:
    eta_{ik} = a_i * (theta_k - b_i)
    s_{ik}   = sigmoid(eta_{ik})
    p_{ik}   = c_i + (1 - c_i) * s_{ik}

We approximate each respondent's posterior over theta by Gauss–Hermite (GH) quadrature
nodes (standard normal prior). The E-step forms w_post_{jk} and posterior moments.
M-steps update item blocks (a_i, b_i) via L-BFGS-B
minimization of the expected complete-data log-likelihood Q.

Notes:
- Keep Y in shape (Q, N). No budgets here.
- Clean failure on NaNs to keep logic simple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logit

EPS = 1e-12

# ---------------------------- Data / Config / Result ----------------------------

@dataclass
class ClassicCalibrationConfig:
    gh_nodes: int = 21                 # number of GH nodes (odd, e.g. 21)
    max_iter: int = 200
    tol: float = 1e-4                   # absolute tolerance on Q improvement
    a_min: float = 0.05                 # lower bound for a_i (avoid flat/slopes)
    a_max: Optional[float] = None       # optional upper bound for a_i
    ridge_a: float = 1e-3                # L2 penalty weight on a_i (stability)
    ridge_b: float = 1e-3                # L2 penalty weight on b_i
    verbose: bool = True
    random_state: Optional[int] = None


@dataclass
class ClassicCalibrationData:
    Y_ij: NDArray   # (Q, N) binary matrix
    c_i: NDArray | float  # (Q,)


@dataclass
class ClassicCalibrationResult:
    a_i: NDArray                  # (Q,)
    b_i: NDArray                  # (Q,)
    theta_j_mean: NDArray         # (N,) posterior means
    theta_j_se: NDArray           # (N,) posterior std (from GH mixture)
    w_post_jk: NDArray            # (N, K) posterior weights on GH nodes
    theta_k: NDArray              # (K,) GH nodes (std normal)
    info: Dict[str, float | int | str]


# ---------------------------- Utilities ----------------------------

def hermgauss_std_normal(n: int) -> Tuple[NDArray, NDArray]:
    """GH nodes/weights for standard normal prior N(0,1).
    Returns (theta, weights) where sum(weights) = 1.
    """
    x, w = hermgauss(n)
    theta = np.sqrt(2.0) * x
    weights = (w / np.sqrt(np.pi)).astype(np.float64)
    weights = weights / np.sum(weights)
    return theta.astype(np.float64), weights


# ---------------------------- Calibrator ----------------------------

class IRT3PLCalibrator:
    """EM-style estimator for classic 3PL (no budget terms).

    Structure mirrors the Reasoning‑aware GEM:
      - prepare_inputs
      - init_params
      - e_step_theta_posterior
      - m_step_items_ab
      - (optional) m_step_items_c
      - fit (main loop)
    """

    def __init__(self, config: ClassicCalibrationConfig | None = None) -> None:
        self.cfg = config or ClassicCalibrationConfig()
        if self.cfg.random_state is not None:
            np.random.seed(self.cfg.random_state)

    # ---------- Prepare ----------
    def prepare_inputs(self, data: ClassicCalibrationData) -> Tuple[NDArray, NDArray, NDArray]:
        """Validate Y, set c_i, create GH nodes and prior weights.
        Returns (Y_ij, theta_k, prior_w_k).
        """
        Y = np.asarray(data.Y_ij, dtype=float)
        if Y.ndim != 2:
            raise ValueError("Y_ij must be 2D (Q,N)")
        if np.isnan(Y).any():
            raise ValueError("Y_ij contains NaN — please clean or impute.")
        c = np.asarray(data.c_i, float)

        # GH nodes/weights
        K = int(self.cfg.gh_nodes)
        theta_k, prior_w_k = hermgauss_std_normal(K)
        return Y, c, theta_k, prior_w_k

    # ---------- Init ----------
    def init_params(self, Y_ij: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """Simple, data-informed initialization of (a_i, b_i).
        b_i via logit of item mean; a_i start at 1 with jitter;
        """
        Q, N = Y_ij.shape
        # Item means with guards
        p_i = np.clip(Y_ij.mean(axis=1), 1e-3, 1 - 1e-3)
        b_i = -logit(p_i)  # since E[theta]=0 at start, approximate b_i ≈ -logit(p_i)

        a_i = np.ones(Q, dtype=float)
        a_i += 0.05 * np.random.randn(Q)
        a_i = np.clip(a_i, self.cfg.a_min, self.cfg.a_max or np.inf)

        return a_i, b_i

    # ---------- E-step ----------
    def e_step_theta_posterior(
        self,
        Y_ij: NDArray, a_i: NDArray, b_i: NDArray, c_i: NDArray,
        theta_k: NDArray, prior_w_k: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute GH posterior weights w_post_{jk}, along with posterior mean/SD per respondent.
        Returns (w_post_jk, theta_j_mean, theta_j_se).
        """
        Q, N = Y_ij.shape
        K = theta_k.shape[0]

        # Likelihood on the grid (Q,N,K) via broadcasting: p_{ik} is (Q,K)
        eta_ik = (a_i[:, None]) * (theta_k[None, :] - b_i[:, None])          # (Q,K)
        s_ik = expit(eta_ik)
        p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik                    # (Q,K)
        p_ik = np.clip(p_ik, EPS, 1.0 - EPS)

        log_p_ijk = (Y_ij[:, :, None]) * np.log(p_ik[:, None, :]) \
                    + (1.0 - Y_ij[:, :, None]) * np.log(1.0 - p_ik[:, None, :])  # (Q,N,K)
        # Sum over items i → respondent-by-node log-likelihood
        ll_jk = np.sum(log_p_ijk, axis=0)                                     # (N,K)

        # Add log prior weight per node
        ll_jk = ll_jk + np.log(prior_w_k[None, :])

        # Normalize to get w_post
        ll_jk -= ll_jk.max(axis=1, keepdims=True)
        w_post_jk = np.exp(ll_jk)
        w_post_jk /= np.sum(w_post_jk, axis=1, keepdims=True)

        theta_j_mean = w_post_jk @ theta_k
        theta_j_m2 = w_post_jk @ (theta_k ** 2)
        theta_j_se = np.sqrt(np.maximum(theta_j_m2 - theta_j_mean ** 2, 0.0))
        return w_post_jk, theta_j_mean, theta_j_se

    # ---------- M-step: item (a_i, b_i) ----------
    def m_step_items_ab(
        self,
        Y_ij: NDArray, w_post_jk: NDArray, theta_k: NDArray,
        a_i: NDArray, b_i: NDArray, c_i: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Update (a_i, b_i) per item by maximizing Q = E_{w}[log L].
        Uses L-BFGS-B with simple bounds and ridge. Independent across items.
        """
        Qn, N = Y_ij.shape
        K = theta_k.shape[0]

        def obj_ab(x: NDArray, i: int) -> float:
            a = np.clip(x[0], self.cfg.a_min, self.cfg.a_max or np.inf)
            b = x[1]
            eta_k = a * (theta_k - b)
            s_k = expit(eta_k)
            p_k = np.clip(c_i[i] + (1.0 - c_i[i]) * s_k, EPS, 1.0 - EPS)
            # Expected log-likelihood for item i: sum over j,k of w_{jk} * log p(y_{ij}|k)
            # Precompute per-respondent weighted sums of log p_k and log(1-p_k)
            logp = np.log(p_k)
            logq = np.log(1.0 - p_k)
            # For efficiency: aggregate Y over respondents with weights per k
            # Q_i = sum_{j,k} w_{jk} [ y_{ij} log p_k + (1-y_{ij}) log(1-p_k) ]
            W_k = np.sum(w_post_jk, axis=0)               # (K,)
            WY_k = (Y_ij[i, :, None] * w_post_jk)         # (N,K)
            sum_WY_k = np.sum(WY_k, axis=0)               # (K,)
            Q_i = np.sum(sum_WY_k * logp + (W_k - sum_WY_k) * logq)
            # Ridge
            Q_i -= 0.5 * self.cfg.ridge_a * (a ** 2) + 0.5 * self.cfg.ridge_b * (b ** 2)
            return -float(Q_i)

        new_a = a_i.copy()
        new_b = b_i.copy()
        bounds = [(self.cfg.a_min, self.cfg.a_max or np.inf), (None, None)]
        for i in range(Qn):
            x0 = np.array([a_i[i], b_i[i]], dtype=float)
            res = minimize(obj_ab, x0, args=(i,), method="L-BFGS-B", bounds=bounds)
            ai_hat = np.clip(res.x[0], self.cfg.a_min, self.cfg.a_max or np.inf)
            bi_hat = res.x[1]
            new_a[i] = ai_hat
            new_b[i] = bi_hat
        return new_a, new_b

    # ---------- Q-function (monitor only) ----------
    def expected_complete_loglik(
        self,
        Y_ij: NDArray, w_post_jk: NDArray, theta_k: NDArray,
        a_i: NDArray, b_i: NDArray, c_i: NDArray,
    ) -> float:
        """Compute the expected complete-data log-likelihood (for convergence monitoring)."""
        eta_ik = (a_i[:, None]) * (theta_k[None, :] - b_i[:, None])  # (Q,K)
        s_ik = expit(eta_ik)
        p_ik = np.clip(c_i[:, None] + (1.0 - c_i)[:, None] * s_ik, EPS, 1.0 - EPS)
        logp = np.log(p_ik)
        logq = np.log(1.0 - p_ik)
        W_k = np.sum(w_post_jk, axis=0)                   # (K,)
        WY_k = Y_ij[:, :, None] * w_post_jk[None, :, :]   # (Q,N,K)
        sum_WY_k = np.sum(WY_k, axis=1)                   # (Q,K)
        Q_val = float(np.sum(sum_WY_k * logp + (W_k[None, :] - sum_WY_k) * logq))
        # Ridges
        Q_val -= 0.5 * self.cfg.ridge_a * float(np.sum(a_i ** 2))
        Q_val -= 0.5 * self.cfg.ridge_b * float(np.sum(b_i ** 2))
        return Q_val

    # ---------- Main fit ----------
    def fit(self, data: ClassicCalibrationData) -> ClassicCalibrationResult:
        Y_ij, c_i, theta_k, prior_w_k = self.prepare_inputs(data)
        Q, N = Y_ij.shape

        a_i, b_i = self.init_params(Y_ij)

        # Initial E-step
        w_post_jk, theta_j_mean, theta_j_se = self.e_step_theta_posterior(Y_ij, a_i, b_i, c_i, theta_k, prior_w_k)
        Q_prev = self.expected_complete_loglik(Y_ij, w_post_jk, theta_k, a_i, b_i, c_i)

        for it in range(1, self.cfg.max_iter + 1):
            # M-step: items
            a_i, b_i = self.m_step_items_ab(Y_ij, w_post_jk, theta_k, a_i, b_i, c_i)
            
            # E-step: theta posterior with updated items
            w_post_jk, theta_j_mean, theta_j_se = self.e_step_theta_posterior(Y_ij, a_i, b_i, c_i, theta_k, prior_w_k)

            # Convergence monitor
            Q_curr = self.expected_complete_loglik(Y_ij, w_post_jk, theta_k, a_i, b_i, c_i)
            if self.cfg.verbose:
                print(f"iter={it:3d}  LL={Q_curr:.6f}  dLL={Q_curr - Q_prev:.3e}")
            if abs(Q_curr - Q_prev) < self.cfg.tol:
                print(f"converged at iter {it}")
                break
            Q_prev = Q_curr

        info = dict(
            gh_nodes=int(self.cfg.gh_nodes),
            iterations=int(it),
            converged=bool(Q_curr - Q_prev < self.cfg.tol),
            objective=float(Q_curr),
        )

        return ClassicCalibrationResult(
            a_i=a_i, b_i=b_i,
            theta_j_mean=theta_j_mean, theta_j_se=theta_j_se,
            w_post_jk=w_post_jk, theta_k=theta_k, info=info,
        )