"""
Reasoning‑Aware 3PL
EM calibration for a 3PL IRT extended with a reasoning/budget term.

Model
-----
η_ij = a_i (θ_j − b_i) + a_i λ_i γ_j X_ij,
p_ij = c_i + (1 − c_i) σ(η_ij), with X_ij = log(1 + B_ij).

Cycle
-----
E‑step (posterior on θ via K‑point Gauss–Hermite) → M‑steps for (a,b), λ, γ (per‑block L‑BFGS‑B) → θ gauge pin.

Index‑revealing names (to aid readability):
  arrays shaped by item i (Q,): a_i, b_i, lambda_i, c_i
  arrays shaped by respondent j (N,): gamma_j, theta_j_mean, theta_j_se
  matrices shaped by (i,j) (Q,N): Y_ij, X_ij
  GH grid: theta_k (K,), w_k (K,); posteriors: w_post_jk (N,K)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logit
from scipy.optimize import minimize

EPS = 1e-6 # Numeric guardrails

# ------------------------------
# User‑facing configuration/data/result
# ------------------------------
@dataclass
class CalibrationConfig:
    # Inference controls
    gh_nodes: int = 17 # 21 # i.e. K
    max_iter: int = 50
    tol: float = 1e-4
    seed: Optional[int] = 0
    verbose: bool = True

    # Regularization (ridge_* * (param − mu_*)²)
    ridge_a: float = 1e-3
    ridge_b: float = 1e-3
    ridge_lambda: float = 0.1
    ridge_gamma: float = 0.1
    # Prior centers (mu_*) for {a_i, b_i, lambda_i, gamma_j}
    mu_a: float = 1.0
    mu_b: float = 0.0
    mu_lambda: float = 0.8
    mu_gamma: float = 0.2

    # Bounds / behavior
    a_min: float = 0.05
    a_max: float = 5.00
    gamma_min: float = -0.5
    gamma_max: float = 2.0 # NOTE: with our standard interpretation, gamma > 1 may be too much, but due to location-scale transform for identifiability, resulted scales may not be intuitive
    lambda_min: float = 0.01
    lambda_max: float = 0.999


@dataclass
class CalibrationData:
    """Observed data.
    Y_ij: (Q,N) binary responses with no NaNs allowed
    B_ij: (Q,N) nonnegative budgets (X_ij = log1p(B_ij) inside), no NaNs allowed
    c_i: scalar or (Q,) guessing per item
    """
    Y_ij: NDArray
    B_ij: NDArray
    c_i: NDArray | float


@dataclass
class CalibrationResult:
    a_i: NDArray                     # (Q,)
    b_i: NDArray                     # (Q,)
    lambda_i: NDArray                # (Q,) in [0,1]
    gamma_j: NDArray                 # (N,)
    theta_j_mean: NDArray            # (N,) posterior mean of theta_j
    theta_j_se: NDArray              # (N,) posterior std error of theta_j (from GH posterior)

    # Bracket-style intuitive uncertainties (half-widths in latent units)
    theta_bracket_half_tau: NDArray   # (N,) τ-bracket half-width for θ
    theta_bracket_half_ideal: NDArray # (N,) nearest-neighbor ideal half-width for θ
    b_bracket_half_tau: NDArray       # (Q,) τ-bracket half-width for b
    b_bracket_half_ideal: NDArray     # (Q,) nearest-neighbor ideal half-width for b
    gamma_bracket_half_tau: NDArray   # (N,) τ-bracket half-width for γ
    gamma_bracket_half_ideal: NDArray # (N,) nearest-neighbor ideal half-width for γ

    w_post_jk: NDArray               # (N,K)
    theta_k: NDArray                 # (K,)
    info: Dict[str, float | list[float]]


# ------------------------------
# Calibrator (GEM orchestrator)
# ------------------------------
class ReasoningAware3PLCalibrator:
    """Estimator for the reasoning‑aware 3PL model with absolute budgets.

    GEM main loop (per iteration):
      1) E‑step: posterior weights w_post_jk over GH nodes theta_k for each respondent.
      2) M‑items: update (a_i, b_i) independently for each item i.
      3) M‑item‑gate: update lambda_i in [0,1] (bounded SciPy minimize) per item i.
      4) M‑model‑gain: update gamma_j independently for each respondent j.
      5) Pin: identifiability (re‑center/scale theta_j; co‑transform a_i,b_i; rescale gamma_j to leave eta_ij invariant).
    """

    def __init__(self, cfg: CalibrationConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        # GH nodes/weights are created in _make_gh() when needed

    def fit(self, data: CalibrationData) -> CalibrationResult:
        """Run EM until convergence.

        Returns
        -------
        CalibrationResult
        Parameters, posterior nodes/weights, log‑lik trace, and θ summaries.
        """

        Y_ij, X_ij, c_i = self._prepare_inputs(data)
        Q, N = Y_ij.shape

        a_i, b_i, lambda_i, gamma_j = self._init_params(Y_ij, c_i)
        theta_k, w_k = self._make_gh(self.cfg.gh_nodes)

        # Storage for E‑step posteriors, init as priors
        w_post_jk = np.tile(w_k[None, :], (N, 1))  # (N,K)
        ll_trace: list[float] = []
        prev_ll = -np.inf

        for t in range(self.cfg.max_iter):
            # 1) E‑step
            w_post_jk, ll_obs = self._estep(
                Y_ij=Y_ij, X_ij=X_ij,
                a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
                theta_k=theta_k, w_k=w_k,
            )
            ll_trace.append(ll_obs)

            # 2) M‑items (a_i, b_i)
            a_i, b_i = self._mstep_items(
                Y_ij=Y_ij, X_ij=X_ij,
                a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
                theta_k=theta_k, w_post_jk=w_post_jk,
            )

            # 3) M‑item‑gate (lambda_i in [0,1])
            lambda_i = self._mstep_item_gate(
                Y_ij=Y_ij, X_ij=X_ij,
                a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
                theta_k=theta_k, w_post_jk=w_post_jk,
            )

            # 4) M‑model‑gain (gamma_j)
            gamma_j = self._mstep_model_gain(
                Y_ij=Y_ij, X_ij=X_ij,
                a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
                theta_k=theta_k, w_post_jk=w_post_jk,
            )

            # 5) Pin (identifiability)
            a_i, b_i, lambda_i, gamma_j = self._pin_identifiability(
                a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j,
                w_post_jk=w_post_jk, theta_k=theta_k,
            )

            if self.cfg.verbose:
                d = ll_trace[-1] - (ll_trace[-2] if t > 1 else np.nan)
                print(f"iter {t+1:3d}  ll={ll_obs:.6f}   dLL={d:.3e}")

            # Convergence
            if abs(ll_obs - prev_ll) < self.cfg.tol:
                print(f"converged at iter {t+1}: ll={ll_trace[-1]:.6f}")
                break
            prev_ll = ll_obs

        # Summaries
        theta_j_mean, theta_j_se = self._summarize_theta(w_post_jk, theta_k)
        info = {"iters": float(t + 1), "final_loglik": float(prev_ll),
                 "ll_trace": ll_trace, "theta_k": theta_k,
                 "input_data" : data}
        # Compute intuitive bracket uncertainties (τ=0.75 default)
        tau = 0.75
        theta_br_tau, theta_br_ideal = self._compute_theta_brackets(
            a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
            X_ij=X_ij, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean, tau=tau,
        )
        b_br_tau, b_br_ideal = self._compute_b_brackets(
            a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
            X_ij=X_ij, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean, tau=tau,
        )
        gamma_br_tau, gamma_br_ideal = self._compute_gamma_brackets(
            a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j, c_i=c_i,
            X_ij=X_ij, w_post_jk=w_post_jk, theta_k=theta_k, theta_j_mean=theta_j_mean, tau=tau,
        )

        return CalibrationResult(
            a_i=a_i, b_i=b_i, lambda_i=lambda_i, gamma_j=gamma_j,
            theta_j_mean=theta_j_mean, theta_j_se=theta_j_se,
            theta_bracket_half_tau=theta_br_tau, theta_bracket_half_ideal=theta_br_ideal,
            b_bracket_half_tau=b_br_tau, b_bracket_half_ideal=b_br_ideal,
            gamma_bracket_half_tau=gamma_br_tau, gamma_bracket_half_ideal=gamma_br_ideal,
            w_post_jk=w_post_jk, theta_k=theta_k, info=info,
        )

    # ---------- E‑step ----------
    def _estep(
        self,
        *,
        Y_ij: NDArray, X_ij: NDArray,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        theta_k: NDArray, w_k: NDArray,
    ) -> Tuple[NDArray, float]:
        """E-step: Posterior w_post_jk and observed log‑likelihood.

        Returns
        -------
        w_post_jk : (N,K)
        p(θ_k | Y_j) with rows summing to 1.
        ll_obs : float
        ∑_j log ∑_k w_k · p(Y_j | θ_k).
        """
        Y = np.asarray(Y_ij, float)
        X = np.asarray(X_ij, float)
        Q, N = Y.shape
        K = theta_k.shape[0]

        # Precompute the item-by-node base term (independent of j): a_i*(theta_k - b_i)
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        # Allocate outputs
        w_post_jk = np.empty((N, K), dtype=float)
        ll_obs = 0.0

        log_wk = np.log(w_k + EPS)  # safe log of prior quadrature weights

        # Per-respondent posterior over theta_k
        for j in range(N):
            budget_i = a_i * lambda_i * gamma_j[j] * X[:, j]  # (Q,)

            eta_ik = base_ik + budget_i[:, None]              # (Q,K)
            s_ik = expit(eta_ik)
            p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)

            # Log-likelihood across items for each node k
            y_i = Y[:, j][:, None]
            ll_items_k = (y_i * np.log(p_ik) + (1.0 - y_i) * np.log(1.0 - p_ik)).sum(axis=0)  # (K,)

            # Posterior over nodes: softmax(log_wk + ll_items_k)
            log_post_k = log_wk + ll_items_k
            m = np.max(log_post_k)
            w = np.exp(log_post_k - m)
            z = np.sum(w)
            w_post_jk[j, :] = w / z

            # Observed-data log-likelihood contribution for respondent j
            ll_obs += m + np.log(z)

        return w_post_jk, float(ll_obs)

    # ---------- M‑step: items (a_i, b_i) ----------
    def _mstep_items(
        self,
        *,
        Y_ij: NDArray, X_ij: NDArray,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        theta_k: NDArray, w_post_jk: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Per-item update of (a_i, b_i) via L-BFGS-B with analytic gradients.
        
        Bounds: a_i ≥ a_min to keep the slope positive and avoid the ill-posed flip (a_i<0)
        Objective = expected NLL + ridge on a,b.
        """
        Q, N = Y_ij.shape
        K = theta_k.shape[0]

        for i in range(Q):
            # Slices for item i
            y_j = Y_ij[i, :]                 # (N,)
            x_j = X_ij[i, :]                 # (N,)
            c = float(c_i[i])                # scalar
            lam = float(lambda_i[i])         # scalar

            # Respondent carrier independent of nodes
            u_j = lam * gamma_j * x_j        # (N,)

            def obj_grad(params: NDArray) -> Tuple[float, NDArray]:
                a, b = float(params[0]), float(params[1])

                # Linear predictor per respondent j and node k
                base_k = a * (theta_k - b)                # (K,)
                budget_j = a * u_j                        # (N,)
                eta_jk = base_k[None, :] + budget_j[:, None]  # (N,K)

                # Probabilities with 3PL guessing
                s_jk = expit(eta_jk)                      # (N,K)
                p_jk = c + (1.0 - c) * s_jk               # (N,K)
                p_jk = np.clip(p_jk, EPS, 1.0 - EPS)

                # Expected negative log-likelihood over GH nodes
                ll_jk = y_j[:, None] * np.log(p_jk) + (1.0 - y_j)[:, None] * np.log(1.0 - p_jk)  # (N,K)
                nll = -np.sum(w_post_jk * ll_jk)          # scalar

                # Logistic derivatives
                g_jk = (1.0 - c) * s_jk * (1.0 - s_jk)    # (N,K)
                r_jk = (y_j[:, None] - p_jk) / (p_jk * (1.0 - p_jk))  # (N,K)

                # Jacobians of eta (free indices j,k)
                deta_da_jk = (theta_k - b)[None, :] + u_j[:, None]  # (N,K)
                deta_db_jk = -a                                     # scalar, broadcasts

                wg = w_post_jk * r_jk * g_jk
                grad_a = -np.sum(wg * deta_da_jk) + 2.0 * self.cfg.ridge_a * (a - self.cfg.mu_a)
                grad_b = -np.sum(wg * deta_db_jk) + 2.0 * self.cfg.ridge_b * (b - self.cfg.mu_b)

                obj = nll + self.cfg.ridge_a * (a - self.cfg.mu_a) ** 2 + self.cfg.ridge_b * (b - self.cfg.mu_b) ** 2
                return float(obj), np.array([grad_a, grad_b], dtype=float)

            x0 = np.array([a_i[i], b_i[i]], dtype=float)
            bounds = [(self.cfg.a_min, self.cfg.a_max), (None, None)]
            res = minimize(obj_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 40, "ftol": 1e-6})
            a_i[i], b_i[i] = res.x

        return a_i, b_i

    # ---------- M‑step: item gate (lambda_i) ----------
    def _mstep_item_gate(
        self,
        *,
        Y_ij: NDArray, X_ij: NDArray,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        theta_k: NDArray, w_post_jk: NDArray,
    ) -> NDArray:
        """Per-item update of lambda_i in [0,1] via L-BFGS-B with analytic gradient.
        Objective = expected NLL + ridge_lambda · (λ − μ_λ)². Skip if no budget signal.
        """
        Q, N = Y_ij.shape
        for i in range(Q):
            y_j = Y_ij[i, :]                 # (N,)
            x_j = X_ij[i, :]                 # (N,)
            c   = float(c_i[i])              # scalar
            a   = float(a_i[i])              # scalar
            b   = float(b_i[i])              # scalar

            # Carriers independent of lambda
            u_j    = a * gamma_j * x_j       # (N,)
            base_k = a * (theta_k - b)       # (K,)

            # If no budget signal, skip optimization
            if np.allclose(u_j, 0.0):
                continue

            def obj_grad_lambda(lam_arr: NDArray) -> Tuple[float, NDArray]:
                lam = float(lam_arr[0])
                eta_jk = base_k[None, :] + (lam * u_j)[:, None]            # (N,K)
                s_jk   = expit(eta_jk)
                p_jk   = c + (1.0 - c) * s_jk
                p_jk   = np.clip(p_jk, EPS, 1.0 - EPS)

                ll_jk  = y_j[:, None] * np.log(p_jk) + (1.0 - y_j)[:, None] * np.log(1.0 - p_jk)
                nll    = -np.sum(w_post_jk * ll_jk)
                obj = nll + self.cfg.ridge_lambda * (lam - self.cfg.mu_lambda) ** 2

                g_jk = (1.0 - c) * s_jk * (1.0 - s_jk)
                r_jk = (y_j[:, None] - p_jk) / (p_jk * (1.0 - p_jk))
                grad = -np.sum((w_post_jk * r_jk * g_jk) * u_j[:, None]) + 2.0 * self.cfg.ridge_lambda * (lam - self.cfg.mu_lambda)
                return float(obj), np.array([grad], dtype=float)

            x0 = np.array([lambda_i[i]], dtype=float)
            bounds = [(0.0, 1.0)]
            res = minimize(obj_grad_lambda, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": 40, "ftol": 1e-6})
            lambda_i[i] = float(np.clip(res.x[0], 0.0, 1.0))

        return lambda_i

    # ---------- M‑step: model gain (gamma_j) ----------
    def _mstep_model_gain(
        self,
        *,
        Y_ij: NDArray, X_ij: NDArray,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        theta_k: NDArray, w_post_jk: NDArray,
    ) -> NDArray:
        """Per-respondent update of gamma_j via L-BFGS-B (unbounded on ℝ) with a ridge prior.

        Objective = expected NLL + ridge_gamma · (γ − μ_γ)². Skip if no budget signal.
        """
        Q, N = Y_ij.shape
        mu_gamma = float(self.cfg.mu_gamma)

        # Precompute item/node base once (independent of j)
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        for j in range(N):
            y_i = Y_ij[:, j]                 # (Q,)
            x_i = X_ij[:, j]                 # (Q,)
            w_k = w_post_jk[j, :]            # (K,)

            # Carrier independent of gamma for this respondent
            v_i = a_i * lambda_i * x_i       # (Q,)
            if np.allclose(v_i, 0.0):
                continue  # no budget signal for respondent j

            def obj_grad_gamma(g_arr: NDArray) -> Tuple[float, NDArray]:
                gamma = float(g_arr[0])
                eta_ik = base_ik + (gamma * v_i)[:, None]             # (Q,K)
                s_ik   = expit(eta_ik)
                p_ik   = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik
                p_ik   = np.clip(p_ik, EPS, 1.0 - EPS)

                ll_ik  = y_i[:, None] * np.log(p_ik) + (1.0 - y_i)[:, None] * np.log(1.0 - p_ik)
                nll    = -np.sum(w_k[None, :] * ll_ik)
                obj    = nll + self.cfg.ridge_gamma * (gamma - mu_gamma) ** 2

                g_ik  = (1.0 - c_i)[:, None] * s_ik * (1.0 - s_ik)
                r_ik  = (y_i[:, None] - p_ik) / (p_ik * (1.0 - p_ik))
                grad  = -np.sum((r_ik * g_ik * w_k[None, :]) * v_i[:, None]) + 2.0 * self.cfg.ridge_gamma * (gamma - mu_gamma)
                return float(obj), np.array([grad], dtype=float)

            x0 = np.array([gamma_j[j]], dtype=float)
            res = minimize(obj_grad_gamma, x0, method="L-BFGS-B", jac=True, options={"maxiter": 40, "ftol": 1e-6})
            gamma_j[j] = float(res.x[0])

        return gamma_j

    # ---------- Identifiability pin ----------
    def _pin_identifiability(
        self,
        *,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray,
        w_post_jk: NDArray, theta_k: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """identifiability (Gauge) fix: standardize the latent θ keeping η invariant on {θ_k}.

        a' = a σ, b' = (b − μ)/σ, γ' = γ/σ. λ unchanged (clipped to [0,1]).
        """
        # Posterior mean and second moment per respondent
        theta_j_mean = w_post_jk @ theta_k                 # (N,)
        theta2_j_mean = w_post_jk @ (theta_k ** 2)         # (N,)

        # Mixture mean/variance across respondents
        mu_theta = float(theta_j_mean.mean())
        var_theta = float(theta2_j_mean.mean() - mu_theta ** 2)
        sigma_theta = float(np.sqrt(max(var_theta, 1e-12)))

        # Co-transform
        a_i = a_i * sigma_theta
        b_i = (b_i - mu_theta) / sigma_theta
        gamma_j = gamma_j / sigma_theta
        
        print(f"mu_theta = {mu_theta: .3}, sigma_theta = {sigma_theta: .5}")

        # post-hoc bound for gamma_j, lambda_i and a_i
        gamma_j = np.clip(gamma_j, self.cfg.gamma_min, self.cfg.gamma_max)
        lambda_i = np.clip(lambda_i, self.cfg.lambda_min, self.cfg.lambda_max)
        a_i = np.clip(a_i, self.cfg.a_min, self.cfg.a_max)
        return a_i, b_i, lambda_i, gamma_j

    # ---------- Utilities ----------

    def _prepare_inputs(self, data: CalibrationData) -> Tuple[NDArray, NDArray, NDArray]:
        """Minimal prep: validate shapes/values, build X_ij = log1p(B_ij), broadcast c_i to (Q,)."""
        Y = np.asarray(data.Y_ij, float)
        B = np.asarray(data.B_ij, float)
        c = np.asarray(data.c_i, float)
        if Y.ndim != 2 or B.ndim != 2 or Y.shape != B.shape:
            raise ValueError("Y_ij and B_ij must be 2D with identical shape (Q,N).")
        if c.ndim != 1 or c.shape[0] != Y.shape[0]:
            raise ValueError("c_i must be (Q,)")
        if not (np.isfinite(Y).all() and np.isfinite(B).all()):
            raise ValueError("Y_ij and B_ij must be finite.")
        if ((Y != 0.0) & (Y != 1.0)).any():
            raise ValueError("Y_ij must be binary 0/1.")
        if (B < 0.0).any():
            raise ValueError("B_ij must be >= 0.")
        X = np.log1p(B)        
        return Y, X, c

    def _make_gh(self, K: int) -> Tuple[NDArray, NDArray]:
        """Gauss–Hermite nodes/weights for a standard Normal prior: sum(weights) ≈ 1.
        We *do not* renormalize beyond the analytic transform; GH is exact for polynomials,
        so weights = w/√π already integrate a constant to 1 up to roundoff.
        """
        if K < 3:
            raise ValueError("gh_nodes must be >= 3.")
        x, w = hermgauss(K)
        theta_k = np.sqrt(2.0) * x
        w_k = w / np.sqrt(np.pi)
        return theta_k.astype(np.float64), w_k.astype(np.float64)
    
    def _init_params(self, Y_ij: NDArray, c_i: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Heuristic Data-aware initialization.
        Strategy
        --------
        θ proxy from column means of Y (logit‑scaled, centered). b from row means
        of Y (centered). a ≈ 1 with mild jitter; λ near μ_λ with jitter (clip to [0,1]);
        γ near μ_γ with small Gaussian noise.
        """
        Y = np.asarray(Y_ij, float)
        Q, N = Y.shape

        # ------------------------------------------------------------
        # (0) Slope a_i  — neutral start (respect lower bound)
        # ------------------------------------------------------------
        a_i = np.maximum(np.ones(Q, dtype=float), self.cfg.a_min)

        # ------------------------------------------------------------
        # (1) Reduced 3PL seeding (ignore budget, take a_i≈1)
        #     p_ij ≈ c_i + (1−c_i)·σ(θ_j − b_i)
        #     s   = (p − c_i)/(1 − c_i) ∈ (0,1)
        # ------------------------------------------------------------
        c_bar = float(np.mean(c_i))

        # 1a) Respondent seed θ_j — from column means, then center to ~0
        #     s_j = ((ȳ_j − c̄)/(1 − c̄)) clipped; θ_j = logit(s_j)
        ybar_j = Y.mean(axis=0)
        denom_j = max(1e-12, 1.0 - c_bar)
        s_j = np.clip((ybar_j - c_bar) / denom_j, EPS, 1.0 - EPS)
        theta_j = logit(s_j)
        theta_j -= float(np.mean(theta_j))  # center latent scale

        # 1b) Item difficulty b_i — from row means with c_i correction
        #     s_i = ((ȳ_i − c_i)/(1 − c_i)) clipped;  b_i = −logit(s_i)
        ybar_i = Y.mean(axis=1)
        denom_i = np.clip(1.0 - c_i, 1e-12, None)
        s_i = np.clip((ybar_i - c_i) / denom_i, EPS, 1.0 - EPS)
        b_i = -logit(s_i)

        # ------------------------------------------------------------
        # (2) Budget coupling priors
        # ------------------------------------------------------------
        # lambda_i: centered near 0.8 (prior center mu_lambda),
        #           add small jitter and clip to [0.05, 0.99]
        lambda_i = 0.8 + self.rng.normal(0.0, 0.05, size=Q)
        lambda_i = np.clip(lambda_i.astype(float), 0.05, 0.99)

        # gamma_j: slightly positive (LLMs usually have some reasoning skill)
        gamma_j = self.rng.normal(0.2, 0.15, size=N).astype(float)

        return a_i, b_i, lambda_i, gamma_j

    def _summarize_theta(self, w_post_jk: NDArray, theta_k: NDArray) -> Tuple[NDArray, NDArray]:
        """Posterior mean and standard error of θ per respondent from w_post_jk over θ_k."""
        theta_j_mean = w_post_jk @ theta_k                      # (N,)
        m2_j = w_post_jk @ (theta_k ** 2)                       # (N,)
        theta_j_se = np.sqrt(np.maximum(m2_j - theta_j_mean ** 2, 0.0))  # (N,)
        return theta_j_mean, theta_j_se

    # ---------- Bracket-style uncertainties ----------
    def _compute_theta_brackets(
        self,
        *,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        X_ij: NDArray, w_post_jk: NDArray, theta_k: NDArray, theta_j_mean: NDArray,
        tau: float = 0.75,
    ) -> Tuple[NDArray, NDArray]:
        """Per-respondent θ brackets (τ-bracket and ideal). Returns two (N,) arrays of half-widths.
        If one side of the bracket is missing, returns NaN for that respondent.
        """
        Q, N = X_ij.shape
        K = theta_k.shape[0]
        # Base term for all items/nodes (independent of j)
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        theta_half_tau = np.full(N, np.nan, dtype=float)
        theta_half_ideal = np.full(N, np.nan, dtype=float)

        for j in range(N):
            theta_hat = float(theta_j_mean[j])
            # Effective difficulties for respondent j
            d_i = b_i - (lambda_i * gamma_j[j] * X_ij[:, j])        # (Q,)
            # Posterior-predictive success per item for respondent j
            budget_i = a_i * lambda_i * gamma_j[j] * X_ij[:, j]     # (Q,)
            eta_ik = base_ik + budget_i[:, None]                    # (Q,K)
            s_ik = expit(eta_ik)
            p_ik = c_i[:, None] + (1.0 - c_i)[:, None] * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            p_avg_i = p_ik @ w_post_jk[j, :]                        # (Q,)

            # Ideal nearest neighbors (ignoring probabilities)
            left_mask = d_i <= theta_hat
            right_mask = d_i >= theta_hat
            if np.any(left_mask) and np.any(right_mask):
                L_val = np.max(d_i[left_mask])
                R_val = np.min(d_i[right_mask])
                theta_half_ideal[j] = 0.5 * (R_val - L_val)

            # τ-bracket: left needs high success prob (≥ τ), right needs low (≤ 1-τ)
            left_tau_mask = left_mask & (p_avg_i >= tau)
            right_tau_mask = right_mask & (p_avg_i <= 1.0 - tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L_tau = np.max(d_i[left_tau_mask])
                R_tau = np.min(d_i[right_tau_mask])
                theta_half_tau[j] = 0.5 * (R_tau - L_tau)

        return theta_half_tau, theta_half_ideal

    def _compute_b_brackets(
        self,
        *,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        X_ij: NDArray, w_post_jk: NDArray, theta_k: NDArray, theta_j_mean: NDArray,
        tau: float = 0.75,
    ) -> Tuple[NDArray, NDArray]:
        """Per-item b brackets (τ-bracket and ideal). Returns two (Q,) arrays of half-widths.
        If one side of the bracket is missing, returns NaN for that item.
        """
        Q, N = X_ij.shape
        b_half_tau = np.full(Q, np.nan, dtype=float)
        b_half_ideal = np.full(Q, np.nan, dtype=float)

        for i in range(Q):
            b_hat = float(b_i[i])
            # Adjusted abilities for this item across respondents
            tilde_theta_j = theta_j_mean + (lambda_i[i] * gamma_j * X_ij[i, :])  # (N,)

            # Posterior-predictive success for this item across respondents
            a = float(a_i[i]); b = float(b_i[i]); c = float(c_i[i])
            base_k = a * (theta_k - b)                                  # (K,)
            budget_j = a * (lambda_i[i] * gamma_j * X_ij[i, :])         # (N,)
            eta_jk = base_k[None, :] + budget_j[:, None]                # (N,K)
            s_jk = expit(eta_jk)
            p_jk = c + (1.0 - c) * s_jk
            p_jk = np.clip(p_jk, EPS, 1.0 - EPS)
            p_avg_j = np.sum(w_post_jk * p_jk, axis=1)                  # (N,)

            # Ideal nearest neighbors in tilde_theta
            left_mask = tilde_theta_j <= b_hat
            right_mask = tilde_theta_j >= b_hat
            if np.any(left_mask) and np.any(right_mask):
                L_val = np.max(tilde_theta_j[left_mask])
                R_val = np.min(tilde_theta_j[right_mask])
                b_half_ideal[i] = 0.5 * (R_val - L_val)

            # τ-bracket: left wants failure (≤ 1-τ), right wants success (≥ τ)
            left_tau_mask = left_mask & (p_avg_j <= 1.0 - tau)
            right_tau_mask = right_mask & (p_avg_j >= tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L_tau = np.max(tilde_theta_j[left_tau_mask])
                R_tau = np.min(tilde_theta_j[right_tau_mask])
                b_half_tau[i] = 0.5 * (R_tau - L_tau)

        return b_half_tau, b_half_ideal

    def _compute_gamma_brackets(
        self,
        *,
        a_i: NDArray, b_i: NDArray, lambda_i: NDArray, gamma_j: NDArray, c_i: NDArray,
        X_ij: NDArray, w_post_jk: NDArray, theta_k: NDArray, theta_j_mean: NDArray,
        tau: float = 0.75,
    ) -> Tuple[NDArray, NDArray]:
        """Per-respondent γ brackets (τ-bracket and ideal). Returns two (N,) arrays of half-widths.
        Items with λ_i X_ij == 0 contribute no γ information and are ignored. NaN if one side missing.
        """
        Q, N = X_ij.shape
        K = theta_k.shape[0]
        # Precompute item/node base once
        base_ik = a_i[:, None] * (theta_k[None, :] - b_i[:, None])  # (Q,K)

        gamma_half_tau = np.full(N, np.nan, dtype=float)
        gamma_half_ideal = np.full(N, np.nan, dtype=float)

        for j in range(N):
            theta_hat = float(theta_j_mean[j])
            denom_i = lambda_i * X_ij[:, j]                            # (Q,)
            valid = denom_i > 0.0
            if not np.any(valid):
                continue
            gamma_star_i = (b_i[valid] - theta_hat) / denom_i[valid]   # (Q_valid,)

            # Posterior-predictive success per valid item at current γ_j
            budget_i = a_i[valid] * lambda_i[valid] * gamma_j[j] * X_ij[valid, j]  # (Q_valid,)
            eta_ik = base_ik[valid, :] + budget_i[:, None]                          # (Q_valid,K)
            s_ik = expit(eta_ik)
            p_ik = c_i[valid, None] + (1.0 - c_i[valid, None]) * s_ik
            p_ik = np.clip(p_ik, EPS, 1.0 - EPS)
            p_avg_i = p_ik @ w_post_jk[j, :]                                        # (Q_valid,)

            ghat = float(gamma_j[j])
            left_mask = gamma_star_i <= ghat
            right_mask = gamma_star_i >= ghat
            if np.any(left_mask) and np.any(right_mask):
                L_val = np.max(gamma_star_i[left_mask])
                R_val = np.min(gamma_star_i[right_mask])
                gamma_half_ideal[j] = 0.5 * (R_val - L_val)

            left_tau_mask = left_mask & (p_avg_i >= tau)
            right_tau_mask = right_mask & (p_avg_i <= 1.0 - tau)
            if np.any(left_tau_mask) and np.any(right_tau_mask):
                L_tau = np.max(gamma_star_i[left_tau_mask])
                R_tau = np.min(gamma_star_i[right_tau_mask])
                gamma_half_tau[j] = 0.5 * (R_tau - L_tau)

        return gamma_half_tau, gamma_half_ideal

# ---------- Post-hoc External Utilities ----------0
def compute_laplace_ses(
    result,                      # CalibrationResult from fit(...)
    data,                        # CalibrationData used in fit(...)
    *,
    blocks = ("a","b","lambda","gamma"),
    ridges: Dict[str, float] | None = None,   # {'ridge_a':..., 'ridge_b':..., 'ridge_lambda':..., 'ridge_gamma':...}
) -> Dict[str, np.ndarray]:
    """
    Post-hoc Laplace (Fisher) standard errors under the fitted model.
    Returns arrays keyed by '<param>_laplace_se'. Uses the final E-step posteriors w_post_jk.

    Notes:
      - Works entirely post-fit (no GEM changes).
      - If you used nonzero ridges in the fit, pass the same values via `ridges=...`
        (we default to 0.0 if not provided).
      - Near boundaries (e.g., a_i≈a_min, lambda_i≈0/1) SEs reflect the penalty and can be prior-dominated.
    
    Usage Example:
    # After you run fit(...)
    res = est.fit(data)

    # If you used nonzero ridges in fit, pass them here for honest SEs:
    ses = compute_laplace_ses(
        res, data,
        blocks=("a","b","lambda","gamma"),           # or any subset
        ridges={"ridge_a":1e-3,"ridge_b":1e-3,"ridge_lambda":1e-5,"ridge_gamma":1e-5},
    )

    # Output : ses["gamma_laplace_se"]  -> (N,)  SEs for gamma_j
    #          ses["b_laplace_se"]      -> (Q,)
    #          ses["a_laplace_se"]      -> (Q,)
    """
    # Extract
    a_i       = np.asarray(result.a_i,       dtype=float)   # (Q,)
    b_i       = np.asarray(result.b_i,       dtype=float)   # (Q,)
    lam_i     = np.asarray(result.lambda_i,  dtype=float)   # (Q,)
    gam_j     = np.asarray(result.gamma_j,   dtype=float)   # (N,)
    theta_k   = np.asarray(result.theta_k,   dtype=float)   # (K,)
    w_post_jk = np.asarray(result.w_post_jk, dtype=float)   # (N,K)

    Y_ij = np.asarray(data.Y_ij, dtype=float)               # (Q,N) not used directly here
    B_ij = np.asarray(data.B_ij, dtype=float)               # (Q,N)
    c_i  = np.asarray(data.c_i,  dtype=float)               # (Q,)

    Q, N = Y_ij.shape
    K = theta_k.shape[0]
    X_ij = np.log1p(B_ij)                                   # absolute budgets, as in fit

    # Ridges (if not provided, try to fetch from result.info; else fall back to 0.)
    def _ridge(name, default=0.0):
        if ridges is not None and name in ridges: return float(ridges[name])
        if getattr(result, "info", None) and name in result.info: return float(result.info[name])
        return float(default)

    ridge_a       = _ridge("ridge_a", 0.0)
    ridge_b       = _ridge("ridge_b", 0.0)
    ridge_lambda  = _ridge("ridge_lambda", 0.0)
    ridge_gamma   = _ridge("ridge_gamma", 0.0)

    # ----- Build eta_{i j k}, p_{i j k}, and Fisher weight W^F_{i j k} -----
    # Shapes:
    # base_ik:   (Q,K)  = a_i[:,None] * (theta_k[None,:] - b_i[:,None])
    # budget_ij: (Q,N)  = (a_i * lam_i)[:,None] * gam_j[None,:] * X_ij
    base_ik   = a_i[:, None] * (theta_k[None, :] - b_i[:, None])              # (Q,K)
    budget_ij = (a_i * lam_i)[:, None] * gam_j[None, :] * X_ij                # (Q,N)
    eta_ijk   = base_ik[:, None, :] + budget_ij[:, :, None]                   # (Q,N,K)

    s_ijk = expit(eta_ijk)                                                    # (Q,N,K)
    p_ijk = c_i[:, None, None] + (1.0 - c_i)[:, None, None] * s_ijk           # (Q,N,K)
    p_ijk = np.clip(p_ijk, EPS, 1.0 - EPS)

    g_ijk  = (1.0 - c_i)[:, None, None] * s_ijk * (1.0 - s_ijk)               # (Q,N,K)
    Wf_ijk = (g_ijk * g_ijk) / (p_ijk * (1.0 - p_ijk) + EPS)                  # (Q,N,K)

    # Posterior weights (broadcast to (Q,N,K))
    w_ijk = w_post_jk[None, :, :]                                             # (1,N,K)

    out: Dict[str, np.ndarray] = {}

    # ----- b_i (1-D): H_bb = sum_{j,k} w Wf * a_i^2 + 2*ridge_b -----
    if "b" in blocks:
        Hbb = (a_i * a_i) * np.sum(Wf_ijk * w_ijk, axis=(1, 2)) + 2.0 * ridge_b   # (Q,)
        b_se = 1.0 / np.sqrt(np.maximum(Hbb, EPS))
        out["b_laplace_se"] = b_se

    # ----- lambda_i (1-D): dη/dλ = a_i * γ_j * X_ij -----
    if "lambda" in blocks:
        dlam_ij = (a_i[:, None] * gam_j[None, :] * X_ij)                          # (Q,N)
        Hll = np.sum((Wf_ijk * w_ijk) * (dlam_ij[:, :, None] ** 2), axis=(1, 2)) + 2.0 * ridge_lambda
        lam_se = 1.0 / np.sqrt(np.maximum(Hll, EPS))
        out["lambda_laplace_se"] = lam_se

    # ----- gamma_j (1-D): dη/dγ = a_i * λ_i * X_ij -----
    if "gamma" in blocks:
        dgam_ij = (a_i * lam_i)[:, None] * X_ij                                   # (Q,N)
        # sum over items i and nodes k for each j
        Wsum_ij = np.sum(Wf_ijk * w_ijk, axis=2)                                  # (Q,N)
        Hgg = np.sum(Wsum_ij * (dgam_ij ** 2), axis=0) + 2.0 * ridge_gamma        # (N,)
        gam_se = 1.0 / np.sqrt(np.maximum(Hgg, EPS))
        out["gamma_laplace_se"] = gam_se

    # ----- (a_i, b_i) joint 2x2 per item -----
    if "a" in blocks or "ab" in blocks:
        # dη/da = (θ_k - b_i) + λ_i γ_j X_ij
        term1_ik = (theta_k[None, :] - b_i[:, None])                               # (Q,K)
        term2_ij = (lam_i[:, None] * gam_j[None, :] * X_ij)                        # (Q,N)
        dda_ijk  = term1_ik[:, None, :] + term2_ij[:, :, None]                     # (Q,N,K)
        # dη/db = -a_i
        ddb_i    = -a_i                                                            # (Q,)

        # Fisher blocks
        Waa = np.sum((Wf_ijk * w_ijk) * (dda_ijk ** 2), axis=(1, 2)) + 2.0 * ridge_a     # (Q,)
        Wbb = (a_i * a_i) * np.sum(Wf_ijk * w_ijk, axis=(1, 2)) + 2.0 * ridge_b          # (Q,)
        Wab = np.sum((Wf_ijk * w_ijk) * (dda_ijk * ddb_i[:, None, None]), axis=(1, 2))   # (Q,)

        # Invert 2x2 Fisher per item safely
        det = Waa * Wbb - Wab * Wab
        # jitter where needed
        bad = det <= EPS
        if np.any(bad):
            jitter = 1e-8
            Waa = Waa + bad * jitter
            Wbb = Wbb + bad * jitter
            det = Waa * Wbb - Wab * Wab

        inv_aa =  Wbb / det
        inv_bb =  Waa / det
        inv_ab = -Wab / det

        a_se = np.sqrt(np.maximum(inv_aa, 0.0))
        b_se2 = np.maximum(inv_bb, 0.0)    # if you want a second estimate via joint block
        ab_corr = inv_ab / (a_se * np.sqrt(np.maximum(inv_bb, 0.0)) + EPS)

        if "a" in blocks:  out["a_laplace_se"] = a_se
        if "ab" in blocks: out["ab_laplace_corr"] = ab_corr
        # If you want to expose the joint-based b SE instead of (or in addition to) the 1D one:
        # out["b_laplace_se_from_ab"] = np.sqrt(b_se2)

    return out


