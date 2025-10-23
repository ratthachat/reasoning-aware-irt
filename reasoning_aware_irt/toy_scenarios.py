from __future__ import annotations
from dataclasses import dataclass
from typing import *

import numpy as np
from numpy.random import default_rng
from scipy.special import expit

def make_reasoning_aware_3pl_toy(
    Q: int = 200,
    N: int = 40,
    B_levels = (0, 256, 1024, 4096, 8192),
    frac_gate_on: float = 0.5,
    gate_on: float = 0.95,
    gate_off: float = 0.05,
    a_low: float = 0.95, a_high: float = 1.05,
    b_span_low: float = -2.0, b_span_high: float = 3.0,
    c_val: float = 0.10,
    theta_span: float = 1.5,
    mu_gamma: float = 0.2, sigma_gamma: float = 0.15,
    seed: int = 0,
):
    """Toy data generators aligned with our GEM model.
    Model: eta_ij = a_i*(theta_j - b_i) + a_i*lambda_i*gamma_j*X_ij,  X_ij = log1p(B_ij)
    """
    
    rng = default_rng(seed)
    Q = int(Q); N = int(N)

    # Ground truth parameters
    theta_j_true = np.linspace(-theta_span, theta_span, N)                 # (N,)
    gamma_j_true = rng.normal(mu_gamma, sigma_gamma, size=N)               # (N,)
    a_i_true = np.linspace(a_low, a_high, Q).astype(float)                 # (Q,)
    b_i_true = np.linspace(b_span_low, b_span_high, Q).astype(float)       # (Q,)
    c_i_true = np.full(Q, float(c_val))                                    # (Q,)

    # λ_i: on/off pattern across items
    num_on = int(round(frac_gate_on * Q))
    lambda_i_true = np.full(Q, float(gate_off))
    lambda_i_true[:num_on] = float(gate_on)
    rng.shuffle(lambda_i_true)

    # Budgets and features (absolute)
    x_levels = np.log1p(np.asarray(B_levels, dtype=float))                 # (L,)
    L = len(x_levels)
    idx_base = np.tile(np.arange(L, dtype=int), int(np.ceil(Q / L)))[:Q]   # (Q,)

    X_ij = np.empty((Q, N), dtype=float)
    for j in range(N):
        idx = idx_base.copy(); rng.shuffle(idx)
        X_ij[:, j] = x_levels[idx]
    B_ij = np.expm1(X_ij)

    # Linear predictor & probabilities
    eta_ij = (a_i_true[:, None] * (theta_j_true[None, :] - b_i_true[:, None])
              + (a_i_true * lambda_i_true)[:, None] * gamma_j_true[None, :] * X_ij)
    eta_ij = np.clip(eta_ij, -40, 40)
    P_ij = c_i_true[:, None] + (1.0 - c_i_true[:, None]) * expit(eta_ij)

    # Draw responses
    Y_ij = rng.binomial(1, P_ij).astype(float)

    # Heuristic θ_j_max (per-respondent max-budget lift)
    X_max_j = X_ij.max(axis=0)                                # (N,)
    theta_j_max = theta_j_true + (np.mean(lambda_i_true) * gamma_j_true) * X_max_j

    return dict(
        Y_ij=Y_ij, B_ij=B_ij, X_ij=X_ij, c_i=c_i_true,
        a_i_true=a_i_true, b_i_true=b_i_true,
        lambda_i_true=lambda_i_true, gamma_j_true=gamma_j_true,
        theta_j_true=theta_j_true, theta_j_max=theta_j_max, P_ij=P_ij,
        meta=dict(B_levels=B_levels, seed=seed),
    )

# =============================================
# scenarios
# ---------------------------------------------
"""Scenario helpers for pre‑diagnostics (wrap toy generator).
Provides both a simple seed-based builder and your original TestScenarios API.
"""

def build_scenarios(methods: List[str] | None = None, **gen_kwargs) -> Tuple[list, list]:
    """Build scenarios by invoking methods on TestScenarios.

    If `methods` is None, all public methods of TestScenarios are used
    (e.g., pass_all, fail_effective_rank, ...). Any `gen_kwargs` are merged
    into `TestScenarios.pass_cfg` before generation.
    """
    scen = TestScenarios()
    scen.pass_cfg.update(gen_kwargs)

    # enumerate public, callable methods if none provided
    if methods is None:
        methods = [m for m in dir(scen) if not m.startswith('_') and callable(getattr(scen, m))]

    toy_list: list = []
    names: list = []
    for m in methods:
        if not hasattr(scen, m):
            raise AttributeError(f"Unknown scenario method '{m}' in TestScenarios")
        toy = getattr(scen, m)()
        toy_list.append(toy)
        names.append(m)
    return toy_list, names

def _recompute_derived(toy: dict) -> dict:
    """Keep derived fields consistent after editing budgets/columns."""
    toy['X_ij'] = np.log1p(toy['B_ij'])
    X_max_j = toy['X_ij'].max(axis=0)
    lam_mean = float(np.mean(toy.get('lambda_i_true', np.ones(toy['X_ij'].shape[0]))))
    if 'theta_j_true' in toy and 'gamma_j_true' in toy:
        toy['theta_j_max'] = toy['theta_j_true'] + lam_mean * toy['gamma_j_true'] * X_max_j
    return toy

@dataclass
class TestScenarios:
    """Generates datasets designed to pass or fail specific pre-checks.

    This preserves your original API, adapted to our generator & key names.
    """
    def __init__(self):
        self.pass_cfg = dict(Q=200,
                             N=50,
                             B_levels=(0, 512, 2048, 10000),
                             seed=42)

    def pass_all(self) -> dict:
        toy = make_reasoning_aware_3pl_toy(**self.pass_cfg)
        # Boost rank by varying budget proportions per model (column-wise Dirichlet)
        Q, N = toy['Y_ij'].shape
        B_levels = self.pass_cfg.get('B_levels', toy['meta'].get('B_levels', (0,128,256,512,1024,2048,4096)))
        x_levels = np.log1p(np.array(B_levels, float))
        L = len(x_levels)
        rng = np.random.default_rng(123)
        X = np.empty_like(toy['X_ij'], dtype=float)
        for j in range(N):
            probs = rng.dirichlet(np.ones(L) * 0.5)
            counts = rng.multinomial(Q, probs)
            idx = np.repeat(np.arange(L), counts)
            rng.shuffle(idx)
            X[:, j] = x_levels[idx]
        toy['B_ij'] = np.expm1(X)
        return _recompute_derived(toy)

    def fail_variance_floor(self) -> dict:  # Test is too hard
        cfg = dict(self.pass_cfg)
        cfg.update(dict(b_span_low=5.0, b_span_high=7.0))
        return make_reasoning_aware_3pl_toy(**cfg)

    def fail_variance_ceiling(self) -> dict:  # Test is too easy
        cfg = dict(self.pass_cfg)
        cfg.update(dict(b_span_low=-5.0, b_span_high=-3.0))
        return make_reasoning_aware_3pl_toy(**cfg)

    def fail_fisher_proxy(self) -> dict:  # Bimodal difficulty (mostly floor/ceiling)
        cfg = dict(self.pass_cfg)
        cfg.update(dict(b_span_low=-30.0, b_span_high=30.0, a_low=2.0, a_high=2.5))
        return make_reasoning_aware_3pl_toy(**cfg)

    def fail_budget_leverage_items(self) -> dict:
        toy = make_reasoning_aware_3pl_toy(**self.pass_cfg)
        # For most items, set budget to be constant across models
        Q, _ = toy['B_ij'].shape
        for i in range(int(Q * 0.8)):
            toy['B_ij'][i, :] = np.median(toy['B_ij'][i, :])
        return _recompute_derived(toy)

    def fail_effective_rank(self) -> dict:
        # Rigorously create a near rank-1 budget matrix
        toy = make_reasoning_aware_3pl_toy(**self.pass_cfg)
        rng = np.random.default_rng(self.pass_cfg['seed'])
        Q, N = toy['B_ij'].shape
        base_col_budget = toy['B_ij'][:, 0].copy()
        for j in range(1, N):
            noise = base_col_budget * rng.uniform(-0.05, 0.05, size=Q)
            col = base_col_budget + noise
            toy['B_ij'][:, j] = np.clip(col, 0.0, None)
        return _recompute_derived(toy)

    def fail_confounding(self) -> dict:  # induce correlation by sorting columns by model score
        toy = make_reasoning_aware_3pl_toy(**self.pass_cfg)
        model_scores = np.mean(toy['Y_ij'], axis=0)
        sorted_indices = np.argsort(model_scores)
        toy['B_ij'] = toy['B_ij'][:, sorted_indices]
        toy['X_ij'] = toy['X_ij'][:, sorted_indices]
        toy['Y_ij'] = toy['Y_ij'][:, sorted_indices]
        # max stays same under permutation, but recompute to be explicit
        return _recompute_derived(toy)

    def fail_budget_response(self) -> dict:
        cfg = dict(self.pass_cfg)
        cfg.update(dict(gate_on=0.0, mu_gamma=0.0, sigma_gamma=0.0,
                        B_levels=(0, 128, 256, 512, 1024, 2048, 4096, 10000)))
        return make_reasoning_aware_3pl_toy(**cfg)