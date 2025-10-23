from dataclasses import dataclass
from typing import *

import numpy as np
from numpy.linalg import svd

import matplotlib.pyplot as plt

def log1p_strict(B_ij: np.ndarray) -> np.ndarray:
    """log1p with strict checks for nonnegativity/finite values.
    Returns X_ij = log1p(B_ij). Raises on invalid input.
    """
    B = np.asarray(B_ij, float)
    if B.ndim != 2:
        raise ValueError("B_ij must be 2D (Q,N)")
    if not np.isfinite(B).all():
        raise ValueError("B_ij must be finite")
    if (B < 0.0).any():
        raise ValueError("B_ij must be nonnegative")
    return np.log1p(B)

@dataclass
class PrecheckConfig:
    """Pre‑diagnostics on data prior to GEM fitting.
    Pure NumPy; shapes/names mirror the calibrator.
    """

    # Variance / entropy coverage for Y (items and models)
    entropy_range: tuple = (0.2, 0.8)
    entropy_proportion: float = 0.3

    # Fisher-information proxy floor
    fisher_proxy_thresh: float = 0.07

    # Budget leverage (per-item/ per-model X spread)
    budget_leverage_std_thresh: float = 0.5
    budget_leverage_proportion: float = 0.5

    # Effective-rank guards (uncentered SVD; drop 1st SV)
    effective_rank_min: float = 5.0
    rank_ratio_min_thresh: float = 0.2

    # Ability–budget confounding (column means corr)
    confounding_corr_thresh: float = 0.8

    # Point-biserial style association
    # pb_corr_thresh is set adaptively in DataPreChecker.__init__ if None:
    # 1.6 / sqrt(N - 2)
    pb_corr_thresh: float | None = None
    pb_proportion_thresh: float = 0.101

class DataPreChecker:
    def __init__(self, Y_ij: np.ndarray, B_ij: np.ndarray, P_ij: Optional[np.ndarray] = None, 
                 config: Optional[PrecheckConfig] = None, print_msg=True):
        self.Y_ij = np.asarray(Y_ij, float)
        self.B_ij = np.asarray(B_ij, float)
        self.P_ij = None if P_ij is None else np.asarray(P_ij, float)
        if self.Y_ij.shape != self.B_ij.shape:
            raise ValueError("Y_ij and B_ij must have identical shape (Q,N)")
        if self.P_ij is not None and self.P_ij.shape != self.Y_ij.shape:
            raise ValueError("P_ij must have the same shape as Y_ij if provided")
        self.Q, self.N = self.Y_ij.shape
        self.X_ij = log1p_strict(self.B_ij)   # absolute budgets
        self.config = config or PrecheckConfig()
        # Adaptive pb-corr rule of thumb: 1.6 * SD of random fluctuation
        if getattr(self.config, 'pb_corr_thresh', None) is None:
            self.config.pb_corr_thresh = 1.6 / np.sqrt(max(self.N - 2, 1))
        self.print_msg = print_msg

    # ---------- tests ----------
    def _test_variance_entropy_coverage(self) -> Dict:
        """Proportion of items/models with mean(Y) in an entropy-friendly band."""
        r_i = np.mean(self.Y_ij, axis=1)
        c_j = np.mean(self.Y_ij, axis=0)
        low, high = self.config.entropy_range
        prop_r = float(np.mean((r_i >= low) & (r_i <= high)))
        prop_c = float(np.mean((c_j >= low) & (c_j <= high)))
        fail = (prop_r < self.config.entropy_proportion) or (prop_c < self.config.entropy_proportion)
        return {
            'name': 'variance_entropy_coverage',
            'status': 'FAIL' if fail else 'PASS',
            'message': f"Variance/Entropy: Items={prop_r:.2f}, Models={prop_c:.2f}",
            'key_numbers': (prop_r, prop_c),
            'key_number_names': ('items_entropy_coverage', 'models_entropy_coverage'),
        }

    def _test_fisher_proxy(self) -> Dict:
        """Mean of y(1−y) across items/models as a crude Fisher-proxy."""
        r_i = np.mean(self.Y_ij, axis=1)
        c_j = np.mean(self.Y_ij, axis=0)
        fisher_r = float(np.mean(r_i * (1.0 - r_i)))
        fisher_c = float(np.mean(c_j * (1.0 - c_j)))
        warn = (fisher_r < self.config.fisher_proxy_thresh) or (fisher_c < self.config.fisher_proxy_thresh)
        return {
            'name': 'fisher_proxy',
            'status': 'WARN' if warn else 'PASS',
            'message': f"Fisher Proxy: Items={fisher_r:.4f}, Models={fisher_c:.4f}",
            'key_numbers': (fisher_r, fisher_c),
            'key_number_names': ('fisher_item_proxy', 'fisher_model_proxy'),
        }

    def _test_budget_leverage(self) -> Dict:
        """Check that X has nontrivial spread along item and model axes."""
        s_i = np.std(self.X_ij, axis=1)
        t_j = np.std(self.X_ij, axis=0)
        prop_i_ok = float(np.mean(s_i > self.config.budget_leverage_std_thresh))
        prop_m_ok = float(np.mean(t_j > self.config.budget_leverage_std_thresh))
        fail = (prop_i_ok < self.config.budget_leverage_proportion) or (prop_m_ok < self.config.budget_leverage_proportion)
        return {
            'name': 'budget_leverage',
            'status': 'FAIL' if fail else 'PASS',
            'message': f"Budget Leverage: Items OK={prop_i_ok:.2f}, Models OK={prop_m_ok:.2f}",
            'key_numbers': (prop_i_ok, prop_m_ok),
            'key_number_names': ('budget_items_variance_score', 'budget_models_variance_score'),
        }

    def _test_budget_matrix_effective_rank(self) -> Dict:
        """SVD-based effective rank of X (uncentered): drop first singular value."""
        sv = svd(self.X_ij, compute_uv=False)
        sv_subset = sv[1:] if sv.size > 1 else sv
        eff_rank = float((np.sum(sv_subset) ** 2) / (np.sum(sv_subset ** 2) + 1e-12)) if sv_subset.size else 0.0
        rank_ratio = float((sv[2] / (sv[1] + 1e-12)) if sv.size > 2 else 0.0)
        fail = (eff_rank < self.config.effective_rank_min) or (rank_ratio < self.config.rank_ratio_min_thresh)
        return {
            'name': 'effective_rank', 'status': 'FAIL' if fail else 'PASS',
            'message': f"Effective Rank={eff_rank:.2f}, Ratio={rank_ratio:.2f}",
            'key_numbers': (eff_rank, rank_ratio),
            'key_number_names': ('eff_rank_uncentered', 'rank_ratio_uncentered'),
        }

    def _test_ability_budget_confounding(self) -> Dict:
        """Correlation between model ability proxy mean(Y) and mean log-budget."""
        ability_proxy = np.mean(self.Y_ij, axis=0)
        log_b_mean = np.mean(np.log1p(self.B_ij), axis=0)
        corr = float(np.corrcoef(ability_proxy, log_b_mean)[0, 1]) if np.std(ability_proxy) > 0 and np.std(log_b_mean) > 0 else np.nan
        fail = np.isfinite(corr) and (abs(corr) > self.config.confounding_corr_thresh)
        return {
            'name': 'ability_budget_confounding',
            'status': 'FAIL' if fail else 'PASS',
            'message': f"Ability-Budget Confounding Corr={corr:.4f}",
            'key_numbers': (corr,),
            'key_number_names': ('ability_budget_corr',),
        }

    def _test_budget_response_association(self) -> Dict:
        """Item-wise |corr(Y_i·, X_i·)| proportion over fixed threshold (point-biserial style)."""
        corrs: List[float] = []
        for i in range(self.Q):
            y = self.Y_ij[i, :]
            x = self.X_ij[i, :]
            if np.std(y) <= 0 or np.std(x) <= 0:
                continue
            c = np.corrcoef(y, x)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(float(c)))
        if not corrs:
            return {
                'name': 'pb_corr_X_vs_Y', 'status': 'WARN',
                'message': 'No items with variance in both X and Y; cannot compute pb-corr.',
                'key_numbers': (0.0,), 'key_number_names': ('proportion_items_w_budget_signal_Y',),
            }
        corrs = np.asarray(corrs)
        prop_sig = float(np.mean(corrs > self.config.pb_corr_thresh))
        warn = prop_sig < self.config.pb_proportion_thresh
        return {
            'name': 'pb_corr_X_vs_Y', 'status': 'WARN' if warn else 'PASS',
            'message': f"Budget-Response Association: Signal Prop={prop_sig:.3f} with threshold={self.config.pb_corr_thresh:.2f}",
            'key_numbers': (prop_sig,), 'key_number_names': ('proportion_items_w_budget_signal_Y',),
        }

    def _test_budget_response_with_prob(self) -> Dict:
        """If P_ij provided, same association test between X and P."""
        if self.P_ij is None:
            return {'name': 'corr_X_vs_P', 'status': 'SKIP', 'message': 'P_ij not provided.', 'key_numbers': (), 'key_number_names': ()}
        corrs: List[float] = []
        for i in range(self.Q):
            p = self.P_ij[i, :]
            x = self.X_ij[i, :]
            if np.std(p) <= 0 or np.std(x) <= 0:
                continue
            c = np.corrcoef(p, x)[0, 1]
            if np.isfinite(c):
                corrs.append(abs(float(c)))
        if not corrs:
            return {'name': 'corr_X_vs_P', 'status': 'WARN', 'message': 'No variance in X or P across items.', 'key_numbers': (0.0,), 'key_number_names': ('proportion_items_w_budget_signal_P',)}
        prop_sig = float(np.mean(np.asarray(corrs) > self.config.pb_corr_thresh))
        warn = prop_sig < self.config.pb_proportion_thresh
        return {
            'name': 'corr_X_vs_P', 'status': 'WARN' if warn else 'PASS',
            'message': f"Budget-Response with PROB: Signal Prop={prop_sig:.3f} with threshold={self.config.pb_corr_thresh:.2f}",
            'key_numbers': (prop_sig,), 'key_number_names': ('proportion_items_w_budget_signal_P',),
        }

    # ---------- runner ----------
    def run_all_tests(self) -> Tuple[bool, bool, List[Dict]]:
        tests = [
            self._test_variance_entropy_coverage,
            self._test_fisher_proxy,
            self._test_budget_leverage,
            self._test_budget_matrix_effective_rank,
            self._test_ability_budget_confounding,
            self._test_budget_response_association,
            self._test_budget_response_with_prob,
        ]
        overall_pass = True
        any_warn = False
        results: List[Dict] = []
        for f in tests:
            res = f()
            results.append(res)
            if res['status'] in ('WARN', 'FAIL'):
                overall_pass = False
            if res['status'] == 'WARN':
                any_warn = True
            if self.print_msg:
                status, msg = res['status'], res['message']
                print(f"[{status:<4s}] {msg}")
        return overall_pass, any_warn, results

# ---------- Precheck-specific plotting ----------
def plot_precheck_results(all_results, plots_per_row=4):
    """
    Plots key numbers from DataPreChecker results across different scenarios.

    Returns:
        tuple: A tuple containing the matplotlib figure and axes objects.
    """
    # 1. Validate and unify the input format
    if not isinstance(all_results, (dict, list)):
        raise TypeError("Input 'all_results' must be a dictionary or a list.")
    if not all_results:
        raise ValueError("Cannot plot empty results. 'all_results' is empty.")

    scenarios = (all_results if isinstance(all_results, dict)
                 else {f'Scenario {i}': res for i, res in enumerate(all_results)})

    # 2. Pass 1: Discover all unique key names for consistent plotting
    all_key_names = set()
    for scenario_results in scenarios.values():
        for test_result in scenario_results:
            all_key_names.update(test_result.get('key_number_names', []))

    if not all_key_names:
        raise ValueError("No 'key_number_names' found in the results to plot.")

    sorted_key_names = sorted(list(all_key_names))
    scenario_labels = list(scenarios.keys())

    # 3. Pass 2: Build the data structure for plotting
    plot_data = {name: [] for name in sorted_key_names}
    for scenario_results in scenarios.values():
        # Extract all key-value pairs from the current scenario
        scenario_values = {}
        for test_result in scenario_results:
            names = test_result.get('key_number_names', [])
            numbers = test_result.get('key_numbers', [])
            scenario_values.update(zip(names, numbers))

        # Append data for each key, using None if a key is missing in this scenario
        for name in sorted_key_names:
            plot_data[name].append(scenario_values.get(name))

    # 4. Create the plots (this part was already quite clean)
    n_subplots = len(sorted_key_names)
    n_cols = plots_per_row
    n_rows = (n_subplots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), squeeze=False)
    axes = axes.flatten()

    for i, key_name in enumerate(sorted_key_names):
        ax = axes[i]
        values = plot_data[key_name]
        x_positions = np.arange(len(scenario_labels))

        # Handle potential None values for plotting by converting them to a number (e.g., 0 or NaN)
        # Using 0 is simple and visually effective for bar charts.
        values_for_plotting = [v if v is not None else 0 for v in values]

        ax.bar(x_positions, values_for_plotting)
        ax.set_title(key_name)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(scenario_labels, rotation=90)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, axis='y', alpha=0.5)

    # Hide any unused subplots
    for j in range(n_subplots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes