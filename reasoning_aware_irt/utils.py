import numpy as np
from numpy import ndarray
from typing import *

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .gem import CalibrationResult

def prob_correct_3pl(theta, a, b, c):
    """Calculates the probability of a correct response for a given theta and 3PL parameters."""
    # Ensure the argument to exp is within a reasonable range to avoid overflow/underflow
    exp_arg = -a[:, None] * (theta[None, :] - b[:, None])
    exp_arg = np.clip(exp_arg, -30, 30)
    return c[:, None] + (1 - c[:, None]) / (1 + np.exp(exp_arg))

def item_variance_3pl(theta, a, b, c):
    """Calculates the variance for a single item at a given theta level."""
    P_theta = prob_correct_3pl(theta, a, b, c)
    return P_theta * (1 - P_theta)

def item_information_function_3pl(theta, a, b, c):
    """Calculates the IIF for a given theta and 3PL parameters."""
    P_theta = prob_correct_3pl(theta, a, b, c)
    # Ensure P_theta is not exactly 0 or 1 to avoid division by zero or log(0)
    P_theta = np.clip(P_theta, 1e-9, 1 - 1e-9)
    numerator = (P_theta - c[:, None])**2
    denominator = (1 - c)**2

    # TODO: handle this when denominator is 1D array
    # if np.isclose(denominator, 0): # Handle case where c is close to 1
    #     return np.zeros_like(theta) if np.isscalar(theta) else np.zeros(len(theta))
    
    # Use the simplified formula
    # I(theta) = a^2 * ((P(theta) - c)/(1 - c))^2 * (1 - P(theta))/P(theta)
    iif = (a[:, None]**2) * (numerator / denominator[:, None]) * ((1 - P_theta) / P_theta)
    return iif

# =============================================
# utils for CalibrationResult | toy_data visualization
# ---------------------------------------------
def create_item_information_figure(a_params, b_params, c_params, item_labels, theta_range=None):
    """
    Creates a Plotly figure of the Item Information Functions (IIF) for selected items.

    Args:
        theta_range: A numpy array of theta values.
        item_ids: A list of item IDs (strings).
        a_params: A list or numpy array of 'a' parameters for each item.
        b_params: A list or numpy array of 'b' parameters for each item.
        c_params: A list or numpy array of 'c' parameters for each item.

    Returns:
        A Plotly go.Figure object.
    """
    if not (len(item_labels) == len(a_params) == len(b_params) == len(c_params)):
        raise ValueError("Input lists/arrays for item_ids, a_params, b_params, and c_params must have the same length.")

    if theta_range is None:
        theta_range = np.linspace(-10, 10, 100)

    # Calculate IIF for the current item across the theta range
    iif_values = item_information_function_3pl(theta_range, a_params, b_params, c_params)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the trace to the figure
    for i in range(len(item_labels)):
        fig.add_trace(go.Scatter(x=theta_range, y=iif_values[i], mode='lines', name=item_labels[i]))

    # Update layout for better visualization
    fig.update_layout(
        title=f'Item Information Functions (IIF) for Selected Simulated Items',
        xaxis_title='Theta (Latent Ability)',
        yaxis_title='Information',
        hovermode='closest'
    )

    return fig

def create_test_information_figure(a_params, b_params, c_params, item_labels, theta_range=None):
    if not (len(item_labels) == len(a_params) == len(b_params) == len(c_params)):
        raise ValueError("Input lists/arrays for item_ids, a_params, b_params, and c_params must have the same length.")

    if theta_range is None:
        theta_range = np.linspace(-10, 10, 100)

    # Calculate IIF for the current item across the theta range
    iif_values = item_information_function_3pl(theta_range, a_params, b_params, c_params)
    tif_values = np.sum(iif_values, axis=0)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the TIF trace
    fig.add_trace(go.Scatter(x=theta_range, y=tif_values, mode='lines', name='Test Information Function'))

    # Update layout for better visualization
    fig.update_layout(
        title='Test Information Function (TIF) for the Simulated Item Bank',
        xaxis_title='Theta (Latent Ability)',
        yaxis_title='Information',
        hovermode='closest',
        xaxis=dict(
            tickmode='linear',
            dtick=1
        )
    )
    return fig

def plot_test_characteristic_curve(
    item_params: Dict[str, ndarray],
    theta_range: ndarray = None,
):
    """
    Calculates and plots the Test Characteristic Curve (TCC) along with reliability bounds
    (Expected Score +/- 2 Standard Deviations) using the 3-parameter logistic (3PL) model.

    Args:
        item_params: Dictionary containing 1D NumPy arrays for item parameters
                     'a', 'b', and 'c', all of length Q (number of items).
        theta_range: 1D NumPy array of ability (theta) values for the plot domain.

    Returns:
        A Plotly figure object (go.Figure) showing the TCC and the 2-SD shaded region.
    """
    
    if theta_range is None:
        theta_range = np.linspace(-10, 10, 100)

    a = item_params['a_i_true']
    b = item_params['b_i_true']
    c = item_params['c_i_true']

    # --- Calculate P(theta) and Variance ---

    # Calculate P(theta) for all items. Resulting shape: (Q, N_theta)
    probabilities_matrix = prob_correct_3pl(theta_range, a, b, c)

    # Calculate TCC by summing probabilities across items (axis=0 for Q items).
    # Resulting shape: (N_theta,)
    tcc_values = np.sum(probabilities_matrix, axis=0)

    # Calculate variance for each item. Resulting shape: (Q, N_theta)
    variance_matrix = item_variance_3pl(theta_range, a, b, c)

    # Calculate total variance (additive assumption). Resulting shape: (N_theta,)
    total_variance_values = np.sum(variance_matrix, axis=0)

    # Calculate standard deviation of the expected score
    sd_expected_score_values = np.sqrt(total_variance_values)

    # --- Create Uncertainty Bounds ---
    upper_bound = tcc_values + 2 * sd_expected_score_values
    lower_bound = tcc_values - 2 * sd_expected_score_values

    # --- Plotly Figure Generation ---
    fig = go.Figure()

    # Add the TCC trace
    fig.add_trace(go.Scattergl(
        x=theta_range,
        y=tcc_values,
        mode='lines',
        name='Test Characteristic Curve (Expected Score)',
        line=dict(color='blue', width=2)
    ))

    # Add the shaded region (Expected Score +/- 2 SD)
    fig.add_trace(go.Scattergl(
        x=np.concatenate([theta_range, theta_range[::-1]]),
        y=np.concatenate([lower_bound, upper_bound[::-1]]),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fill='toself',
        fillcolor='rgba(0, 100, 80, 0.3)',
        name='Expected Score +/- 2 SD'
    ))

    # Add boundary lines (invisible)
    fig.add_trace(go.Scattergl(
        x=theta_range,
        y=lower_bound,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0)'),
        showlegend=False
    ))

    fig.add_trace(go.Scattergl(
        x=theta_range,
        y=upper_bound,
        mode='lines',
        line=dict(color='rgba(0, 0, 0, 0)'),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title='Test Characteristic Curve (TCC) with Standard Deviation Uncertainty (+/- 2 SD)',
        xaxis_title='Theta (Latent Ability)',
        yaxis_title='Score',
        hovermode='x unified'
    )

    return fig

def plot_b_theta_histograms(ax, generative_data: dict | CalibrationResult, 
                            title: str, avg_Y: float | None = None):
    """Single-panel histogram plot: θ_j_true, b_i_true, and optional θ_j_max overlay.
    generative_data is either an output from the paper's GEM (CalibrationResult type)
    or a simulated toy data from .toy_scenarios.make_reasoning_aware_3pl_toy (dict type)
    """

    if isinstance(generative_data, dict):
        theta = generative_data.get('theta_j_true', None)
        b = generative_data.get('b_i_true', None)
        tmax = generative_data.get('theta_j_max', None)
    elif isinstance(generative_data, CalibrationResult):
        theta = generative_data.theta_j_mean
        b = generative_data.b_i
        tmax = theta + generative_data.gamma_j * np.max(generative_data.info["data_input"].B_ij, axis=0)

    all_values = np.concatenate([b, theta, tmax])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    bins = np.linspace(min_val, max_val, 50)

    if b is not None:
        ax.hist(b, bins=bins, alpha=0.7, label='b_i_true', color='skyblue')
    if theta is not None:
        ax.hist(theta, bins=bins, alpha=0.7, label='theta_j_true', color='lightcoral')
    if tmax is not None:
        ax.hist(tmax, bins=bins, alpha=0.7, label='theta_j_max', color='lightgreen')
    ax.set_title(title)
    if avg_Y is not None:
        ax.text(0.95, 0.95, f'Avg Y: {avg_Y:.3f}',
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes, # <-- Key change: Use ax.transAxes
                color='black', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    ax.legend()
    ax.grid(True, linewidth=0.4, alpha=0.5)

def plot_multiple_b_theta_histograms(generative_data_list: Sequence[dict | CalibrationResult], 
                                     data_names_list: Sequence[str], 
                                     plots_per_row: int = 4, 
                                     avg_Y_list: Sequence[float] | None = None):
    """Quick grid using plot_b_theta_histograms, with optional mean(Y) subtitles.
    """
    m = len(generative_data_list)
    cols = min(plots_per_row, m)
    rows = int(np.ceil(m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.8*rows), squeeze=False)
    for idx, gen_data in enumerate(generative_data_list):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        avgY = None
        if avg_Y_list is not None and idx < len(avg_Y_list):
            avgY = avg_Y_list[idx]
        plot_b_theta_histograms(ax, gen_data, data_names_list[idx], avgY)
    # hide unused axes
    for k in range(m, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')
    fig.tight_layout()
    return fig, axes