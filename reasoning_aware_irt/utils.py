import numpy as np
from numpy import ndarray
from typing import *

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import plotly.graph_objects as go

from .gem import CalibrationResult
from .classic_irt_em import ClassicCalibrationResult

def prob_correct_3pl(theta, a, b, c):
    """Calculates the probability of a correct response for a given theta and 3PL parameters.
    theta is 1D N-dimensional
    a,b,c are 1D Q-dimensional

    return:
    prob_matrix 2D-array of shape (Q, N)
    """
    # Ensure the argument to exp is within a reasonable range to avoid overflow/underflow
    exp_arg = -a[:, None] * (theta[None, :] - b[:, None])
    exp_arg = np.clip(exp_arg, -30, 30)
    return c[:, None] + (1 - c[:, None]) / (1 + np.exp(exp_arg))

def item_variance_3pl(theta, a, b, c):
    """Calculates the variance for a single item at a given theta level.
    
    arguments and return shapes are the same as prob_correct_3pl
    """
    P_theta = prob_correct_3pl(theta, a, b, c)
    return P_theta * (1 - P_theta)

def item_information_function_3pl(theta, a, b, c):
    """Calculates the IIF for a given theta and 3PL parameters.
    
    arguments and return shapes are the same as prob_correct_3pl
    """
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

def location_scale_transform(x, mu=None, sigma=None):
    if mu is None: # assuming both mu & sigma are none for consistency here
        mu = x.mean()
        mu_sq = (x**2).mean()

        variance = mu_sq - mu**2
        sigma = np.sqrt(max(variance, 1e-12))
    
    x_new = (x-mu)/sigma
    return x_new, mu, sigma

# =============================================
# utils for raw data (score matrix) visualization
# ---------------------------------------------
def visualize_score_matrix(score_matrix, tester_threshold_idx=None, question_threshold_idx=None):
    """
    Visualizes the summations along axis 0 and axis 1 of a score matrix.

    Args:
        score_matrix (np.ndarray): The score matrix (e.g., Y_ij).
        tester_threshold_idx (int, optional): Index to draw a vertical line on the tester raw scores plot and color points differently.
        question_threshold_idx (int, optional): Index to draw a vertical line on the question raw scores plot and color points differently.
    """
    # Calculate summations
    sum_axis_0 = score_matrix.sum(axis=0)
    sum_axis_1 = score_matrix.sum(axis=1)

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot summation along axis 0 (histogram)
    axes[0, 0].set_title('Summation along Axis 0 (Testers)')
    axes[0, 0].set_xlabel('Total Score')
    axes[0, 0].set_ylabel('Frequency')
    if tester_threshold_idx is not None:
        axes[0, 0].hist(sum_axis_0[:tester_threshold_idx], bins=20, alpha=0.7, label='Group 1', color='skyblue')
        axes[0, 0].hist(sum_axis_0[tester_threshold_idx:], bins=20, alpha=0.7, label='Group 2', color='orange')
        axes[0, 0].legend()
    else:
        axes[0, 0].hist(sum_axis_0, bins=20)

    # Plot summation along axis 1 (histogram)
    axes[0, 1].set_title('Summation along Axis 1 (Questions)')
    axes[0, 1].set_xlabel('Total Score')
    axes[0, 1].set_ylabel('Frequency')
    if question_threshold_idx is not None:
        axes[0, 1].hist(sum_axis_1[:question_threshold_idx], bins=20, alpha=0.7, label='Group 1', color='red')
        axes[0, 1].hist(sum_axis_1[question_threshold_idx:], bins=20, alpha=0.7, label='Group 2', color='green')
        axes[0, 1].legend()
    else:
        axes[0, 1].hist(sum_axis_1, bins=20, color='darkred')


    # Plot raw values of summation along axis 0
    axes[1, 0].set_title('Raw Values of Summation along Axis 0 (Testers)')
    axes[1, 0].set_xlabel('Tester Index')
    axes[1, 0].set_ylabel('Total Score')
    if tester_threshold_idx is not None:
        axes[1, 0].axvline(x=tester_threshold_idx, color='k', linestyle='--')
        axes[1, 0].plot(np.arange(tester_threshold_idx), sum_axis_0[:tester_threshold_idx], marker='o', linestyle='', color='skyblue', label='Group 1')
        axes[1, 0].plot(np.arange(tester_threshold_idx, len(sum_axis_0)), sum_axis_0[tester_threshold_idx:], marker='o', linestyle='', color='orange', label='Group 2')
        axes[1, 0].legend()
    else:
        axes[1, 0].plot(sum_axis_0, marker='o', linestyle='')


    # Plot raw values of summation along axis 1
    axes[1, 1].set_title('Raw Values of Summation along Axis 1 (Questions)')
    axes[1, 1].set_xlabel('Question Index')
    axes[1, 1].set_ylabel('Total Score')
    if question_threshold_idx is not None:
        axes[1, 1].axvline(x=question_threshold_idx, color='k', linestyle='--')
        axes[1, 1].plot(np.arange(question_threshold_idx), sum_axis_1[:question_threshold_idx], marker='o', linestyle='', color='red', label='Group 1')
        axes[1, 1].plot(np.arange(question_threshold_idx, len(sum_axis_1)), sum_axis_1[question_threshold_idx:], marker='o', linestyle='', color='green', label='Group 2')
        axes[1, 1].legend()
    else:
         axes[1, 1].plot(sum_axis_1, marker='o', linestyle='', color='darkred')


    plt.tight_layout()
    plt.show()
    
# =============================================
# utils for CalibrationResult | data & model parameters visualization
# ---------------------------------------------

def create_item_information_figure(a_params, b_params, c_params, item_labels, theta_range=None):
    """
    Creates a Plotly figure of the Item Information Functions (IIF) for each of selected items.

    Args:
        a_params: A (Q,) numpy array of 'a'.
        b_params: A (Q,) numpy array of 'b'.
        c_params: A (Q,) numpy array of 'c'.
        item_labels: A list of Q item labels (strings).
        theta_range: A (N,) numpy array of theta values (Default = linspace(-10, 10, N) with N=100)

    Returns:
        A Plotly go.Figure object.
    """
    if not (len(item_labels) == len(a_params) == len(b_params) == len(c_params)):
        raise ValueError("Input lists/arrays for item_ids, a_params, b_params, and c_params must have the same length.")

    if theta_range is None:
        theta_range = np.linspace(-10, 10, 100) # shape (N,)

    # Calculate IIF of shape (Q,N)
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
    """
    Creates a Plotly figure of the test Information Functions (TIF) for all selected items.

    Arguments are the same as create_item_information_figure
    """
    if not (len(item_labels) == len(a_params) == len(b_params) == len(c_params)):
        raise ValueError("Input lists/arrays for item_ids, a_params, b_params, and c_params must have the same length.")

    if theta_range is None:
        theta_range = np.linspace(-10, 10, 100)

    # Calculate IIF for the current item across the theta range
    iif_values = item_information_function_3pl(theta_range, a_params, b_params, c_params)
    tif_values = np.sum(iif_values, axis=0) # (1, N)

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
    probabilities_matrix = prob_correct_3pl(theta_range, a, b, c) # (Q,N)
    tcc_values = np.sum(probabilities_matrix, axis=0) # (N,)

    variance_matrix = item_variance_3pl(theta_range, a, b, c) # (Q,N)
    total_variance_values = np.sum(variance_matrix, axis=0) # (N,)

    # --- Create Uncertainty Bounds ---
    sd_expected_score_values = np.sqrt(total_variance_values)
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

def plot_b_theta_histograms(ax, generative_data: dict | CalibrationResult | ClassicCalibrationResult, 
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
    elif isinstance(generative_data, ClassicCalibrationResult):
        theta = generative_data.theta_j_mean
        b = generative_data.b_i
        tmax = None

    all_values = np.concatenate([b, theta])
    if tmax is not None:
        all_values = np.concatenate([all_values, tmax])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    bins = np.linspace(min_val, max_val, 50)

    if b is not None:
        ax.hist(b, bins=bins, alpha=0.7, label='b_i', color='skyblue')
    if theta is not None:
        ax.hist(theta, bins=bins, alpha=0.7, label='theta_j', color='lightcoral')
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

def plot_multiple_b_theta_histograms(generative_data_list: Sequence[dict | CalibrationResult | ClassicCalibrationResult], 
                                     data_names_list: Sequence[str], 
                                     plots_per_row: int = 4, 
                                     avg_Y_list: Sequence[float] | None = None):
    """Quick grid using plot_b_theta_histograms, with optional mean(Y) subtitles.
    generative_data is either an output from the paper's GEM (CalibrationResult type)
    or any data of dict type containing keys 'theta_j_true', 'b_i_true' and 'theta_j_max'
    (see .toy_scenarios.make_reasoning_aware_3pl_toy )
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

def plot_parameter_scatter_with_se(
    ax: Axes,
    true_values: Union[np.ndarray, List[float]],
    estimated_values: Union[np.ndarray, List[float]],
    estimated_se: Union[np.ndarray, List[float], None] = None,
    param_label: str = "",
    min_lim: float | None = None,
    max_lim: float | None = None,
    x_label: str | None = None, 
    y_label: str | None = None,
    title: str | None = None,
):
    """
    Generates a scatter plot of estimated vs true parameter values,
    including standard error (SE) bars on the estimated values (Y-axis).
    """
    # 1. Plot the Central Estimates (Scatter Plot)
    # We use ax.scatter instead of sns.scatterplot here because we'll add
    # error bars in a separate layer, often simplifying the control.
    ax.scatter(x=true_values, y=estimated_values, color='blue', alpha=0.6, s=20) # , label=None

    # 2. Add Error Bars based on estimated_se
    # The 'yerr' argument takes the standard error (or range) for the vertical axis (estimated values).
    # We plot the error bars without re-plotting the central point (fmt='none').
    if estimated_se is None:
        estimated_se = np.zeros_like(estimated_values)

    ax.errorbar(
        x=true_values,
        y=estimated_values,
        yerr=estimated_se, # Estimated SE provides the error distance
        fmt='none',       # Do not plot the data points again
        ecolor='gray',    # Error bar color
        elinewidth=0.8,   # Error bar line thickness
        capsize=3,        # Size of the caps on the error bars
        alpha=0.5,        # Transparency
        zorder=0          # Ensure error bars are plotted underneath the central points
    )

    # 3. Set Labels, Title, and 1:1 Line (Unchanged)
    if x_label is None or y_label is None or title is None: # default true vs. estimated
        x_label = f'True {param_label}'
        y_label = f'Estimated {param_label}'
        title = f'Estimated {param_label} vs. True'
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Calculate global min/max for the 1:1 line and axis limits
    # We should consider the full range of the error bars for robust axis limits
    min_data = min(true_values.min(), (estimated_values - estimated_se).min())
    max_data = max(true_values.max(), (estimated_values + estimated_se).max())

    # Add a small buffer for visualization
    buffer = (max_data - min_data) * 0.05
    min_val = min_data - buffer
    max_val = max_data + buffer

    # Plot the 1:1 reference line
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', 
            # label='1:1 Line'
            )

    # Set axis limits based on buffered min/max except specified by a user
    if min_lim is None:
        min_lim = min_val
    if max_lim is None:
        max_lim = max_val

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
