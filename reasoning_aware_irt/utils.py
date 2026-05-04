import numpy as np
import pandas as pd
import math
from numpy import ndarray
from typing import *

from scipy import stats
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import plotly.graph_objects as go
import seaborn as sns

from .gem_item_gate import CalibrationResult
from .classic_irt_em import ClassicCalibrationResult

def prob_correct_3pl(theta_j, a_i, b_i, c_i, gamma_j=None, kappa_j=None, x_ij=None):
    """Calculates the probability of a correct response for a given theta and 3PL parameters.
    theta_j : (N,)
    a_i,b_i,c_i : (Q,)

    --Optional for Reasoning-Aware 3PL--
    gamma_j, kappa_j : (N,)
    x_ij : (Q,N) -- i.e. np.log1p(B_ij)
    
    return:
    prob_matrix 2D-array of shape (Q, N)
    """
    # Ensure the argument to exp is within a reasonable range to avoid overflow/underflow
    Q = len(b_i)
    N = len(theta_j)
    assert len(a_i)==Q and len(c_i)==Q
    
    if gamma_j is not None:
        assert len(gamma_j)==N
        assert x_ij.shape==(Q,N)
        
        if kappa_j is not None: # quad Reasoning-3PL
            assert len(kappa_j)==N
            diff_ij = (theta_j[None, :] + gamma_j[None, :]*x_ij + kappa_j[None, :]*x_ij**2) - b_i[:, None]
        else: # linear Reasoning-3PL
            diff_ij = (theta_j[None, :] + gamma_j[None, :]*x_ij) - b_i[:, None]
    else: # classic 3PL
        diff_ij = theta_j[None, :] - b_i[:, None]
    
    exp_arg = -(a_i[:, None] * diff_ij)
    exp_arg = np.clip(exp_arg, -30, 30)
    return c_i[:, None] + (1 - c_i[:, None]) / (1 + np.exp(exp_arg))

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
    se_axis="x_axis",
    param_label: str = "",
    min_ylim: float | None = None,
    max_ylim: float | None = None,
    min_xlim: float | None = None,
    max_xlim: float | None = None,
    x_label: str | None = None, 
    y_label: str | None = None,
    title: str | None = None,
    plot_ref_line: bool = True,
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

    xerr = np.zeros_like(estimated_values)
    yerr = np.zeros_like(estimated_values)
    if se_axis=="x_axis":
        xerr = estimated_se
    else:
        yerr = estimated_se

    ax.errorbar(
        x=true_values,
        y=estimated_values,
        xerr=xerr,
        yerr=yerr, # Estimated SE provides the error distance
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
    min_data = min(true_values.min(), np.nanmin(estimated_values - estimated_se))
    max_data = max(true_values.max(), np.nanmax(estimated_values + estimated_se))

    # Add a small buffer for visualization
    buffer = (max_data - min_data) * 0.05
    min_val = min_data - buffer
    max_val = max_data + buffer

    # Plot the 1:1 reference line
    if plot_ref_line:
        ax.plot([min_xlim, max_xlim], [max_ylim, min_ylim], color='red', linestyle='--', 
                # label='1:1 Line'
                )

    # Set axis limits based on buffered min/max except specified by a user
    if min_ylim is None:
        min_ylim = min_val
    if max_ylim is None:
        max_ylim = max_val
    if min_xlim is None:
        min_xlim = min_val
    if max_xlim is None:
        max_xlim = max_val

    ax.set_xlim(min_xlim, max_xlim)
    ax.set_ylim(min_ylim, max_ylim)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

#######################################################################
## paper plot
def plot_overlaid_barplot(ax, categories, vals_bg, vals_fg, title, x_label="x-axis", color='g'):
    """
    Plots a single overlaid bar chart on a specific matplotlib axis.
    """
    # Build the dataframe dynamically
    df = pd.DataFrame({
        'category': categories,
        'background': vals_bg,
        'foreground': vals_fg
    })
    
    # Plot the background (pastel) bars
    sns.set_color_codes("pastel")
    sns.barplot(x="background", y="category", data=df, color=color, ax=ax)
    
    # Plot the foreground (muted) bars over them
    sns.set_color_codes("muted")
    sns.barplot(x="foreground", y="category", data=df, color=color, ax=ax)
    
    # Formatting
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('') # Replaces your `y=""` trick

def plot_grid(N, M, datasets, pdf_name=None, 
              plot_func=plot_overlaid_barplot, subplot_titles=[],
              M_multiplier=4, N_multiplier=1.5, plot_single_legend=False):
    """
    Creates an N x M grid of subplots and populates them using plot_overlaid_barplot.
    
    datasets: A list of dictionaries containing the data for each subplot.
    """
    # 1. Global style setup
    # sns.set_context('paper', font_scale=1.2)
    # sns.set_style('whitegrid')
    
    # 2. Create the figure and N x M axes
    # Adjust figsize based on the grid dimensions
    fig, axes = plt.subplots(N, M, figsize=(M * M_multiplier, N * N_multiplier))
    
    # Ensure axes is a 1D array so we can easily loop over it, 
    # even if N=1 or M=1
    axes_flat = np.atleast_1d(axes).flatten()
    
    # 3. Loop through the datasets and plot
    for i, data in enumerate(datasets):
        if i >= len(axes_flat):
            print(f"Warning: More datasets ({len(datasets)}) than subplots ({len(axes_flat)}).")
            break
            
        ax = axes_flat[i]
        
        if plot_func==plot_overlaid_barplot:
            plot_func(ax=ax, categories=data['categories'],
                      vals_bg=data['bg'], vals_fg=data['fg'],
                      title=data['title'], x_label=data['x_label'],
                      color=data['color'])
        else: # DEFAULT alternative: plot_dumbbell - 1 dumbbell plot = multi bar-graph plots
            group_data = data # it's actually a group of datasets - (superset of bar-graph data), different data preparation
            subplot_title = subplot_titles[i]
            categories = group_data[0]['categories']

            plot_legend = True if i==0 or not plot_single_legend else False
            plot_dumbbell(ax, group_data, categories, title=subplot_title, plot_legend=plot_legend)
        
    # 4. Hide any unused subplots (if datasets < N * M)
    for j in range(len(datasets), len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    # Prevent labels from overlapping
    plt.tight_layout()
    plt.show()

    if pdf_name is not None:
        print(f'save figure-grid to {pdf_name}')
        fig.savefig(pdf_name, format="pdf", bbox_inches="tight", dpi=300, pad_inches=0.0, transparent=True)

def plot_dumbbell(ax=None, datasets=None, categories=None, x_label="", y_labels=None,
                  title="", plot_legend=True, figsize=(10, 6), save_path=None, x_ticks=None,
                  nickname=[], x_nickname=0.52, y_nickname=0.9, se_vals=None, **argv):
    """
    Plots a single dumbbell subplot. Each dataset in 'datasets' becomes one y-axis row.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if y_labels is None:
        y_labels = [d['title'] for d in datasets]
    colors = sns.color_palette("Set2", len(categories)) # Distinct colors for legend
    
    for i, data in enumerate(datasets):
        vals = data['bg'] #data['values']
        se_vals = data.get('se_vals', None)
        # 1. Draw the horizontal line segment connecting the min and max values
        ax.plot([min(vals), max(vals)], [i, i], color='gray', alpha=0.4, zorder=1, lw=2.5)
        
        # 2. Draw the dots for each category
        for j, val in enumerate(vals):
            # Only assign the label on the first row so the legend doesn't duplicate
            leg_label = categories[j] if i == 0 else ""
            ax.scatter(val, i, color=colors[j], s=120, zorder=2, 
                       label=leg_label, edgecolor='w', alpha=0.7)


            # --- Draw the horizontal error bar ---
            if se_vals is not None and j < len(se_vals):
                ax.errorbar(val, i, xerr=se_vals[j], fmt='none', ecolor=colors[j], 
                            capsize=4, capthick=1.5, zorder=1.5, alpha=0.7)
            
            if len(nickname)>0 and i==0:
                ax.text(x_nickname, y_nickname-j*0.1, nickname[j], 
                color=colors[j], 
                fontsize=10, 
                transform=ax.transAxes)
            
    # Formatting
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(y_labels)
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    ax.set_xlabel(x_label)
    if title: ax.set_title(title, fontsize=9)
    
    # Place legend cleanly outside the plot or in a good spot
    if plot_legend:
        ax.legend(title="Method", loc='center left', bbox_to_anchor=(1, 0.5))
    if save_path is not None:
        print(f'save fig to {save_path} ... success!')
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.0, transparent=True)

def _plot_single_heatmap(ax, corr_matrix, title):
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar_kws={'label': 'Rank Correlation Coefficient'},
        linewidths=.5,
        linecolor='black',
        vmin=0, vmax=1,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Permutation Index', fontsize=12)
    ax.set_ylabel('Permutation Index', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

def plot_correlation_matrices_dynamic(corr_matrices, titles=None):
    n = len(corr_matrices)
    if n == 0:
        print("No correlation matrices provided to plot.")
        return

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 7))
    # Ensure axes is always a 2D array even if nrows or ncols is 1
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    for i in range(n):
        row = i // ncols
        col = i % ncols
        current_title = titles[i] if titles and i < len(titles) else f'Correlation Matrix {i + 1}'
        _plot_single_heatmap(axes[row, col], corr_matrices[i], current_title)

    # Turn off any unused subplots
    for i in range(n, nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

def plot_binned_with_quad_fit(x, y, num_bins=10, best_bin_thresh=30, min_points_per_bin=3, use_quantiles=False,
                              ax=None, return_best_bin=False, plot_quad=False, x_label="x-axis", y_label="y-axis",
                              display_legend=True, color='blue', label='Binned Data', alpha=0.7, size_norm=1,
                              show_count=True):
    x, y = np.asarray(x), np.asarray(y)
    if not use_quantiles: # remove sparse left&right data
        p02, p98 = np.percentile(x, [2, 98])
        valid_mask = (x >= p02) & (x <= p98)
        x, y = x[valid_mask], y[valid_mask]
    if len(x) == 0:
        print("No data to plot."); return
    
    # 1. Setup Bins
    x_min, x_max = x.min(), x.max()
    if x_min == x_max:
        bin_edges = np.array([x_min - 0.5, x_min + 0.5])
        bin_centers = np.array([x_min])
    else:
        if use_quantiles:
            # Quantile binning: forces roughly equal number of points per bin
            probabilities = np.linspace(0, 1, num_bins + 1)
            bin_edges = np.quantile(x, probabilities)
            # If many identical 'x' values exist, quantiles can overlap. 
            # np.unique prevents errors but might slightly reduce num_bins in extreme edge cases.
            bin_edges = np.unique(bin_edges) 
        else:
            # Traditional equal-width binning (will create sparse bins if data is skewed)
            bin_edges = np.linspace(x_min, x_max, num_bins + 1)
            
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2. Calculate Statistics using binned_statistic
    y_averages = binned_statistic(x, y, statistic='mean', bins=bin_edges).statistic
    n_points_per_bin = binned_statistic(x, y, statistic='count', bins=bin_edges).statistic

    # 3. Setup Plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Mask 1: For plotting dots (all bins with data)
    has_data_mask = ~np.isnan(y_averages) & (n_points_per_bin > min_points_per_bin)
    
    if not np.any(has_data_mask):
        ax.text(0.5, 0.5, "No data points found.", transform=ax.transAxes, ha='center')
        return

    # Scatter plot with scaled sizes
    max_pts = n_points_per_bin.max()
    sizes = (n_points_per_bin[has_data_mask] / max_pts * 400 + 20)
    ax.scatter(bin_centers[has_data_mask], y_averages[has_data_mask], 
               s=sizes*size_norm, alpha=alpha, c=[color], edgecolors='w', label=label)

    # Add count labels
    if show_count:
        for i in np.where(has_data_mask)[0]:
            ax.text(bin_centers[i], y_averages[i], f'{int(n_points_per_bin[i])}', 
                    fontsize=12, ha='center', va='bottom', weight='bold')

    # 4. Constrained Fitting
    # Mask 2: For fitting (only high-confidence bins)
    has_data_mask2 = has_data_mask & (n_points_per_bin > best_bin_thresh)
    best_bin = None

    if np.any(has_data_mask2):
        fit_x = bin_centers[has_data_mask2]
        fit_y = y_averages[has_data_mask2]
        best_bin = fit_x[np.argmax(fit_y)]

        try:
            # Curve fit with a non-zero guess for 'a' to help convergence
            popt, _ = curve_fit(quadratic_func, fit_x, fit_y, 
                                p0=[-1e-6, 1e-3, np.mean(fit_y)], 
                                bounds=([-np.inf, 0, -np.inf], [0, np.inf, np.inf]))

            if plot_quad:
                x_fit = np.linspace(x_min, x_max, 100)
                ax.plot(x_fit, quadratic_func(x_fit, *popt), 'r--', label='Constrained Fit')
            
                # Formatting info text
                text_str = f"a={popt[0]:.4e}\nb={popt[1]:.4e}\nc={popt[2]:.2f}\nbest_bin={best_bin:.0f}"
                ax.text(0.05, 0.95, text_str, transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        except Exception as e:
            print(f"Fit failed: {e}")

    # Final visual adjustments
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if display_legend:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    if return_best_bin:
        return best_bin
    else:
        return None
    
def plot_combined_binned_data(datasets, save_path=None, xlabel="X-axis", ylabel="Y-axis", 
                              title="Combined Binned Data",figsize=(10, 6),
                              show_legend=True, nickname=False, x_nickname=0.03, y_nickname=0.4):
    """
    Plots multiple binned datasets onto a single 2D figure with transparency and a legend.
    """
    fig, ax = plt.subplots(figsize=figsize)
    best_bin_list = []
    
    # Generate distinct colors for the datasets
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)+1))

    for i, data in enumerate(datasets):
        plot_function = data.get('plot_function')
        kwargs = data.get('plot_function_argv', {}).copy()
        
        # Inject styling into the underlying plot function
        kwargs['color'] = colors[i+1]
        kwargs['label'] = data.get('title', f'Dataset {i+1}')
        kwargs['alpha'] = 0.6  # Make transparent so overlapping bins are visible
        
        best_bin = plot_function(data['x'], data['y'], ax=ax, **kwargs)
        best_bin_list.append(best_bin)
        
        if nickname:
            ax.text(x_nickname, y_nickname-i*0.1, data['nickname'], 
            color=colors[i+1], 
            fontsize=9, 
            transform=ax.transAxes)

    # Global formatting
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Fix the legend (avoids duplicate legends if the underlying function calls ax.legend())
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Dictionary trick to remove duplicates if any
        by_label = dict(zip(labels, handles))
        if show_legend:
            ax.legend(by_label.values(), by_label.keys(), framealpha=0.9)

    plt.tight_layout()
    
    if save_path is not None:
        print(f'save fig to {save_path} ... success!')
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.0, transparent=True)
        
    plt.show()
    return best_bin_list

def plot_bins_2d(X, y, num_bins=10, min_points_per_bin=3, use_quantiles=False,
                 ax=None, fig=None, row=None, col=None, use_plotly=False, return_best_bin=False, 
                 x1_label="Feature 1", x2_label="Feature 2", z_label="Target (Y)", alpha=0.8, box_aspect=(1, 1, 1),
                 label="", cmap='viridis', figsize=(10, 6), show_colorbar=True, save_path=None, show_count=True,
                 title=None, title_pad=0, **kwargs):
    
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2: raise ValueError("X must be a 2D array [N, 2]")
    x1, x2 = X[:, 0], X[:, 1]
    
    if not use_quantiles:
        p02_x1, p98_x1 = np.percentile(x1, [2, 98])
        p02_x2, p98_x2 = np.percentile(x2, [2, 98])
        valid_mask = (x1 >= p02_x1) & (x1 <= p98_x1) & (x2 >= p02_x2) & (x2 <= p98_x2)
        x1, x2, y = x1[valid_mask], x2[valid_mask], y[valid_mask]
        
    if len(x1) == 0: return None

    # 1. Setup Bins
    def get_bin_edges(data):
        if np.min(data) == np.max(data): return np.array([np.min(data) - 0.5, np.min(data) + 0.5])
        if use_quantiles: return np.unique(np.quantile(data, np.linspace(0, 1, num_bins + 1)))
        return np.linspace(np.min(data), np.max(data), num_bins + 1)

    bin_edges_x1, bin_edges_x2 = get_bin_edges(x1), get_bin_edges(x2)
    bin_centers_x1 = (bin_edges_x1[:-1] + bin_edges_x1[1:]) / 2
    bin_centers_x2 = (bin_edges_x2[:-1] + bin_edges_x2[1:]) / 2

    # 2. Calculate Statistics
    stats_mean = binned_statistic_2d(x1, x2, y, statistic='mean', bins=[bin_edges_x1, bin_edges_x2])
    stats_count = binned_statistic_2d(x1, x2, y, statistic='count', bins=[bin_edges_x1, bin_edges_x2])
    
    CX1, CX2 = np.meshgrid(bin_centers_x1, bin_centers_x2, indexing='ij')

    # Masking
    cx1_flat, cx2_flat = CX1.flatten(), CX2.flatten()
    y_flat, counts_flat = stats_mean.statistic.flatten(), stats_count.statistic.flatten()

    has_data_mask = ~np.isnan(y_flat) & (counts_flat > min_points_per_bin)
    if not np.any(has_data_mask): return None

    plot_x1, plot_x2 = cx1_flat[has_data_mask], cx2_flat[has_data_mask]
    plot_y, plot_counts = y_flat[has_data_mask], counts_flat[has_data_mask]

    best_idx = np.argmax(plot_y)
    best_bin = (plot_x1[best_idx], plot_x2[best_idx])

    # ---------------------------------------------------------
    # 3. PLOTTING (Plotly vs Matplotlib split)
    # ---------------------------------------------------------
    max_pts = plot_counts.max()

    if use_plotly:
        import plotly.graph_objects as go
        
        # Plotly size scaling (diameter based instead of area)
        sizes_pl = (plot_counts / max_pts * 40 + 8) 
        hover_text = [f"{x1_label}: {xi1:.3f}<br>{x2_label}: {xi2:.3f}<br>{z_label}: {yi:.3f}<br>Count: {int(c)}" 
                      for xi1, xi2, yi, c in zip(plot_x1, plot_x2, plot_y, plot_counts)]

        trace = go.Scatter3d(
            x=plot_x1, y=plot_x2, z=plot_y,
            mode='markers+text',
            text=[str(int(c)) for c in plot_counts],
            textposition='top center',
            textfont=dict(color='black', size=10),
            marker=dict(
                size=sizes_pl, color=plot_y, colorscale='Viridis', 
                showscale=False if fig is not None else True, # Hide colorbar in subplots
                line=dict(color='white', width=1)
            ),
            hoverinfo='text', hovertext=hover_text, name='Binned Data'
        )

        if fig is not None:
            fig.add_trace(trace, row=row, col=col)
        else:
            fig = go.Figure(data=[trace])
            fig.update_layout(scene=dict(xaxis_title=x1_label, yaxis_title=x2_label, zaxis_title=z_label))
            fig.show()

    else:
        # Existing Matplotlib Logic
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect(box_aspect)
            
        if title is not None:
            ax.set_title(title, pad=title_pad, y=1.0, fontsize=8)
        
        sizes_mpl = (plot_counts / max_pts * 300 + 50)
        scatter = ax.scatter(plot_x1, plot_x2, plot_y, s=sizes_mpl, c=plot_y, cmap=cmap, edgecolors='w',
                             alpha=alpha, linewidths=1, label=label)
        # scatter = ax.scatter(plot_x1, plot_x2, plot_y, s=sizes_mpl, c=plot_y, cmap='viridis', edgecolors='w')

        for i in range(len(plot_x1)):
            if show_count:
                ax.text(plot_x1[i], plot_x2[i], plot_y[i], f'{int(plot_counts[i])}', fontsize=9, ha='center', va='bottom', weight='bold')

        # Shrink the tick numbers on the axes so they don't overlap 
        ax.tick_params(axis='x', labelsize=7, pad=-5)
        ax.tick_params(axis='y', labelsize=7, pad=-5)
        ax.tick_params(axis='z', labelsize=7, pad=-3) # Z usually needs slightly less negative padding

        ax.set_xlabel(x1_label, fontsize=8, labelpad=-8)
        ax.set_ylabel(x2_label, fontsize=8, labelpad=-7)
        ax.set_zlabel(z_label, fontsize=8, labelpad=-8)
        
        if ax.figure.axes[0] == ax and show_colorbar: # Only add colorbar if it's not a tightly packed subplot
            plt.colorbar(scatter, ax=ax, label=z_label, pad=0.1)

    if save_path is not None:
        print(f'save fig to {save_path} ... success!')
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300, pad_inches=0.0, transparent=True)
    
    return best_bin if return_best_bin else None

#######################################################################
## Statistical Test Utils
def compare_models(p1, p2, y, metric='cross-entropy', test='paired-t', alternative='less', alpha=0.05):
    """
    Compares two probabilistic models using strictly proper scoring rules.
    
    Parameters:
    p1, p2      : array-like, probabilistic predictions from Model 1 and Model 2 (0 to 1).
    y           : array-like, true binary labels (0 or 1).
    metric      : 'cross-entropy' (Log Loss) or 'square error' (Brier Score).
    test        : 'paired-t' (compares means) or 'wilcoxon' (compares medians/ranks).
    alternative : 'less' (default: is p1 loss < p2 loss?), 'two-sided', or 'greater'.
    alpha       : significance level.
    """
    p1, p2, y = np.array(p1), np.array(p2), np.array(y)
    
    # 1. Calculate Instance-Level Losses (Smaller is better)
    if metric == 'square error':
        loss1 = (p1 - y)**2
        loss2 = (p2 - y)**2
        metric_name = "Brier Score (Squared Error)"
        
    elif metric == 'cross-entropy':
        # Clip probabilities to avoid log(0) mathematically
        eps = 1e-15
        p1_c = np.clip(p1, eps, 1 - eps)
        p2_c = np.clip(p2, eps, 1 - eps)
        loss1 = -(y * np.log(p1_c) + (1 - y) * np.log(1 - p1_c))
        loss2 = -(y * np.log(p2_c) + (1 - y) * np.log(1 - p2_c))
        metric_name = "Log Loss (Cross-Entropy)"
        
    else:
        raise ValueError("metric must be 'square error' or 'cross-entropy'")

    # 2. Perform Statistical Test
    diff = loss1 - loss2
    if test == 'paired-t':
        # ttest_rel compares the means of two related samples
        stat, p_val = stats.ttest_rel(loss1, loss2, alternative=alternative)
        test_name = "Paired t-test"
        
    elif test == 'wilcoxon':
        # wilcoxon compares the median difference / win-rates
        stat, p_val = stats.wilcoxon(loss1, loss2, alternative=alternative)
        test_name = "Wilcoxon Signed-Rank Test"
        
    else:
        ties = np.sum(diff == 0)
        n_trials = len(diff) - ties
        
        # If alternative='less', we hypothesize Loss1 < Loss2. 
        # So we count M1 wins, and test if the success probability is > 0.5
        if alternative == 'less':
            wins = np.sum(diff < 0)
            res = stats.binomtest(wins, n_trials, p=0.5, alternative='greater')
        elif alternative == 'greater':
            wins = np.sum(diff > 0) # Testing if Loss1 > Loss2
            res = stats.binomtest(wins, n_trials, p=0.5, alternative='greater')
        else: # two-sided
            wins = np.sum(diff < 0)
            res = stats.binomtest(wins, n_trials, p=0.5, alternative='two-sided')
            
        stat = wins
        p_val = res.pvalue
        test_name = "Sign Test (Binomial)"
        reason = "it strictly wins significantly more often (ignoring magnitude)"

    # 3. Print the Summary and Conclusion
    mean_l1, mean_l2 = np.mean(loss1), np.mean(loss2)
    win_rate = np.mean(loss1 < loss2) * 100
    
    print(f"--- Model Comparison Report ---")
    print(f"Metric Used : {metric_name}")
    print(f"Test Used   : {test_name} (alternative='{alternative}')")
    print(f"Model 1 Mean Loss : {mean_l1:.4f}")
    print(f"Model 2 Mean Loss : {mean_l2:.4f}")
    print(f"Model 1 Win Rate  : {win_rate:.1f}% of individual predictions\n")
    
    print(f"Test Statistic : {stat:.4f}")
    print(f"p-value        : {p_val:.4e}\n")
    
    # 4. Formulate the explicit statistical conclusion
    if p_val < alpha:
        if alternative == 'less':
            claim = "Model 1 is significantly BETTER than Model 2."
            reason = "its loss is significantly lower" if test == 'paired-t' else "it wins significantly more often"
        elif alternative == 'greater':
            claim = "Model 1 is significantly WORSE than Model 2."
            reason = "its loss is significantly higher" if test == 'paired-t' else "it loses significantly more often"
        else: # two-sided
            winner = "Model 1" if mean_l1 < mean_l2 else "Model 2"
            claim = f"There is a significant difference. {winner} is better."
            reason = "their distributions differ significantly"
            
        print(f"✅ CONCLUSION: {claim}")
        print(f"   At alpha={alpha}, we reject the null hypothesis because {reason}.")
    else:
        print(f"❌ CONCLUSION: No significant difference detected.")
        print(f"   At alpha={alpha}, we fail to reject the null hypothesis. Any observed difference may be due to chance.")

def visualize_paired_losses(p1, p2, y, metric='square error'):
    """
    Visualizes the instance-level losses to reveal magnitude vs frequency paradoxes.
    """
    p1, p2, y = np.array(p1), np.array(p2), np.array(y)
    
    if metric == 'square error':
        loss1 = (p1 - y)**2
        loss2 = (p2 - y)**2
    elif metric == 'cross-entropy':
        eps = 1e-15
        p1_c = np.clip(p1, eps, 1 - eps)
        p2_c = np.clip(p2, eps, 1 - eps)
        loss1 = -(y * np.log(p1_c) + (1 - y) * np.log(1 - p1_c))
        loss2 = -(y * np.log(p2_c) + (1 - y) * np.log(1 - p2_c))
    else:
        raise ValueError("metric must be 'square error' or 'cross-entropy'")

    diff = loss1 - loss2
    win_rate = np.mean(diff < 0) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter of Losses
    # Blue means Model 1 won (Loss 1 < Loss 2)
    colors = np.where(diff < 0, '#1f77b4', '#d62728') 
    ax1.scatter(loss1, loss2, alpha=0.6, c=colors, edgecolor='k')
    
    # Identity line (y=x)
    max_val = max(loss1.max(), loss2.max())
    ax1.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Tie Line')
    
    ax1.set_xlabel('Model 1 Loss')
    ax1.set_ylabel('Model 2 Loss')
    ax1.set_title(f'Instance Losses (M1 Win Rate: {win_rate:.1f}%)')
    ax1.text(0.05, 0.95, 'M1 Wins (Blue)', color='#1f77b4', transform=ax1.transAxes, weight='bold')
    ax1.text(0.95, 0.05, 'M2 Wins (Red)', color='#d62728', transform=ax1.transAxes, ha='right', weight='bold')
    
    # Plot 2: Histogram of Differences
    sns.histplot(diff, bins=40, kde=True, ax=ax2, color='gray')
    ax2.axvline(0, color='black', linestyle='--', lw=2)
    ax2.set_xlabel('Difference (Loss 1 - Loss 2)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Loss Differences')
    ax2.text(0.05, 0.95, 'M1 is Better\n(Negative Diff)', transform=ax2.transAxes, color='#1f77b4', weight='bold')
    ax2.text(0.95, 0.95, 'M2 is Better\n(Positive Diff)', transform=ax2.transAxes, ha='right', color='#d62728', weight='bold')
    
    plt.tight_layout()
    plt.show()