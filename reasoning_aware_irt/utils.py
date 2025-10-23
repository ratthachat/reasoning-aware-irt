
import numpy as np
import matplotlib.pyplot as plt
from typing import *

"""TODO: change 'toy' name to like irt_model_dict something
"""

def plot_b_theta_histograms(ax, toy: dict, title: str, avg_Y: float | None = None):
    """Single-panel histogram plot: θ_j_true, b_i_true, and optional θ_j_max overlay."""
    theta = toy.get('theta_j_true', None)
    b = toy.get('b_i_true', None)
    tmax = toy.get('theta_j_max', None)

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

def plot_multiple_b_theta_histograms(toy_data_list: Sequence[dict], toy_data_names: Sequence[str], plots_per_row: int = 4, avg_Y_list: Sequence[float] | None = None):
    """Quick grid using plot_b_theta_histograms, with optional mean(Y) subtitles.
    """
    m = len(toy_data_list)
    cols = min(plots_per_row, m)
    rows = int(np.ceil(m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.8*rows), squeeze=False)
    for idx, toy in enumerate(toy_data_list):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        avgY = None
        if avg_Y_list is not None and idx < len(avg_Y_list):
            avgY = avg_Y_list[idx]
        plot_b_theta_histograms(ax, toy, toy_data_names[idx], avgY)
    # hide unused axes
    for k in range(m, rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')
    fig.tight_layout()
    return fig, axes