import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import math

from operator import itemgetter

from .utils import prob_correct_3pl

def estimate_latent(mask, a_i, b_i, c_i, y_i):
    sub_a = a_i[mask]
    sub_b = b_i[mask]
    sub_c = c_i[mask]
    sub_y = y_i[mask][:, None] # (sub_Q,1)

    latent_k = np.linspace(-3, 4, 141)
    p_ik = prob_correct_3pl(latent_k, sub_a, sub_b, sub_c)
    ll_k = (sub_y * np.log(p_ik) + (1.0 - sub_y) * np.log(1.0 - p_ik)).sum(axis=0)  # (K,)
    return latent_k[np.argmax(ll_k)]

def estimate_gamma_kappa(theta_base, theta_best, optimal_log_budget):
    gamma, kappa = 0,0
    kappa = (theta_base - theta_best)/optimal_log_budget**2
    gamma = -2*kappa*optimal_log_budget
    return gamma, kappa

def plot_sliced_bins(X, y, slice_axis=0, num_slices=6, num_bins_per_slice=10, fit_object=None,
                     min_points_per_bin=1, use_quantiles=False, share_y=True, title="GrandTitle",
                     x1_label="Feature 1 (X1)", x2_label="Feature 2 (X2)", y_label="Target (Y)",
                     show_fig=False):
    """
    Slices a 2D dataset along 'slice_axis' into 'num_slices' bins. 
    For each slice, generates a 1D binned scatter plot of the OTHER axis vs y.
    
    Args:
        slice_axis (int): 0 to slice by X1 (plot X2 vs Y), 1 to slice by X2 (plot X1 vs Y).
        num_slices (int): Number of subplots (number of bins to divide the slicing axis into).
        num_bins_per_slice (int): Number of 1D bins calculated INSIDE each subplot.
        share_y (bool): If True, all subplots share the same Y-axis scale for easier comparison.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2: 
        raise ValueError("X must be a 2D array [N, 2]")

    #############################
    # Add-on : to estimated eff-latent for each B_ij bin
    latent_list = [] # will be non-empty if fit_object is not None
    best_B_j, best_latent, gamma, kappa = None, None, None, None
    if fit_object is not None: 
        assert (slice_axis==1 and x2_label=="B_ij") or (slice_axis==0 and x1_label=="B_ij")
        use_quantiles = True # only quantile mode, only B_ij as slice (1st binned)
    #############################
    
    # 1. Route the variables based on user choice
    if slice_axis == 0:
        x_slice, x_plot = X[:, 0], X[:, 1]
        slice_name, plot_name = x1_label, x2_label
    elif slice_axis == 1:
        x_slice, x_plot = X[:, 1], X[:, 0]
        slice_name, plot_name = x2_label, x1_label
    else:
        raise ValueError("slice_axis must be 0 (for x1) or 1 (for x2).")

    # Filter sparse data (inherited from your original logic)
    if not use_quantiles: 
        p02_s, p98_s = np.percentile(x_slice, [2, 98])
        p02_p, p98_p = np.percentile(x_plot, [2, 98])
        valid_mask = (x_slice >= p02_s) & (x_slice <= p98_s) & (x_plot >= p02_p) & (x_plot <= p98_p)
        x_slice, x_plot, y = x_slice[valid_mask], x_plot[valid_mask], y[valid_mask]

    if len(x_slice) == 0:
        print("No data to plot after filtering.")
        return

    # 2. Determine the edges for the SLICES (Subplots)
    if use_quantiles:
        slice_edges = np.unique(np.quantile(x_slice, np.linspace(0, 1, num_slices + 1)))
        num_slices = len(slice_edges) - 1 # np.unique might reduce length if heavy ties exist
    else:
        slice_edges = np.linspace(np.min(x_slice), np.max(x_slice), num_slices + 1)

    # 3. Setup Layout Grid
    cols = min(3, num_slices) # Max 3 columns per row
    rows = math.ceil(num_slices / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=share_y)
    
    # Force axs to be a 1D flat array for easy looping
    if rows * cols == 1: axs = np.array([axs])
    axs = axs.flatten()
    
    # 4. Generate Subplots
    for i in range(num_slices):
        # Grab the data that falls into this specific slice
        if i == num_slices - 1:
            mask = (x_slice >= slice_edges[i]) & (x_slice <= slice_edges[i+1])
        else:
            mask = (x_slice >= slice_edges[i]) & (x_slice < slice_edges[i+1])

        latent_title = ""
        grand_latent_title = ""
        if fit_object is not None:
            latent = estimate_latent(mask, fit_object.a_i, x_plot, fit_object.info["c_i"], y )
            latent_list.append(latent)
            latent_title = f" - latent={latent:.2f}"
            best_latent, best_idx = np.max(latent_list), np.argmax(latent_list)
            best_B_j = int((slice_edges[best_idx] + slice_edges[best_idx+1])/2)
            gamma, kappa = estimate_gamma_kappa(latent_list[0], best_latent, np.log1p(best_B_j))
            grand_latent_title = f" -- Budget:{best_B_j} - best_latent:{best_latent:.2f} - g:{gamma:.2e} - k:{kappa:.2e}"
        
        sub_x = x_plot[mask]
        sub_y = y[mask]
        
        # Formatting the subplot
        ax = axs[i]
        ax_title = rf"{slice_name} $\in$ [{slice_edges[i]:.2f}, {slice_edges[i+1]:.2f}]" + latent_title
        
        ax.set_title(ax_title, 
                     # fontsize=11 # will be ugly in paper
                    )
        ax.set_xlabel(plot_name)
        if i % cols == 0 or not share_y: # Only label Y on the left-most columns if shared
            ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.6)

        if len(sub_x) == 0:
            ax.text(0.5, 0.5, "No data in slice", ha='center', va='center', transform=ax.transAxes)
            continue

        # --- 1D Binning Logic INSIDE the slice (Matches your original function) ---
        if use_quantiles:
            plot_edges = np.unique(np.quantile(sub_x, np.linspace(0, 1, num_bins_per_slice + 1)))
        else:
            plot_edges = np.linspace(np.min(sub_x), np.max(sub_x), num_bins_per_slice + 1)
            
        plot_centers = (plot_edges[:-1] + plot_edges[1:]) / 2
        
        y_avgs = binned_statistic(sub_x, sub_y, statistic='mean', bins=plot_edges).statistic
        counts = binned_statistic(sub_x, sub_y, statistic='count', bins=plot_edges).statistic
        
        valid_bins = ~np.isnan(y_avgs) & (counts > min_points_per_bin)
        
        if not np.any(valid_bins):
            ax.text(0.5, 0.5, "Insufficient points per bin", ha='center', va='center', transform=ax.transAxes)
            continue
            
        # Draw Scatters & Text
        max_counts = counts[valid_bins].max()
        sizes = (counts[valid_bins] / max_counts * 400 + 20)
        
        ax.scatter(plot_centers[valid_bins], y_avgs[valid_bins], 
                   s=sizes, alpha=0.7, edgecolors='w', label="Binned Data")
        
        for j in np.where(valid_bins)[0]:
            ax.text(plot_centers[j], y_avgs[j], f'{int(counts[j])}', 
                    fontsize=10, ha='center', va='bottom', weight='bold')

    # 5. Hide unused subplots (if num_slices doesn't perfectly fill the R x C grid)
    for j in range(num_slices, rows * cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.subplots_adjust(top=0.91)
    plt.suptitle(title+grand_latent_title)
    
    if show_fig:
        plt.show()
    else:
        plt.close() # save memory

    ret_dict = {"best_B_j" : best_B_j, 
        "best_latent" : best_latent, 
        "gamma" : gamma, 
        "kappa": kappa
        }
    return ret_dict

######################################################################
# model's print_summary helper
def print_model_summary(model_list, fit_results, Y_ij, sort_by="m", ascending=False):
    data = []
    for j, m in enumerate(model_list):
        d={} # temp dict
        d['b_theta'] = fit_results.theta_j_mean[j]
        d['theta_se'] = fit_results.theta_j_se[j]
        d['gamma'] = fit_results.gamma_j[j]
        d['gamma_se'] = fit_results.gamma_bracket_half_tau[j]

        d['kappa'] = np.clip(fit_results.kappa_j[j], a_min=-1.00, a_max=-0.00001)
        d['kappa_se'] = fit_results.kappa_bracket_half_tau[j] # fit_results.kappa_bracket_half_ideal[j]

        gse, kse = d['gamma_se'], d['kappa_se'] # fit_results.gamma_bracket_half_ideal[j] , fit_results.kappa_bracket_half_tau[j]# 
        N_SE = 0.0
        MIN_BUDGET, MAX_BUDGET = 32, 16384
        d['optimal_log_budget'] = -(d['gamma']+N_SE*gse)/(2*(d['kappa']+N_SE*kse))   # conservative with N_SE
        d['optimal_budget'] = np.clip(np.exp(d['optimal_log_budget']) - 1, MIN_BUDGET, MAX_BUDGET)

        d["optimal_log_budget"] = np.log1p(d["optimal_budget"]) # also clip here
        
        d['avg_budget'] = fit_results.info['B_ij'][:,j].mean()
        x_j = np.log1p(d['avg_budget'])
        d['eff_theta'] = d['b_theta'] + x_j* d['gamma'] + x_j**2 * d['kappa']
        d['best_theta'] = d['b_theta'] + d['optimal_log_budget'] * d['gamma'] + d['optimal_log_budget']**2 * d['kappa']
        
        d['eff_se'] = d['theta_se'] + d['optimal_log_budget']/2 * gse + (d['optimal_log_budget']/2)**2 * kse
        d['best_se'] = d['theta_se'] + d['optimal_log_budget'] * gse + d['optimal_log_budget']**2 * kse
        
        d['avg_y'] = Y_ij[:, j].mean()
        d['m'] = m
        data.append(d)

    data.sort(key=itemgetter(sort_by), reverse=not ascending)

    for d in data:
        # se is quite small, so we ignore them
        print(
            f"{d['m']:<{35}} | "
            f"BaseTheta: {d['b_theta']:6.2f} ± {d['theta_se']:.3f}| "
            f"Gamma: {d['gamma']:7.4f} ± {d['gamma_se']:.4f}| "
            f"Kappa: {d['kappa']:7.4f} ± {d['kappa_se']:.4f} |"
            f"Avg Y : {d['avg_y']:7.3f}"
        )
        print(
            f"Best Budget: {d['optimal_budget']:5.0f}|"
            f"BestTheta: {d['best_theta']:7.3f} ± {d['best_se']:7.3f}|"
            f"Avg. EffTheta: {d['eff_theta']:7.3f} ± {d['eff_se']:7.3f}|"
        )
        print()
    return data