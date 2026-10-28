import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tinydb
from pathlib import Path
from scipy.optimize import curve_fit
from itertools import product

# Import the new database handler
from database_stuff.database_handler import SimulationDB


"""
# --- Define a static, scalable color map for consistency across all plots ---
# This ensures that a given (m_q, m_cg) pair always has the same color.

STATIC_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

# Generate all possible (m_q, m_cg) pairs up to a reasonable limit (e.g., 5)
M_Q_RANGE = range(1, 6)
M_CG_RANGE = range(1, 6)
PARAM_COMBINATIONS = list(product(M_Q_RANGE, M_CG_RANGE))

# Create a persistent color map dictionary
PARAM_COLOR_MAP = {
    params: STATIC_COLORS[i % len(STATIC_COLORS)] 
    for i, params in enumerate(PARAM_COMBINATIONS)
}


def _ensure_adiabatic_column(df):
    """
    # Ensures the DataFrame has a boolean 'adiabatic' column.
    # If the column is missing or has NaNs, it's inferred from the instance name.
    """
    # if 'adiabatic' not in df.columns or df['adiabatic'].isnull().any():
    #     # If the column doesn't exist, create it based on instance name
    #     # We assume runs with "_c_" in the name are "classical" (non-adiabatic)
    #     df['adiabatic'] = ~df['instance_name'].str.contains('_c_', na=False)
    
    # Fill any remaining NaNs with False (a safe default) and ensure boolean type
    df['adiabatic'] = df['adiabatic'].fillna(False).astype(bool)
    return df
"""

def load_data_from_db(db_path='simulation_results_v2.json', query_params=None):
    """
    Loads simulation data from the TinyDB database into a pandas DataFrame.
    """
    db = SimulationDB(db_path)
    
    if query_params:
        q = tinydb.Query()
        conditions = []
        for key, values in query_params.items():
            # Convert numpy arrays to lists for TinyDB querying
            if isinstance(values, np.ndarray):
                values = values.tolist()

            if isinstance(values, list):
                value_conditions = [(q[key] == v) for v in values]
                if value_conditions:
                    field_query = value_conditions[0]
                    for cond in value_conditions[1:]:
                        field_query |= cond
                    conditions.append(field_query)
            else:
                conditions.append(q[key] == values)
        
        if conditions:
            final_query = conditions[0]
            for cond in conditions[1:]:
                final_query &= cond
            results = db.search(final_query)
        else:
            results = db.all()
    else:
        results = db.all()
    
    db.close()
    
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)

def quad_func(x, a, b, c):
    """Quadratic function for fitting."""
    return a * x**2 + b * x + c

def analyze_and_save_optimal_effort(db, data, params, num_lowest_points = 10):
    """
    Analyzes effort vs. n_hops data to find the optimal effort,
    fits a quadratic function, and saves the results to the database.

    Args:
        db (SimulationDB): The database handler for analysis results.
        data (np.ndarray): The raw data array containing steps and efforts.
        params (dict): A dictionary of parameters for the current data group.

    Returns:
        tuple: A tuple containing (optimal_n_hops, optimal_effort, popt).
               Returns (None, None, None) if fitting fails.
    """
    if data.size == 0:
        return None, None, None, None, None



    
    steps = data[:, 0]
    efforts = data[:, 2]


    # remove the first few points from consideration, as they can sometimes be luckily low (ie. random sampling is better than PT)
    efforts[steps < 5] = np.inf
    
    
    unique_steps = np.unique(steps)
    mean_efforts = np.array([np.mean(efforts[steps == h]) for h in unique_steps])
    sem_efforts = np.array([np.std(efforts[steps == h], ddof=1) / np.sqrt(len(efforts[steps == h])) if len(efforts[steps == h]) > 1 else 0 for h in unique_steps])
    if len(unique_steps) < 3:
        return None, None, None, None, None

    # Select points for fitting (lowest 10 effort points)
    sorted_indices = np.argsort(mean_efforts)
    fit_indices = sorted_indices[:num_lowest_points]

    x_fit = unique_steps[fit_indices]
    y_fit = mean_efforts[fit_indices]
    y_err_fit = sem_efforts[fit_indices]

    try:

        popt, pcov = curve_fit(quad_func, x_fit, y_fit, sigma=y_err_fit, absolute_sigma=True)#, bounds = [(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)])
        a, b, c = popt
        #print("a:", a, "b:", b, "c:", c)
        
        if a > 0:  # We are looking for a minimum
            optimal_n_hops = -b / (2 * a)
            optimal_effort = quad_func(optimal_n_hops, *popt)
            
            # Calculate error in nhops and effort using error propagation
            # from the covariance matrix pcov.
            # var(f(x,y)) = (df/dx)^2*var(x) + (df/dy)^2*var(y) + 2*(df/dx*df/dy)*cov(x,y)
            
            # Error for optimal_n_hops = -b / (2a)
            d_a = b / (2 * a**2)  # Partial derivative wrt a
            d_b = -1 / (2 * a)    # Partial derivative wrt b
            var_a, var_b, cov_ab = pcov[0, 0], pcov[1, 1], pcov[0, 1]
            opt_n_hops_error = np.sqrt(d_a**2 * var_a + d_b**2 * var_b + 2 * d_a * d_b * cov_ab)

            # Save results
            analysis_results = {
                'optimal_n_hops': optimal_n_hops,
                'optimal_n_hops_error': opt_n_hops_error,
                'optimal_effort': optimal_effort,
                'error_in_effort': np.sqrt(pcov[0, 0] * optimal_n_hops**4 + pcov[1, 1] * optimal_n_hops**2 + pcov[2, 2] + 2 * optimal_n_hops**3 * pcov[0, 1] + 2 * optimal_n_hops**2 * pcov[0, 2] + 2 * optimal_n_hops * pcov[1, 2]),
                'quad_fit_params': popt.tolist()
            }
            db.insert(params, analysis_results)
            
            return optimal_n_hops, optimal_effort, popt, opt_n_hops_error,fit_indices
        else:
            return None, None, None, None, fit_indices # Not a valley

    except (RuntimeError, ValueError) as e:
        print(f"Quadratic fit failed for {params}: {e}")
        return None, None, None, None, None
"""
def plot_optimal_effort_vs_nspins(df, output_dir):
    """
    # Plots the optimal effort vs. the number of spins, showing adiabatic
    # and standard runs on the same plot with different styles.
    """
    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping plot.")
        return

    df = _ensure_adiabatic_column(df.copy())

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group by the 'adiabatic' flag to apply different styles
    for is_adiabatic, adiabatic_group in df.groupby('adiabatic'):
        # Use different linestyles to distinguish adiabatic runs
        linestyle = '--' if is_adiabatic else '-'
        marker = 'o' if is_adiabatic else 'x'
        adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

        # Plot classical results for this group
        classical_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] == 0].copy()
        if not classical_df.empty:
            classical_df.sort_values('n_spins', inplace=True)
            label = f'Classical (m_q=0){adiabatic_label_suffix}'
            
            ax.plot(
                classical_df['n_spins'],
                classical_df['optimal_effort'],
                marker=marker,
                linestyle=linestyle,
                color='lightgreen',
                label=label,
                alpha=0.7
            )
            if 'error_in_effort' in classical_df.columns and not classical_df['error_in_effort'].isnull().all():
                ax.errorbar(
                    classical_df['n_spins'],
                    classical_df['optimal_effort'],
                    yerr=classical_df['error_in_effort'],
                    fmt='none',
                    ecolor='lightgreen',
                    elinewidth=1,
                    capsize=3
                )

        # Plot quantum results for this group
        quantum_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] > 0].copy()
        if not quantum_df.empty:
            for params, group in quantum_df.groupby(['m_quantum_replicas', 'm_cg']):
                m_q, m_cg = params
                group.sort_values('n_spins', inplace=True)
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                label = f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}'
                
                ax.plot(
                    group['n_spins'],
                    group['optimal_effort'],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                    alpha=0.7
                )
                if 'error_in_effort' in group.columns and not group['error_in_effort'].isnull().all():
                    ax.errorbar(
                        group['n_spins'],
                        group['optimal_effort'],
                        yerr=group['error_in_effort'],
                        fmt='none',
                        ecolor=color,
                        elinewidth=1,
                        capsize=3
                    )

    ax.set_yscale('log')
    ax.set_xlabel('Number of Spins (n)')
    ax.set_ylabel('Optimal Effort (log scale)')
    ax.set_title('Calculated Optimal Effort Scaling (Adiabatic vs. Standard)')
    ax.legend()
    
    output_path = output_dir / 'calculated_optimal_effort_vs_n_spins_combined.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_optimal_effort_timeadjusted_vs_nspins(df, output_dir, classical_time = 1e-9,quantum_time = 1e-5, fev_time = 1e-4):
    """
    # Plots the optimal effort vs. the number of spins for different numbers
    # of quantum replicas and coarse-graining factors.
    """
    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping plot.")
        return

    df = _ensure_adiabatic_column(df.copy())

    fig, ax = plt.subplots(figsize=(10, 8))
    
    for is_adiabatic, adiabatic_group in df.groupby('adiabatic'):
        linestyle = '--' if is_adiabatic else '-'
        marker = 'o' if is_adiabatic else 'x'
        adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

        # Plot classical results
        classical_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] == 0].copy()
        if not classical_df.empty:
            classical_df.sort_values('n_spins', inplace=True)
            ax.plot(
                classical_df['n_spins'],
                (classical_time+fev_time) * classical_df['optimal_effort'],
                marker=marker,
                linestyle=linestyle,
                color='lightgreen',
                label=f'Classical (m_q=0){adiabatic_label_suffix}',
                alpha=0.7
            )

        # Plot quantum results, grouped by m_q and m_cg
        quantum_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] > 0].copy()
        if not quantum_df.empty:
            # Use the global color map for consistency
            for params, group in quantum_df.groupby(['m_quantum_replicas', 'm_cg']):
                m_q, m_cg = params
                group = group.copy()
                group.sort_values('n_spins', inplace=True)
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray') # Default to gray if not in map
                ax.plot(
                    group['n_spins'],
                    (quantum_time+fev_time) * group['optimal_effort'],
                    marker=marker,
                    linestyle=linestyle,
                    color=color,
                    label=f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}',
                    alpha=0.7
                )

    ax.set_yscale('log')
    ax.set_xlabel('Number of Spins (n)')
    ax.set_ylabel('Time (s, log scale)')
    ax.set_title('Expected Time Scaling with System Size')
    ax.legend()
    
    output_path = output_dir / 'expected_time_vs_n_spins.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

def plot_optimal_nhops_vs_nspins(df, output_dir):
    """
    # Plots the optimal n_hops vs. the number of spins for different numbers
    # of quantum replicas and coarse-graining factors.
    """
    if df.empty or 'optimal_n_hops' not in df.columns:
        print("DataFrame is empty or missing 'optimal_n_hops' column, skipping plot.")
        return

    df = _ensure_adiabatic_column(df.copy())

    fig, ax = plt.subplots(figsize=(10, 8))

    for is_adiabatic, adiabatic_group in df.groupby('adiabatic'):
        linestyle = '--' if is_adiabatic else '-'
        marker = 'o' if is_adiabatic else 'x'
        adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

        # Plot classical results
        classical_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] == 0].copy()
        if not classical_df.empty:
            classical_df.sort_values('n_spins', inplace=True)
            ax.errorbar(
                classical_df['n_spins'],
                classical_df['optimal_n_hops'],
                yerr=classical_df.get('optimal_n_hops_error', None),
                marker=marker,
                linestyle=linestyle,
                color='lightgreen',
                label=f'Classical (m_q=0){adiabatic_label_suffix}',
                alpha=0.7, capsize=3
            )

        # Plot quantum results, grouped by m_q and m_cg
        quantum_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] > 0].copy()
        if not quantum_df.empty:
            for params, group in quantum_df.groupby(['m_quantum_replicas', 'm_cg']):
                m_q, m_cg = params
                group = group.copy()
                group.sort_values('n_spins', inplace=True)
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                ax.errorbar(group['n_spins'], group['optimal_n_hops'], yerr=group.get('optimal_n_hops_error', None), marker=marker, linestyle=linestyle, color=color, label=f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}', alpha=0.7, capsize=3)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Spins (n)')
    ax.set_ylabel('Optimal Number of Hops (log scale)')
    ax.set_title('Calculated Optimal Number of Hops Scaling with System Size')
    ax.legend()
    output_path = output_dir / 'calculated_optimal_nhops_vs_n_spins.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)"""

def plot_effort_vs_nhops_v2(df, output_dir, analysis_db):
    """
    Plots effort vs. hops for each n_spins, separating adiabatic and classical runs.
    This function has been refactored for clarity and correctness.
    """
    if df.empty or 'data_save' not in df.columns:
        print("DataFrame is empty or missing 'data_save' column, skipping plot.")
        return

    df = _ensure_adiabatic_column(df.copy())
    
    n_spins_unique = sorted(df['n_spins'].unique())
    if not n_spins_unique:
        return

    fig, axs = plt.subplots(len(n_spins_unique), 1, figsize=(12, 8 * len(n_spins_unique)), squeeze=False)

    for i, n_spins in enumerate(n_spins_unique):
        ax = axs[i, 0]
        spins_df = df[df['n_spins'] == n_spins]

        # Define a local function to process and plot a group of data
        def process_and_plot_group(group_df, label_prefix, color, linestyle, marker):
            if group_df['data_save'].empty:
                return
            
            try:
                # Concatenate all data_save arrays from the group
                full_data = np.concatenate([arr for arr in group_df['data_save'] if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0])
                if full_data.size == 0: return

                full_data = full_data[full_data[:, 0] > 0] # Filter out 0-hop data
                if full_data.size == 0: return

                steps = full_data[:, 0]
                efforts = full_data[:, 2]

                unique_steps = np.unique(steps)
                if len(unique_steps) == 0: return
                
                mean_efforts = np.array([np.mean(efforts[steps == h]) for h in unique_steps])
                sem_efforts = np.array([np.std(efforts[steps == h], ddof=1) / np.sqrt(len(efforts[steps == h])) if len(efforts[steps == h]) > 1 else 0 for h in unique_steps])
                
                ax.scatter(unique_steps, mean_efforts, marker=marker, label=label_prefix, color=color)
                ax.fill_between(unique_steps, mean_efforts - sem_efforts, mean_efforts + sem_efforts, color=color, alpha=0.2)
                
                # --- Analysis Part ---
                # Construct a precise params dict for saving analysis results
                base_params = group_df.iloc[0].to_dict()
                analysis_params = {
                    'n_spins': base_params.get('n_spins'),
                    'm_replicas': base_params.get('m_replicas'),
                    'm_quantum_replicas': base_params.get('m_quantum_replicas'),
                    'm_cg': base_params.get('m_cg'),
                    'adiabatic': base_params.get('adiabatic')
                }

                num_lowest_points = max(12, len(mean_efforts) // 6)
                opt_n_hops, _, popt, _, indices = analyze_and_save_optimal_effort(
                    analysis_db, full_data, analysis_params, num_lowest_points=num_lowest_points
                )
                
                if indices is not None:
                    ax.plot(unique_steps[indices], mean_efforts[indices], 'x', color=color, alpha=0.5, markersize=8)
                
                if popt is not None and opt_n_hops is not None:
                    ax.axvline(opt_n_hops, linestyle='--', color=color, alpha=0.7)

            except (ValueError, IndexError) as e:
                print(f"Could not process data for {label_prefix} at n_spins={n_spins}. Error: {e}")

        # Iterate through adiabatic and classical data for the current n_spins
        for is_adiabatic, data_by_adiabatic in spins_df.groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

            # Handle classical runs
            classical_df = data_by_adiabatic[data_by_adiabatic['m_quantum_replicas'] == 0]
            if not classical_df.empty:
                for m_rep, group in classical_df.groupby('m_replicas'):
                    label = f'Classical (m={m_rep}){adiabatic_label_suffix}'
                    process_and_plot_group(group, label, 'lightgreen', linestyle, marker)

            # Handle quantum runs
            quantum_df = data_by_adiabatic[data_by_adiabatic['m_quantum_replicas'] > 0]
            if not quantum_df.empty:
                for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg']):
                    m_rep, m_q, m_cg = params
                    color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                    label = f'm={m_rep}, m_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}'
                    process_and_plot_group(group, label, color, linestyle, marker)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Effort (log scale)')
        ax.set_title(f'Effort vs. Hops for {n_spins} Spins')
        ax.legend(fontsize='small')
        ax.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    output_path = output_dir / 'effort_vs_nhops_v2.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_success_prob_vs_nhops(df, output_dir):
    """
    Plots the success probability vs. n_hops for each n_spins, separating
    adiabatic and classical runs.
    """
    if df.empty or 'data_save' not in df.columns:
        print("DataFrame is empty or missing 'data_save' column, skipping plot.")
        return

    df = _ensure_adiabatic_column(df.copy())

    n_spins_unique = sorted(df['n_spins'].unique())
    if not n_spins_unique:
        return

    fig, axs = plt.subplots(len(n_spins_unique), 1, figsize=(12, 8 * len(n_spins_unique)), squeeze=False)

    for i, n_spins in enumerate(n_spins_unique):
        ax = axs[i, 0]
        spins_df = df[df['n_spins'] == n_spins]

        # Iterate through adiabatic and classical data for the current n_spins
        for is_adiabatic, data_by_adiabatic in spins_df.groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

            def plot_prob_for_group(group_df, label, color, marker):
                if group_df['data_save'].empty: return
                try:
                    full_data = np.concatenate([arr for arr in group_df['data_save'] if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 0])
                    if full_data.size == 0: return
                    
                    hops = full_data[:, 0]
                    probs = full_data[:, 3] # Probability is the 4th column

                    unique_hops = np.unique(hops)
                    mean_probs = [np.mean(probs[hops == h]) for h in unique_hops]
                    sem_probs = [np.std(probs[hops == h], ddof=1) / np.sqrt(len(probs[hops == h])) if len(probs[hops == h]) > 1 else 0 for h in unique_hops]
                    
                    ax.errorbar(unique_hops, mean_probs, yerr=sem_probs, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.8, capsize=3)
                except (ValueError, IndexError) as e:
                    print(f"Could not process success probability for {label} at n_spins={n_spins}. Error: {e}")

            # Handle classical runs
            classical_df = data_by_adiabatic[data_by_adiabatic['m_quantum_replicas'] == 0]
            if not classical_df.empty:
                for m_rep, group in classical_df.groupby('m_replicas'):
                    plot_prob_for_group(group, f'Classical (m={m_rep}){adiabatic_label_suffix}', 'lightgreen', marker)

            # Handle quantum runs
            quantum_df = data_by_adiabatic[data_by_adiabatic['m_quantum_replicas'] > 0]
            if not quantum_df.empty:
                for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg']):
                    m_rep, m_q, m_cg = params
                    color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                    label = f'm={m_rep}, m_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}'
                    plot_prob_for_group(group, label, color, marker)


        ax.set_yscale('log')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Success Probability (log scale)')
        ax.set_title(f'Success Probability vs. Hops for {n_spins} Spins')
        ax.legend(fontsize='small')
        ax.grid(True, which="both", ls="-", alpha=0.2)


    plt.tight_layout()
    output_path = output_dir / 'success_prob_vs_nhops.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping.")
        return

    df = _ensure_adiabatic_column(df.copy())

    # Get all unique m_q values across both adiabatic and non-adiabatic runs
    unique_mq = sorted(df['m_quantum_replicas'].unique())
    # Exclude m_q=0 from this list, as it's handled separately as 'classical'
    unique_mq = [mq for mq in unique_mq if mq > 0] 
    
    if not unique_mq:
        print("No quantum data to plot for separate m_q plots.")
        return

    # Setup for two-column plots
    num_plots = len(unique_mq)
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols

    # --- Main plot figure (fits) ---
    fig_main, axs_main = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows), sharex=True, squeeze=False)
    
    # --- Residuals plot figure ---
    fig_res, axs_res = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True, squeeze=False)

    def log_model(n, c, alpha, b):
        alpha = 1/3
        return c * (n ** alpha) + b

    def plot_fit_and_residuals(ax_main, ax_res, group_df, label, color, linestyle, marker):
        if len(group_df) < 3: 
            print(f"Skipping '{label}': needs at least 3 points for this fit.")
            return

        group_df = group_df.sort_values('n_spins').copy()
        n_spins = group_df['n_spins'].values
        optimal_effort = group_df['optimal_effort'].values
        
        if np.any(optimal_effort <= 0):
            print(f"Skipping '{label}' due to non-positive effort values.")
            return

        log_effort = np.log(optimal_effort)
        
        c_guess, b_guess = np.polyfit(np.sqrt(n_spins), log_effort, 1)
        p0 = [max(c_guess, 0.01), 1/3, b_guess]
        
        try:
            popt, _ = curve_fit(
                log_model, n_spins, log_effort, p0=p0, 
                bounds=([0, 0.1, -np.inf], [np.inf, 2.0, np.inf]), maxfev=20000
            )
            c_fit, alpha_fit, b_fit = popt
        except RuntimeError:
            print(f"Fit failed to converge for '{label}'.")
            return

        label_with_fit = f"{label} (c: {c_fit:.3f})"
        n_smooth = np.linspace(n_spins.min(), n_spins.max(), 100)
        log_effort_pred_smooth = log_model(n_smooth, c_fit, alpha_fit, b_fit)
        log_effort_pred = log_model(n_spins, c_fit, alpha_fit, b_fit)
        residuals = log_effort - log_effort_pred

        # Plot on the main axes
        ax_main.plot(n_spins, optimal_effort, marker=marker, linestyle='', color=color, alpha=0.8, label=label_with_fit)
        ax_main.plot(n_smooth, np.exp(log_effort_pred_smooth), color=color, linestyle=linestyle, alpha=0.6)
        
        # Plot on the residuals axes
        ax_res.plot(n_spins, residuals, marker=marker, linestyle=linestyle, color=color, alpha=0.7)

    for i, m_q in enumerate(unique_mq):
        row, col = i // ncols, i % ncols
        ax_main = axs_main[row, col]
        ax_res = axs_res[row, col]

        # Iterate through adiabatic and non-adiabatic data for the current m_q
        for is_adiabatic, adiabatic_group in df[df['m_quantum_replicas'] == m_q].groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

            for m_cg, group in adiabatic_group.groupby('m_cg'):
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                plot_fit_and_residuals(ax_main, ax_res, group, f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}', color, linestyle, marker)

        # Plot classical (m_q=0) data on each subplot for reference
        for is_adiabatic, classical_group in df[df['m_quantum_replicas'] == 0].groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''
            plot_fit_and_residuals(ax_main, ax_res, classical_group, f'Classical (m_q=0){adiabatic_label_suffix}', 'lightgreen', linestyle, marker)

        ax_main.set_yscale("log")
        ax_main.set_xscale('function', functions=(lambda x: x**(1/3), lambda x: x**3))
        ax_main.set_ylabel('Optimal Effort')
        ax_main.set_xlabel("Number of Spins (n)")
        ax_main.set_title(f'Stretched Exponential Fit for m_q = {m_q}')
        ax_main.legend(loc='best', fontsize='small')
        
        ax_res.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax_res.set_ylabel('Residuals')
        ax_res.set_title(f'Residuals for Fit (m_q = {m_q})')

    # Clean up layout and save figures
    for i in range(num_plots, nrows * ncols):
        row, col = i // ncols, i % ncols
        fig_main.delaxes(axs_main[row, col])
        fig_res.delaxes(axs_res[row, col])

    fig_main.tight_layout()
    output_path_main = output_dir / 'effort_scaling_by_mq_fit.png'
    fig_main.savefig(output_path_main)
    print(f"Saved plot to {output_path_main}")
    plt.close(fig_main)

    fig_res.tight_layout()
    output_path_res = output_dir / 'effort_scaling_by_mq_residuals.png'
    fig_res.savefig(output_path_res)
    print(f"Saved plot to {output_path_res}")
    plt.close(fig_res)
"""
def plot_sqrt_scaling_fit_by_m_cg_and_residuals(df, output_dir):
    #
    # Fits the dataset to a stretched exponential E = A * exp(c * n^alpha).
    # Creates a separate subplot for each m_cg value, showing variations in m_q.
    # The classical case (m_q=0) is plotted on all subplots for reference.
    # Generates two plots: one for the fits and another for the residuals.
    # NOTE: This function has not been updated to group by the 'adiabatic' parameter.
    #
    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping.")
        return

    df = _ensure_adiabatic_column(df.copy())

    # Get all unique m_cg values across both adiabatic and non-adiabatic runs for quantum data
    unique_mcg = sorted(df[df['m_quantum_replicas'] > 0]['m_cg'].unique())
    
    if not unique_mcg:
        print("No quantum data to plot for separate m_cg plots.")
        return

    # Setup for two-column plots
    num_plots = len(unique_mcg)
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols

    # --- Main plot figure (fits) ---
    fig_main, axs_main = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows), sharex=True, squeeze=False)
    
    # --- Residuals plot figure ---
    fig_res, axs_res = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True, squeeze=False)

    def log_model(n, c, alpha, b):
        alpha = 1/3
        return c * (n ** alpha) + b

    def plot_fit_and_residuals(ax_main, ax_res, group_df, label, color, linestyle, marker):
        if len(group_df) < 3: 
            print(f"Skipping '{label}': needs at least 3 points for this fit.")
            return

        group_df = group_df.sort_values('n_spins').copy()
        n_spins = group_df['n_spins'].values
        optimal_effort = group_df['optimal_effort'].values
        
        if np.any(optimal_effort <= 0):
            print(f"Skipping '{label}' due to non-positive effort values.")
            return

        log_effort = np.log(optimal_effort)
        
        c_guess, b_guess = np.polyfit(np.sqrt(n_spins), log_effort, 1)
        p0 = [max(c_guess, 0.01), 1/3, b_guess]
        
        try:
            popt, _ = curve_fit(
                log_model, n_spins, log_effort, p0=p0, 
                bounds=([0, 0.1, -np.inf], [np.inf, 2.0, np.inf]), maxfev=20000
            )
            c_fit, alpha_fit, b_fit = popt
        except RuntimeError:
            print(f"Fit failed to converge for '{label}'.")
            return

        label_with_fit = f"{label} (c: {c_fit:.3f})"
        n_smooth = np.linspace(n_spins.min(), n_spins.max(), 100)
        log_effort_pred_smooth = log_model(n_smooth, c_fit, alpha_fit, b_fit)
        log_effort_pred = log_model(n_spins, c_fit, alpha_fit, b_fit)
        residuals = log_effort - log_effort_pred

        # Plot on the main axes
        ax_main.plot(n_spins, optimal_effort, marker=marker, linestyle='', color=color, alpha=0.8, label=label_with_fit)
        ax_main.plot(n_smooth, np.exp(log_effort_pred_smooth), color=color, linestyle=linestyle, alpha=0.6)
        
        # Plot on the residuals axes
        ax_res.plot(n_spins, residuals, marker=marker, linestyle=linestyle, color=color, alpha=0.7)

    for i, m_cg in enumerate(unique_mcg):
        row, col = i // ncols, i % ncols
        ax_main = axs_main[row, col]
        ax_res = axs_res[row, col]

        # Iterate through adiabatic and non-adiabatic data for the current m_cg
        for is_adiabatic, adiabatic_group in df[df['m_cg'] == m_cg].groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

            # Plot quantum data for this m_cg
            quantum_mcg_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] > 0]
            for m_q, group in quantum_mcg_df.groupby('m_quantum_replicas'):
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                plot_fit_and_residuals(ax_main, ax_res, group, f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}', color, linestyle, marker)

        # Plot classical (m_q=0) data on each subplot for reference
        for is_adiabatic, classical_group in df[df['m_quantum_replicas'] == 0].groupby('adiabatic'):
            linestyle = '--' if is_adiabatic else '-'
            marker = 'o' if is_adiabatic else 'x'
            adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''
            plot_fit_and_residuals(ax_main, ax_res, classical_group, f'Classical (m_q=0){adiabatic_label_suffix}', 'lightgreen', linestyle, marker)


        ax_main.set_yscale("log")
        ax_main.set_xscale('function', functions=(lambda x: x**(1/3), lambda x: x**3))
        ax_main.set_ylabel('Optimal Effort')
        ax_main.set_xlabel("Number of Spins (n)")
        ax_main.set_title(f'Stretched Exponential Fit for m_cg = {m_cg}')
        ax_main.legend(loc='best', fontsize='small')
        
        ax_res.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax_res.set_ylabel('Residuals')
        ax_res.set_title(f'Residuals for Fit (m_cg = {m_cg})')

    # Clean up layout and save figures
    for i in range(num_plots, nrows * ncols):
        row, col = i // ncols, i % ncols
        fig_main.delaxes(axs_main[row, col])
        fig_res.delaxes(axs_res[row, col])

    fig_main.tight_layout()
    output_path_main = output_dir / 'effort_scaling_by_mcg_fit.png'
    fig_main.savefig(output_path_main)
    print(f"Saved plot to {output_path_main}")

    plt.close(fig_main)

    fig_res.tight_layout()
    output_path_res = output_dir / 'effort_scaling_by_mcg_residuals.png'
    fig_res.savefig(output_path_res)
    print(f"Saved plot to {output_path_res}")
    plt.close(fig_res)


def plot_sqrt_scaling_fit_and_residuals(df, output_dir):
    # Fits the dataset to a stretched exponential E = A * exp(c * n^alpha).
    # Linearizes to log(E) = c * n^alpha + b.
    # This captures the topological entropic barrier (alpha) without 
    # a polynomial prefactor.
    # NOTE: This function has not been updated to group by the 'adiabatic' parameter.
    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping.")
        return

    df = _ensure_adiabatic_column(df.copy())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2.5, 1]})

    def log_model(n, c, alpha, b):
        alpha = 1/3
        return c * (n ** alpha) + b

    def plot_fit_and_residuals(group_df, label, color, linestyle, marker):
        if len(group_df) < 3: 
            print(f"Skipping '{label}': needs at least 3 points for this fit.")
            return

        group_df = group_df.sort_values('n_spins').copy()
        n_spins = group_df['n_spins'].values
        optimal_effort = group_df['optimal_effort'].values
        err_optimal_effort = group_df.get('error_in_effort', None)

        if np.any(optimal_effort <= 0):
            print(f"Skipping '{label}' due to non-positive effort values.")
            return

        log_effort = np.log(optimal_effort)
        
        # Propagate linear error to log-space error
        sigma = None
        absolute_sigma = False
        if err_optimal_effort is not None and not err_optimal_effort.isna().all():
            linear_err = err_optimal_effort.values
            valid_err_mask = linear_err > 0
            if np.any(valid_err_mask):
                sigma = np.full_like(optimal_effort, np.inf, dtype=float)
                sigma[valid_err_mask] = linear_err[valid_err_mask] / optimal_effort[valid_err_mask]
                absolute_sigma = True
        
        c_guess, b_guess = np.polyfit(np.sqrt(n_spins), log_effort, 1)
        p0 = [max(c_guess, 0.01), 1/3, b_guess]
        
        try:
            popt, _ = curve_fit(
                log_model, 
                n_spins, 
                log_effort, 
                p0=p0, 
                bounds=([0, 0.1, -np.inf], [np.inf, 2.0, np.inf]),
                sigma=sigma if absolute_sigma else None,
                absolute_sigma=absolute_sigma,
                maxfev=20000
            )
            c_fit, alpha_fit, b_fit = popt
        except RuntimeError:
            print(f"Fit failed to converge for '{label}'.")
            return

        label_with_fit = f"{label} (c: {c_fit:.3f})"

        n_smooth = np.linspace(4,40,100)#n_spins.min(), n_spins.max(), 100)
        log_effort_pred_smooth = log_model(n_smooth, c_fit, alpha_fit, b_fit)
        
        log_effort_pred = log_model(n_spins, c_fit, alpha_fit, b_fit)
        residuals = log_effort - log_effort_pred

        # --- Top panel: Data and Fit ---
        if absolute_sigma:
            ax1.errorbar(n_spins, optimal_effort, yerr=sigma, fmt=marker, color=color, 
                         alpha=0.8, label=label_with_fit)
        else:
            ax1.plot(n_spins, optimal_effort, marker=marker, linestyle='', color=color, 
                     alpha=0.8, label=label_with_fit)
                     
        ax1.plot(n_smooth, np.exp(log_effort_pred_smooth), color=color, linestyle=linestyle, alpha=0.6)

        # --- Bottom panel: Residuals ---
        if absolute_sigma:
            ax2.errorbar(n_spins, residuals, yerr=sigma, fmt=marker, linestyle=linestyle, 
                         color=color, alpha=0.7)
        else:
            ax2.plot(n_spins, residuals, marker=marker, linestyle=linestyle, color=color, alpha=0.7)
        print("method:", label, "c_fit:", c_fit, "alpha_fit:", alpha_fit, "b_fit:", b_fit)
        print("For N = 100, this would need ", np.exp(log_model(100, c_fit, alpha_fit, b_fit)), "effort")
    
    for is_adiabatic, adiabatic_group in df.groupby('adiabatic'):
        linestyle = '--' if is_adiabatic else '-'
        marker = 'o' if is_adiabatic else 'x'
        adiabatic_label_suffix = ', Adiabatic' if is_adiabatic else ''

        classical_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] == 0]
        if not classical_df.empty:
            plot_fit_and_residuals(classical_df, f'Classical (m_q=0){adiabatic_label_suffix}', 'lightgreen', linestyle, marker)

        quantum_df = adiabatic_group[adiabatic_group['m_quantum_replicas'] > 0]
        if not quantum_df.empty:
            for params, group in quantum_df.groupby(['m_quantum_replicas', 'm_cg']):
                m_q, m_cg = params
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray') 
                plot_fit_and_residuals(group, f'm_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}', color, linestyle, marker)
            
    ax1.set_yscale("log")
    ax1.set_xscale('function', functions=(lambda x: x**(1/3),lambda x: x**3))
    ax1.set_ylabel('Optimal Effort')
    ax1.set_xlabel("Number of Spins (n)")

    ax1.set_title(r'Data and Stretched Exponential Fit ($E = A \cdot e^{c n^\alpha}$)')
    ax1.legend(loc='best', fontsize='small')
    
    ax2.set_xlabel('Number of Spins (n)')
    ax2.set_ylabel('Residuals')
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / 'effort_scaling_stretched_exp_fit.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)"""

if __name__ == "__main__":
    
    # Define the output directory for plots
    output_dir = Path(__file__).resolve().parent / "plots_v5"
    output_dir.mkdir(exist_ok=True)

    # An empty dictionary will load all data from the database.
    query_params = {}

    print(f"Loading data with parameters: {query_params}")
    
    # Load simulation data
    this_dir = Path(__file__).resolve().parent
    results_dir = this_dir / "results"
    # Note: Ensure your simulation results JSON file is correctly named and located.
    main_df = load_data_from_db(db_path=results_dir / 'simulation_results_v2.json', query_params=query_params)
    
    # This DB will store the analysis results (optimal effort, etc.)
    # It will be cleared and recreated each time the script runs.
    analysis_db_path = this_dir / 'analysis_results.json'
    if analysis_db_path.exists():
        analysis_db_path.unlink() # Clear previous analysis
    analysis_db = SimulationDB(analysis_db_path)

    if main_df.empty:
        print("No simulation data found. Exiting.")
    else:
        print(f"Successfully loaded {len(main_df)} records from simulation database.")
        
        # Generate analysis data and initial plots
        # This updated function now passes the 'adiabatic' flag to the analysis DB
        plot_effort_vs_nhops_v2(main_df, output_dir, analysis_db)
        
        # This updated function will show adiabatic runs with a different linestyle
        plot_success_prob_vs_nhops(main_df, output_dir)
        
        analysis_db.close() # Close DB to flush all analysis results to disk

        """# Load the analysis data we just created, which now includes the 'adiabatic' flag
        analysis_df  = load_data_from_db(db_path=analysis_db_path, query_params=query_params)
        
        if analysis_df.empty:
            print("No analysis data was generated. Skipping final plots.")
        else:
            print(f"Loaded {len(analysis_df)} analysis records.")
            
            # These plotting functions have been updated to handle the 'adiabatic' flag
            #plot_optimal_effort_vs_nspins(analysis_df, output_dir)
            #plot_optimal_effort_timeadjusted_vs_nspins(analysis_df, output_dir, fev_time=1e-7)
            #plot_optimal_nhops_vs_nspins(analysis_df, output_dir)

            # the 'adiabatic' flag. They will plot all data together.
            # Applying the pattern from the functions above is required to separate them.
            plot_sqrt_scaling_fit_and_residuals(analysis_df, output_dir)
            #plot_sqrt_scaling_fit_by_mq_and_residuals(analysis_df, output_dir)
            plot_sqrt_scaling_fit_by_m_cg_and_residuals(analysis_df, output_dir)
        
        print("Plotting pipeline finished.")"""

"""
# A new, cleaner plot_experiments.py

# Make sure you have the required libraries installed:
# pip install pandas seaborn matplotlib
from analysis import PlottingAnalyser, flexible_plot

def main():

    # 1. Initialize the analyser. This loads the data once and is very fast
    #    on subsequent calls.
    analyser = PlottingAnalyser(results_path="C:/Users/Stuart Ferguson/OneDrive - University of Edinburgh/Documents/PhD/CODE/QeHO/QePT/results/simulation_results_v2.json")#'analysis_results.json')

    # 2. Generate a plot filtered for a specific value, as you requested.
    #    Here we plot only the data where 'm_cg' is 1.
    print("Generating plot for m_cg = 1...")
    flexible_plot(
        analyser=analyser,
        x_col='n_spins',
        y_col='optimal_effort',
        filters={'m_cg': 1},
        plot_title='Optimal Effort vs. Number of Spins (m_cg = 1)',
        save_path='plots/new/effort_vs_nspins_mcg1.png',
        show_plot=False
    )

    # 3. Generate a more complex plot, grouping by a parameter.
    #    Here we plot for specific numbers of spins and create separate
    #    lines for each value of 'm_q'.
    print("\nGenerating plot grouped by 'm_q'...")
    flexible_plot(
        analyser=analyser,
        x_col='n_spins',
        y_col='success_prob',
        filters={'n_spins': [10, 15, 20, 25, 30]},
        group_by='m_q',
        plot_title='Success Probability vs. N Spins (Grouped by m_q)',
        save_path='plots/new/prob_vs_nspins_by_mq.png',
        show_plot=False
    )

    # 4. Generate a plot with a logarithmic scale to see trends.
    print("\nGenerating plot with log scale...")
    flexible_plot(
        analyser=analyser,
        x_col='n_hops',
        y_col='optimal_effort',
        filters={'m_cg': 2, 'm_q': 1},
        log_y=True,
        plot_title='Optimal Effort vs. N Hops (Log Scale, m_cg=2, m_q=1)',
        save_path='plots/new/effort_vs_nhops_log_filtered.png',
        show_plot=False
    )

    print("\nDone generating plots.")


if __name__ == '__main__':
    main()
"""