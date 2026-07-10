import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tinydb
from pathlib import Path
from scipy.optimize import curve_fit
from itertools import product

# Import the new database handler
from database_stuff.database_handler import SimulationDB



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
        return None, None, None



    
    steps = data[:, 0]
    efforts = data[:, 2]
    
    unique_steps = np.unique(steps)
    mean_efforts = np.array([np.mean(efforts[steps == h]) for h in unique_steps])
    sem_efforts = np.array([np.std(efforts[steps == h], ddof=1) / np.sqrt(len(efforts[steps == h])) if len(efforts[steps == h]) > 1 else 0 for h in unique_steps])

    if len(unique_steps) < 3:
        return None, None, None

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
                'quad_fit_params': popt.tolist()
            }
            db.insert(params, analysis_results)
            
            return optimal_n_hops, optimal_effort, popt, opt_n_hops_error
        else:
            return None, None, None, None # Not a valley

    except (RuntimeError, ValueError) as e:
        print(f"Quadratic fit failed for {params}: {e}")
        return None, None, None, None


def plot_optimal_effort_vs_nspins(df, output_dir):
    """
    Plots the optimal effort vs. the number of spins for different numbers
    of quantum replicas and coarse-graining factors.
    """
    if df.empty or 'optimal_effort' not in df.columns:
        print("DataFrame is empty or missing 'optimal_effort' column, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot classical results
    classical_df = df[df['m_quantum_replicas'] == 0].copy()
    if not classical_df.empty:
        classical_df.sort_values('n_spins', inplace=True)
        ax.plot(
            classical_df['n_spins'],
            classical_df['optimal_effort'],
            marker='o',
            linestyle='-',
            color='lightgreen',
            label='Classical (m_q=0)',
            alpha=0.7
        )

    # Plot quantum results, grouped by m_q and m_cg
    quantum_df = df[df['m_quantum_replicas'] > 0].copy()
    if not quantum_df.empty:
        # Use the global color map for consistency
        for params, group in quantum_df.groupby(['m_quantum_replicas', 'm_cg']):
            m_q, m_cg = params
            group = group.copy()
            group.sort_values('n_spins', inplace=True)
            color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray') # Default to gray if not in map
            ax.plot(
                group['n_spins'],
                group['optimal_effort'],
                marker='o',
                linestyle='--',
                color=color,
                label=f'm_q={m_q}, m_cg={m_cg}',
                alpha=0.7
            )

    ax.set_yscale('log')
    ax.set_xlabel('Number of Spins (n)')
    ax.set_ylabel('Optimal Effort (log scale)')
    ax.set_title('Calculated Optimal Effort Scaling with System Size')
    ax.legend()
    
    output_path = output_dir / 'calculated_optimal_effort_vs_n_spins.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

def plot_effort_vs_nhops_v2(df, output_dir, analysis_db):
    """
    Plots the effort vs. step index and overlays quadratic fits.
    """
    if df.empty or 'data_save' not in df.columns:
        print("DataFrame is empty or missing 'data_save' column, skipping plot.")
        return
    
    n_spins_unique = sorted(df['n_spins'].unique())
    
    if not n_spins_unique:
        return

    fig, axs = plt.subplots(len(n_spins_unique), 1, figsize=(10, 8 * len(n_spins_unique)), squeeze=False)

    for i, n_spins in enumerate(n_spins_unique):
        ax = axs[i, 0]
        spins_df = df[df['n_spins'] == n_spins]

        # All groups to plot
        groups_to_plot = []
        # Classical
        classical_df = spins_df[spins_df['m_quantum_replicas'] == 0]
        if not classical_df.empty:
            groups_to_plot.append((
                {'m_replicas': classical_df['m_replicas'].iloc[0], 'm_quantum_replicas': 0, 'm_cg': classical_df['m_cg'].iloc[0], 'n_spins': n_spins},
                classical_df,
                'Classical (m_q=0)',
                'lightgreen'
            ))

        quantum_df = spins_df[spins_df['m_quantum_replicas'] > 0]
        if not quantum_df.empty:
            for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg']):
                m_rep, m_q, m_cg = params
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray') # Use the static map, with a default color

                groups_to_plot.append((
                    {'m_replicas': m_rep, 'm_quantum_replicas': m_q, 'm_cg': m_cg, 'n_spins': n_spins},
                    group,
                    f'm={m_rep}, m_q={m_q}, m_cg={m_cg}',
                    color
                ))

        print("groups_to_plot:", groups_to_plot)
        
        for params, group_df, label, color in groups_to_plot:
            if not group_df['data_save'].empty:
                try:
                    full_data = np.concatenate([arr for arr in group_df['data_save'] if arr.ndim == 2 and arr.shape[0] > 0])
                    if full_data.size == 0:
                        continue
                    full_data = full_data[full_data[:, 0] > 0]

                    steps = full_data[:, 0]
                    efforts = full_data[:, 2]

                    unique_steps = np.unique(steps)
                    mean_efforts = [np.mean(efforts[steps == h]) for h in unique_steps]
                    sem_efforts = [np.std(efforts[steps == h], ddof=1) / np.sqrt(len(efforts[steps == h])) if len(efforts[steps == h]) > 1 else 0 for h in unique_steps]
                        
                    errorbar_plot = ax.errorbar(unique_steps, mean_efforts, yerr=sem_efforts, marker='.', linewidth = 0, label=label, color=color)

                    # Analyze, save, and plot fit
                    num_lowest_points_ = 15
                    argsorted = np.argsort(mean_efforts)
                    unique_steps_sorted = unique_steps[argsorted]
                    mean_efforts_sorted = np.array(mean_efforts)[argsorted]
                    optimal_n_hops, optimal_effort, popt, opt_n_hops_error = analyze_and_save_optimal_effort(analysis_db, full_data, params, num_lowest_points=num_lowest_points_)
                    ax.plot(unique_steps_sorted[:num_lowest_points_], mean_efforts_sorted[:num_lowest_points_], 'x', color=errorbar_plot.lines[0].get_color(), alpha=0.5, label=f'Fit points for {label}')
                    if popt is not None:
                        plot_steps = np.linspace(min(unique_steps_sorted), max(unique_steps_sorted), 200)
                        ax.plot(plot_steps, quad_func(plot_steps, *popt), linestyle=':', color=errorbar_plot.lines[0].get_color())
                        
                        if optimal_n_hops is not None:
                           ax.axvline(optimal_n_hops, linestyle='--', color=errorbar_plot.lines[0].get_color(), alpha=0.7, 
                                      label=f'Optimal h={optimal_n_hops:.2f}')


                except (ValueError, IndexError) as e:
                    print(f"Could not process 'data_save' for {label} at n_spins={n_spins}. Error: {e}")

        ax.set_yscale('log')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Effort (log scale)')
        ax.set_title(f'Effort vs. Hops for {n_spins} Spins')
        ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'effort_vs_nhops_v2.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

def plot_success_prob_vs_nhops(df, output_dir):
    """
    Plots the success probability vs. the number of hops (n_hops) for each n_spins.
    """
    if df.empty or 'data_save' not in df.columns:
        print("DataFrame is empty or missing 'data_save' column, skipping plot.")
        return

    n_spins_unique = sorted(df['n_spins'].unique())
    if not n_spins_unique:
        return

    fig, axs = plt.subplots(len(n_spins_unique), 1, figsize=(12, 8 * len(n_spins_unique)), squeeze=False)

    for i, n_spins in enumerate(n_spins_unique):
        ax = axs[i, 0]
        spins_df = df[df['n_spins'] == n_spins]

        # --- Group data for plotting ---
        groups_to_plot = []
        
        # Classical runs
        classical_df = spins_df[spins_df['m_quantum_replicas'] == 0]
        if not classical_df.empty:
            groups_to_plot.append((classical_df, 'Classical (m_q=0)', 'lightgreen'))

        # Quantum runs
        quantum_df = spins_df[spins_df['m_quantum_replicas'] > 0]
        if not quantum_df.empty:
            for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg']):
                m_rep, m_q, m_cg = params
                color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                label = f'm={m_rep}, m_q={m_q}, m_cg={m_cg}'
                groups_to_plot.append((group, label, color))

        # --- Plotting loop ---
        for group_df, label, color in groups_to_plot:
            if not group_df['data_save'].empty:
                try:
                    # Concatenate data from all runs in the group
                    full_data = np.concatenate([arr for arr in group_df['data_save'] if arr.ndim == 2 and arr.shape[0] > 0])
                    if full_data.size == 0:
                        continue
                    
                    # The 'data_save' array has [n_hops, m_replicas, effort, probability]
                    hops = full_data[:, 0]
                    probs = full_data[:, 3]

                    unique_hops = np.unique(hops)
                    mean_probs = [np.mean(probs[hops == h]) for h in unique_hops]
                    sem_probs = [np.std(probs[hops == h], ddof=1) / np.sqrt(len(probs[hops == h])) if len(probs[hops == h]) > 1 else 0 for h in unique_hops]
                    
                    ax.errorbar(unique_hops, mean_probs, yerr=sem_probs, marker='.', linestyle='-', label=label, color=color, alpha=0.8)

                except (ValueError, IndexError) as e:
                    print(f"Could not process 'data_save' for {label} at n_spins={n_spins}. Error: {e}")

        ax.set_yscale('log')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('Success Probability (log scale)')
        ax.set_title(f'Success Probability vs. Hops for {n_spins} Spins')
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'success_prob_vs_nhops.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    
    # Define the output directory for plots
    output_dir = Path(__file__).resolve().parent / "plots_v2"
    output_dir.mkdir(exist_ok=True)

    # An empty dictionary will load all data from the database.
    query_params = {}

    print(f"Loading data with parameters: {query_params}")
    
    # Load simulation data
    this_dir = Path(__file__).resolve().parent
    results_dir = this_dir / "results"
    main_df = load_data_from_db(db_path=results_dir / 'simulation_results_v2.json', query_params=query_params)
    
    # DB for analysis results
    analysis_db = SimulationDB(this_dir / 'analysis_results.json')

    if main_df.empty:
        print("No simulation data found. Exiting.")
    else:
        print(f"Successfully loaded {len(main_df)} records from simulation database.")
        main_df.head(30)
        # Generate plots and analysis
        plot_effort_vs_nhops_v2(main_df, output_dir, analysis_db)
        plot_success_prob_vs_nhops(main_df, output_dir)
        analysis_db.close()
        analysis_df  = load_data_from_db(db_path=this_dir / 'analysis_results.json')
        # Now, load the analysis data we just created to plot the optimal effort
        #analysis_df = load_data_from_db(db_path=this_dir / 'analysis_results.json')
        analysis_df.head()
        plot_optimal_effort_vs_nspins(analysis_df, output_dir)
        

        print("Plotting pipeline finished.")
        
    
