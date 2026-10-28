import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tinydb
from pathlib import Path
from scipy.optimize import curve_fit
from itertools import product

# Import the new database handler
from database_stuff.database_handler import SimulationDB

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
            #optimal_effort = quad_func(optimal_n_hops, *popt)
            
            # Calculate error in nhops and effort using error propagation
            # from the covariance matrix pcov.
            # var(f(x,y)) = (df/dx)^2*var(x) + (df/dy)^2*var(y) + 2*(df/dx*df/dy)*cov(x,y)
            
            # Error for optimal_n_hops = -b / (2a)
            d_a = b / (2 * a**2)  # Partial derivative wrt a
            d_b = -1 / (2 * a)    # Partial derivative wrt b
            var_a, var_b, cov_ab = pcov[0, 0], pcov[1, 1], pcov[0, 1]
            opt_n_hops_error = np.sqrt(d_a**2 * var_a + d_b**2 * var_b + 2 * d_a * d_b * cov_ab)

            optimal_n_hops_int = int(optimal_n_hops)
            optimal_effort_at_opt_nhops = mean_efforts[optimal_n_hops_int]
            err_in_optimal_effort_at_opt_nhops = sem_efforts[optimal_n_hops_int]
            
            
            # Save results
            analysis_results = {
                'optimal_n_hops': optimal_n_hops,
                'optimal_n_hops_error': opt_n_hops_error,
                'optimal_effort': optimal_effort_at_opt_nhops,
                'error_in_effort': err_in_optimal_effort_at_opt_nhops,
                'quad_fit_params': popt.tolist()
            }
            db.insert(params, analysis_results)
            
            return optimal_n_hops, optimal_effort_at_opt_nhops, popt, opt_n_hops_error,fit_indices
        else:
            print(f"Could not process optimal data for {params}")
            return None, None, None, None, fit_indices # Not a valley

    except (RuntimeError, ValueError) as e:
        print(f"Quadratic fit failed for {params}: {e}")
        return None, None, None, None, None


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
                    'adiabatic': base_params.get('adiabatic'),
                    "alpha_zero": base_params.get("alpha_zero"),
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
                if not is_adiabatic:
                    for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg']):
                        m_rep, m_q, m_cg = params
                        color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                        label = f'm={m_rep}, m_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}'
                        process_and_plot_group(group, label, color, linestyle, marker)
                else:
                    for params, group in quantum_df.groupby(['m_replicas', 'm_quantum_replicas', 'm_cg', "alpha_zero"]):
                        m_rep, m_q, m_cg, alpha_zero= params
                        color = PARAM_COLOR_MAP.get((m_q, m_cg), 'gray')
                        label = f'm={m_rep}, m_q={m_q}, m_cg={m_cg}{adiabatic_label_suffix}, alpha_zero={alpha_zero}'
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
                    mean_probs = np.array([np.mean(probs[hops == h]) for h in unique_hops])
                    sem_probs = [np.std(probs[hops == h], ddof=1) / np.sqrt(len(probs[hops == h])) if len(probs[hops == h]) > 1 else 0 for h in unique_hops]
                    ax.errorbar(unique_hops, 1-mean_probs, yerr=sem_probs, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.8, capsize=3)

                    #ax.errorbar(unique_hops, mean_probs, yerr=sem_probs, marker=marker, linestyle=linestyle, label=label, color=color, alpha=0.8, capsize=3)
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
        ax.set_xscale('log')
        ax.set_xlabel('Number of Hops')
        ax.set_ylabel('1- Success Probability (log scale)')
        ax.set_title(f'Success Probability vs. Hops for {n_spins} Spins')
        ax.legend(fontsize='small')
        ax.grid(True, which="both", ls="-", alpha=0.2)


    plt.tight_layout()
    output_path = output_dir / 'success_prob_vs_nhops.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close(fig)



    
# A new, cleaner plot_experiments.py

# Make sure you have the required libraries installed:
# pip install pandas seaborn matplotlib
from analysis import PlottingAnalyser, flexible_plot

# --- Example fitting function ---
def stretched_exp_fit(n,a, b):
    return a * 2**(b * n**(1/3)) 

def plot_experiments(results_path):
    this_dir = Path(__file__).resolve().parent
    save_path = this_dir/'plots/new'
    analyser = PlottingAnalyser(results_path=results_path)

    print("Generating plot for m_cg = 1, styled by adiabatic, colored by m_quantum_replicas...")
    flexible_plot(
        analyser=analyser,
        x_col='n_spins',
        y_col='optimal_effort',
        y_err_col='error_in_effort',
        filters=[{'m_cg': 1, "n_spins": np.arange(4,13,1)},{"m_quantum_replicas":0}],
        hue_by="m_quantum_replicas",
        style_by="adiabatic",
        plot_title='Optimal Effort vs. Number of Spins (m_cg = 1, m_q = 1)',
        save_path=save_path/"effort_vs_nspins_m_cg_1_m_q_1.png",
        show_plot=False,
        log_y=True,
        fit_func=stretched_exp_fit,
        show_fit_params=True,
        cubic_x=True,
    )

    print("\nGenerating plot with custom palettes...")
    flexible_plot(
        analyser=analyser,
        x_col='n_spins',
        y_col='optimal_effort',
        y_err_col='error_in_effort',
        filters=[{"adiabatic":True, "alpha_zero":True},{"adiabatic":True, "alpha_zero":False}],
        hue_by='m_quantum_replicas',
        plot_title='Optimal Effort vs. N Spins',
        save_path=save_path/"effort_vs_nspins_m_q_og.png",
        show_plot=False,
        log_y=True,
        classical_baseline = True,
        fit_func=stretched_exp_fit,
        show_fit_params=True,
        cubic_x=True,
    )
    #

    flexible_plot(
            analyser=analyser,
            x_col='n_spins',
            y_col='optimal_effort',
            y_err_col='error_in_effort',
            filters=[{"m_quantum_replicas":0, "n_spins":np.arange(5,40)}],
            plot_title='Optimal Effort vs. Number of Spins (classcial)',
            save_path=save_path/"effort_vs_nspins_c.png",
            show_plot=False,
            log_y=True,
            fit_func=stretched_exp_fit,
            show_fit_params=True,
            cubic_x=True,
        )
    
    
     
   

    print("\nDone generating plots.")




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
    
    
    # This DB will store the analysis results (optimal effort, etc.)
    # It will be cleared and recreated each time the script runs.
    analysis_db_path = this_dir / 'analysis_results.json'
    if analysis_db_path.exists():
        #analysis_db_path.unlink() # Clear previous analysis
        plot_experiments(this_dir / 'analysis_results.json')
    else:
        raw_results_dir=results_dir / 'simulation_results_v2.json'
        print("analysis_results doesnt exist, creating analysis from raw data in ",raw_results_dir)
        main_df = load_data_from_db(db_path=raw_results_dir, query_params=query_params)
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
            plot_experiments(this_dir / 'analysis_results.json')

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
