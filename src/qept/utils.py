import pickle
import numpy as np
from qemcmc import EnergyModel
from pathlib import Path
from typing import List, Tuple, Union
import os

def get_effort_from_p(p:float, s:float, evals_per_run:int):
    # Calculate the expected number of runs required to find the optimal solution
    R = np.log(1 - s) / np.log(1 - p)
    
    # Calculate the effort as the product of the expected number of runs and the number of hops per run
    effort = R * evals_per_run
    return effort


def get_optimal_N_hops_from_data(n_spins,results_dir,proposal,gridsearch = False):
    """
    Get optimal N_hops from experimental data.
    """
    #if proposal == "local":
    #    proposal = "local_2"
    sa_results = load_sa_results([n_spins,], results_dir, proposal=proposal, gridsearch = gridsearch)

    optimal_n_hops = sa_results['mean_optimal_nhops']
    return optimal_n_hops

def get_effort_p(
    global_optima: float,
    results: Union[List[float], np.ndarray],
    s: float,
    evals_per_run: int
) -> Tuple[float, float]:
    """
    Calculate the effort and probability of finding the global optimum.

    Args:
        global_optima (float): The global minimum energy value.
        results (array-like): List of final energies from multiple SA runs.
        s (float): Required success probability.
        evals_per_run (int): Number of evaluations (hops) per SA run.

    Returns:
        tuple:
            effort (float): Estimated computational effort required to find the optimum.
            p (float): Probability of finding the global optimum in the results.
    """
    # If no optimal values found in results, must return maximum effort (bound)
    bound = evals_per_run * len(results) * (100*s) # cannot be inifity because of skopt
    
    # Probability of finding the optimal solution
    p = np.sum(np.isclose(np.array(results), global_optima)) / len(results)
    
    if p >= s:
        return evals_per_run, 1
    
    # If no results or all results are NaN, return NaN effort and probability
    if np.any(np.isnan(results)):
        return np.nan, np.nan
    
    # If no solutions found, return maximum effort
    elif np.isclose(p, 0):
        return bound, 0  

    # Calculate the expected number of runs required to find the optimal solution
    R = np.log(1 - s) / np.log(1 - p)
    
    if np.any(not np.isfinite(R)):
        return bound, 0
    
    # Calculate the effort as the product of the expected number of runs and the number of hops per run
    effort = R * evals_per_run
    return effort, p
def get_models(n_spins: int, models_path: str = 'models') -> List[EnergyModel]:
    """
    Load models from a pickle file.
    """
    str_nspins = str(n_spins).zfill(3)
    
    model_dir = Path(models_path) / f'{str_nspins}.obj'
    model_dir = model_dir.resolve()
    with open(model_dir, 'rb') as f:
        models = pickle.load(f)
    print("Got models from: ", model_dir)
    return models

def save_models(n_spins, models):
    """
    Save models to a pickle file.
    """
    str_nspins = str(n_spins).zfill(3)
    model_dir = Path('models') / f'{str_nspins}.obj'
    model_dir = model_dir.resolve()
    with open(model_dir, 'wb') as f:
        pickle.dump(models, f)
        
    
def load_sa_results(n_spins_list, results_dir, proposal, gridsearch = False, ten = False):
    """
    Loads simulated annealing results for a given proposal type.
    """
    results = {
        'mean_optimal_efforts': [],
        'sem_optimal_efforts': [],
        'mean_optimal_nhops': [],
        'sem_optimal_nhops': [],
        'data_save': []
    }

    for n_spins in n_spins_list:
        if gridsearch:
            if proposal == "local":
                results_path = os.path.join(results_dir, 'results_SA_gridsearch', 'opt_results_local', f"{str(n_spins).zfill(3)}.pkl")
                if ten:
                    results_path = os.path.join(results_dir, 'results_SA_gridsearch_10', 'opt_results_local', f"{str(n_spins).zfill(3)}.pkl")
            elif proposal == "qemcmc":
                results_path = os.path.join(results_dir, 'results_SA_gridsearch', 'opt_results_qemcmc', f"{str(n_spins).zfill(3)}.pkl")
            else:
                raise ValueError(f"Unknown proposal type: {proposal}")
            
        else:
            if proposal == "local":
                results_path = os.path.join(results_dir, 'results_SA', 'opt_results_local', f"{str(n_spins).zfill(3)}.pkl")
            elif proposal == "qemcmc":
                results_path = os.path.join(results_dir, 'results_SA', 'opt_results_qemcmc', f"{str(n_spins).zfill(3)}.pkl")
            elif proposal == "local_2":
                results_path = os.path.join(results_dir, 'results_SA', 'opt_results', f"{str(n_spins).zfill(3)}.pkl")
            elif proposal == "uniform":
                results_path = os.path.join(results_dir, 'results_SA', 'opt_results_uniform', f"{str(n_spins).zfill(3)}.pkl")
            else:
                raise ValueError(f"Unknown proposal type: {proposal}")

        try:
            with open(results_path, 'rb') as f:
                data = pickle.load(f)
            results['mean_optimal_efforts'].append(data['mean_optimal_efforts'])
            results['sem_optimal_efforts'].append(data['sem_optimal_efforts'])
            results['mean_optimal_nhops'].append(data['mean_optimal_nhops'])
            results['sem_optimal_nhops'].append(data['sem_optimal_nhops'])
            results['data_save'].append(data['data_save'])
        except FileNotFoundError:
            for key in results:
                results[key].append(np.nan)
            print("No results file found at", results_path)

    for key, value in results.items():
        results[key] = value#np.array(value)

    return results