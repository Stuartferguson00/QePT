from tqdm import tqdm
import warnings
from pathlib import Path
import numpy as np

# database handler
from database_stuff.database_handler import SimulationDB

# Existing simulation logic imports
from qept.utils import get_models
from qept.analysis_code.pt_analyser import PTParamAnalyzer
from qemcmc.coarse_grain import CoarseGraining
import os


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)





if __name__ == "__main__":
    # Get a unique ID for this process to create a unique database file
    # This makes it safe to run multiple instances of this script at the same time.
    process_id = os.getpid()

    # Initialize the new database
    results_dir = Path(__file__).resolve().parent / "results"
    db = SimulationDB(db_path=results_dir / f'simulation_results_v2_{process_id}.json')
    print(f"Initialized database at simulation_results_v2_{process_id}.json")

    # --- Simulation Parameters ---
    # These can be adjusted as in the original script
    num_models = 5  # Number of models to optimize
    dir_ = Path(__file__).resolve().parent
    n_spins_list = [12,14,16,18,20]#np.arange(4,6,1) # Using a smaller range for demonstration
    m_replicas_list = [4]
    m_quantum_replicas_list = [2]
    m_cg_list = [2] # Coarse graining factor

    print("Starting new simulation runs...")
    # --- Main Simulation Loop ---
    # This loop is adapted from the original pt_optimal_param_finder.py
    for n_spin in tqdm(n_spins_list, desc="n_spins"):
        for m_replica in tqdm(m_replicas_list, desc="m_replicas", leave=False):
            for m_quantum_replica in tqdm(m_quantum_replicas_list, desc="m_quantum_replicas", leave=False):
                if m_quantum_replica >= m_replica:
                    continue # Skip cases where quantum replicas are more than total replicas

                # Define parameters for the run
                proposals = ["local"] * (m_replica - m_quantum_replica) + ["qemcmc"] * m_quantum_replica
                
                quantum_args_dict = None
                if m_quantum_replica > 0:
                    quantum_args_dict = {
                        'gamma': (0.25, 0.6),
                        'time': (2, 20),
                        'delta_time': 0.8,
                        "m": m_cg_list[0] # Assuming one m_cg for simplicity
                    }

                # Store all parameters for this run
                run_params = {
                    "n_spins": n_spin,
                    "m_replicas": m_replica,
                    "m_quantum_replicas": m_quantum_replica,
                    "proposals": proposals,
                    "m_cg": m_cg_list[0] if m_quantum_replica > 0 else None,
                    "num_models": num_models,
                }
                
                print(f"Running with params: {run_params}")

                analyzer = PTParamAnalyzer(
                    n_spins=n_spin,
                    m_replicas=m_replica,
                    models=get_models(n_spin, dir_ / "models"),
                    proposals=proposals,
                    quantum_args_dict=quantum_args_dict
                )

                # Run the simulation
                analyzer.run(num_models=num_models)

                # Retrieve the results and save to the database
                if analyzer.results_dict:
                    db.insert(parameters=run_params, results=analyzer.results_dict)
                    print(f"Successfully saved results for {run_params} to DB.")
                    # Explicitly flush the cache to disk to ensure data is saved
                    # in case of a crash.
                    db._db.storage.flush()
                else:
                    print("No results were generated for this run.")

    print("All simulation runs completed and data saved to database.")
    db.close()

    
