
# Add the parent directory to the path so we can import from qoptimizer
import os, sys, warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qept.qept import QePT
from qept.utils import get_models
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from tqdm import trange

if __name__ == "__main__":
    dir_ = Path(__file__).resolve().parent




    n_spin = 7
    m_replica = 4
    m_quantum_replica = 2
    repetitions = 2
    
    
    gamma = (0.25,0.6)
    time_ = (2,20)
    delta_time = 0.8
    quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': 0.8}
    

    if m_quantum_replica >0:
        gamma = (0.25,0.6)
        time_ = (2,20)
        delta_time = 0.8
        quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': delta_time}
    else:
        quantum_args_dict = None



    high_temp = 10.0
    low_temp = 0.01
    temps = np.logspace(high_temp, low_temp, m_replica)

    model = get_models(n_spin, models_path=dir_/'models')[0]
    
    model_lowest = model.lowest_energy
    optimal_energies_found = []


    for rep in trange(repetitions):
        replica_chains = QePT(model, proposals = ["local",]*(m_replica-m_quantum_replica) + ["qemcmc"]*m_quantum_replica, quantum_args_dict = quantum_args_dict)
        current_states = replica_chains.run(n_steps = 100, temps = temps, n_steps_between_exchange = 10, verbose = True)
        
        
        min_energy_found = np.min([model.get_energy(state.bitstring) for state in current_states])
        optimal_energies_found.append(min_energy_found)
    print("percentage of runs that found the ground state:", np.sum(np.isclose(optimal_energies_found, model_lowest)/repetitions*100))
    # plot bar graph of results
    plt.figure(figsize=(8, 6))
    plt.hist(optimal_energies_found, bins=20, alpha=0.7, color='blue', label='QePT Results')
    plt.axvline(x=model_lowest, color='red', linestyle='--', label='Classical Ground Truth')    
    plt.xlabel('Energy found')
    plt.ylabel('Counts')
    plt.title('Optimal Energies Found by QePT')
    plt.legend()
    plt.show()  

