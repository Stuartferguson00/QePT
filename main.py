
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




    n_spin = 10
    n_steps= 300
    m_replica = 4
    m_quantum_replica = 2
    q_reps = 20
    c_reps = 100
    
    
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
    temps = np.logspace(np.log10(high_temp), np.log10(low_temp), m_replica)

    #pick a random integer, represeni=ting a model, between 0 and 100
    
    model_index = np.random.randint(0, 100)
    model = get_models(n_spin, models_path=dir_/'models')[model_index]
    
    model_lowest = model.lowest_energy
    optimal_energies_found = []


    all_energies_found=[]


    for rep in trange(c_reps):
        replica_chains = QePT(model, proposals = ["local",]*(m_replica), quantum_args_dict = quantum_args_dict)
        current_states, energy_history = replica_chains.run(n_steps = n_steps, temps = temps, n_steps_between_exchange = 10, verbose = True)
        
        
        min_energy_found = np.min([model.get_energy(state.bitstring) for state in current_states])
        optimal_energies_found.append(min_energy_found)
        all_energies_found.append(energy_history)

    q_optimal_energies_found = []
    q_all_energies_found = []
    for rep in trange(q_reps):
        q_replica_chains = QePT(model, proposals = ["local",]*(m_replica-m_quantum_replica) + ["qemcmc"]*m_quantum_replica, quantum_args_dict = quantum_args_dict)
        q_current_states, q_energy_history = q_replica_chains.run(n_steps = n_steps, temps = temps, n_steps_between_exchange = 10, verbose = True)
        
        
        q_min_energy_found = np.min([model.get_energy(state.bitstring) for state in q_current_states])
        q_optimal_energies_found.append(q_min_energy_found)
        q_all_energies_found.append(q_energy_history)
    



    print("percentage of classical runs that found the ground state:", np.sum(np.isclose(optimal_energies_found, model_lowest)/c_reps*100))
    print("percentage of runs that found the ground state with quantum proposals:", np.sum(np.isclose(q_optimal_energies_found, model_lowest)/q_reps*100))
    all_energies_found = np.array(all_energies_found)
    q_all_energies_found = np.array(q_all_energies_found)
    # plot the average energy convergence history for all runs
    # make timesteps array,
    timesteps_ = np.linspace(0, n_steps+1, 2*(n_steps//10)+1)
    timesteps_ = np.repeat(timesteps_, 2)  # Repeat each timestep twice to match energy history recording points   

    timesteps_[::2] = timesteps_[::2]-0.5  # Shift timesteps to align with energy history recording points
    timesteps_[1::2] = timesteps_[1::2]+0.5  # Shift timesteps to align with energy history recording points
    timesteps_[0] = 0
    timesteps  = timesteps_
    plt.figure(figsize=(10, 5))
    # define a red colormap
    color = plt.cm.Reds(0.5)
    color = plt.cm.Blues(0.5)
    for i in range(all_energies_found[0].shape[0]):
        label = "PT, T = "+str(temps[i])

        color = plt.cm.Reds(0.5 + 0.5*i/all_energies_found[0].shape[0])  # Vary color for each run
        plt.plot(timesteps, np.mean(all_energies_found[:,i,:],axis = 0), label=label, color=color, alpha=0.7)
    for i in range(q_all_energies_found[0].shape[0]):
        label = "QePT, T = "+str(temps[i])
        color = plt.cm.Blues(0.5 + 0.5*i/q_all_energies_found[0].shape[0])  # Vary color for each run
        plt.plot(timesteps,np.mean(q_all_energies_found[:,i,:], axis = 0), label=label, color=color, alpha=0.7)


    # plot exact minima
    plt.axhline(y=model_lowest, color='red', linestyle='--', label='Ground Truth')  

    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.title('Average Energy Convergence History')
    plt.legend()
    plt.show()








    # plot bar graph of results
    plt.figure(figsize=(8, 6))
    plt.hist(optimal_energies_found, bins=20, alpha=0.7, color='blue', label='QePT Results')
    plt.hist(q_optimal_energies_found, bins=20, alpha=0.7, color='orange', label='QePT with Quantum Proposals')
    plt.axvline(x=model_lowest, color='red', linestyle='--', label='Classical Ground Truth')    
    plt.xlabel('Energy found')
    plt.ylabel('Counts')
    plt.title('Optimal Energies Found by QePT')
    plt.legend()
    plt.show()  

