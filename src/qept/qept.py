"""
Quantum-Enhanced Parallel Tempering (QePT) Implementation

This module implements a parallel tempering algorithm that can utilize quantum-enhanced
MCMC (QeMCMC) alongside classical MCMC methods for enhanced sampling of glassy boltzmann distribtuions. 
Primarilty built for optimisation, but can easily be refactored for sampling applications. 
"""

from qemcmc.MCMC import MCMC
from qemcmc.QeMCMC_ import QeMCMC
from qemcmc.ClassicalMCMC import ClassicalMCMC
from qemcmc.helpers import get_random_state
from qemcmc.helpers import MCMCState 
from qemcmc.energy_models import EnergyModel
from typing import List, Dict
import numpy as np
from tqdm import trange


class QePT():
    """
    Quantum-Enhanced Parallel Tempering Algorithm
    
    This class implements a parallel tempering Monte Carlo algorithm that supports
    multiple proposal methods including classical local/uniform moves and quantum-enhanced
    MCMC (QeMCMC) methods. Different replicas can use different proposal mechanisms.
    
    """
    def __init__(self,  model:EnergyModel, proposals:List[str], quantum_args_dict: Dict = None):
        """
        Initialize the QePT algorithm with specified model and proposal methods.
        
        Args:
            model: The energy model to be sampled. Must have methods:
                    - get_energy(state): Return energy of a given state
                    - num_spins: Number of spins/variables in the system
            proposals (list): List of proposal method strings for each replica.
                    Valid options: 'local', 'uniform', 'qemcmc'
            quantum_args_dict (dict, optional): Dictionary containing quantum MCMC parameters.
                    Required keys for 'qemcmc' proposals:
                    - 'gamma': Quantum parameter
                    - 'time': Evolution time parameter
                    Optional keys:
                    - 'delta_time': Time step (default: 0.8)
        
        Raises:
            ValueError: If invalid proposal method is specified or required quantum
                        arguments are missing for QeMCMC replicas.
        """
        self.model = model
        self.m_replicas = len(proposals)
        
        # Initialize MCMC objects for each replica based on proposal method
        mcmcs = []
        for proposal in proposals:
            if proposal == "local":
                # Classical MCMC with local proposal (single spin flip)
                mcmc = ClassicalMCMC( model, np.nan, proposal)  # Explicitly call classical MCMC's __init__
            elif proposal == "uniform":
                # Classical MCMC with uniform proposal
                mcmc = ClassicalMCMC( model, np.nan, proposal)
            elif proposal == "qemcmc":
                # Quantum-enhanced MCMC - requires additional parameters
                try:
                    gamma = quantum_args_dict['gamma']
                    time = quantum_args_dict['time']
                    delta_time = quantum_args_dict.get('delta_time', 0.8)  # Default to 0.8 if not provided
                except KeyError as e:
                    raise ValueError(f"Missing required quantum argument: {e}")
                except TypeError:
                    raise ValueError("quantum_args_dict must be provided for 'qemcmc' proposals")
                
                mcmc = QeMCMC(model, gamma, time, np.nan, delta_time)
                self.update = mcmc.get_s_prime  # Store update function reference
                #print("sample_sizes: ", self.sample_sizes)  # Debug output (commented)
            else:
                raise ValueError(f"Invalid proposal method: {proposal}. Choose from 'local', 'uniform', 'qemcmc'")
            mcmcs.append(mcmc)

        self.mcmcs = mcmcs

    def update_n(self, mcmc: MCMC, current_state: MCMCState, n: int) -> MCMCState:
        """
        Perform n MCMC update steps on a single replica.
        
        This method applies the MCMC update procedure n times sequentially
        to evolve the current state of a replica.
        
        Args:
            mcmc (MCMC): The MCMC object for this replica
            current_state (MCMCState): Current state of the replica
            n (int): Number of update steps to perform
            
        Returns:
            MCMCState: The updated state after n steps
        """
        for m in range(n):
            current_state = self.update_once(mcmc, current_state)
        return current_state

    def update_once(self, mcmc: MCMC, current_state: MCMCState) -> MCMCState: 
        """
        Perform a single MCMC update step.
        
        This method implements the standard Metropolis-Hastings update:
        1. Propose a new state using the MCMC's proposal mechanism
        2. Calculate the energy of the proposed state
        3. Accept or reject based on the Metropolis criterion
        4. Update the current state if accepted
        
        Args:
            mcmc (MCMC): The MCMC object containing the proposal mechanism and temperature
            current_state (MCMCState): Current state containing bitstring and energy
            
        Returns:
            MCMCState: Updated state (either new proposed state if accepted, or original state)
        """
        s = current_state.bitstring
        energy_s = current_state.energy
        
        # Propose a new state using the MCMC's proposal mechanism
        s_prime = mcmc.update(s)
        # Calculate energy of the proposed state
        energy_sprime = mcmc.model.get_energy(s_prime)
        
        # Apply Metropolis acceptance criterion
        accepted = mcmc.test_accept(energy_s, energy_sprime, temperature=mcmc.temp)
        
        # Update current_state if the proposal was accepted
        if accepted:
            current_state = MCMCState(s_prime, accepted, energy_sprime)
            # Note: energy_s assignment below is redundant since we return current_state
            energy_s = energy_sprime  # This line could be removed as optimization
        return current_state
        
    def swap_accept(self, conf1: str, conf2: str, temp1: float, temp2: float) -> bool:
        """
        Determine whether to accept a replica exchange (swap) between two configurations.
        
        This method implements the standard parallel tempering acceptance criterion:
        P_accept = min(1, exp(Δ)) where Δ = (1/T₁ - 1/T₂)(E₁ - E₂)
        
        Args:
            conf1: Configuration (bitstring) of first replica
            conf2: Configuration (bitstring) of second replica  
            temp1 (float): Temperature of first replica
            temp2 (float): Temperature of second replica
            
        Returns:
            bool: True if swap should be accepted, False otherwise
            
        Note:
            This method recalculates energies which could be optimized by passing
            pre-calculated energies as parameters.
        """
        # Calculate energy difference weighted by temperature difference
        delta = (1/temp1 - 1/temp2) * (self.model.get_energy(conf1) - self.model.get_energy(conf2))
        # Apply Metropolis criterion for replica exchange
        return np.exp(delta) > np.random.uniform(0, 1)
    
    def run(self, n_steps: int, temps: np.ndarray, n_steps_between_exchange: int, verbose: bool = False) -> tuple:  # Fixed return type
        """
        Execute the complete QePT algorithm for the specified number of steps.
        
        This method runs the parallel tempering algorithm with the following procedure:
        1. Initialize random configurations for all replicas
        2. For each exchange cycle:
            a. Update all replicas in parallel for n_steps_between_exchange steps
            b. Attempt replica exchanges between adjacent temperature pairs
            c. Record energies and swap statistics
        
        Args:
            n_steps (int): Total number of MCMC steps to perform
            temps (np.ndarray): Array of temperatures for each replica (must match m_replicas)
            n_steps_between_exchange (int): Number of MCMC steps between replica exchange attempts
            
        Returns:
            tuple: (swap_tracker, energies, current_states) where:
                - swap_tracker (np.ndarray): Record of successful swaps between replicas
                - energies (np.ndarray): Energy trajectories for all replicas
                - current_states (list): Final states of all replicas
                

        """        

        # Initialize random starting configurations for all replicas
        current_states = []
        for i, mcmc in enumerate(self.mcmcs):
            mcmc.temp = temps[i]  # Assign temperature to each replica
            # Generate random initial state - Note: calling get_random_state twice may be inefficient
            random_state = get_random_state(self.model.num_spins)
            initial_energy = mcmc.model.get_energy(random_state)
            current_states.append(MCMCState(random_state, True, initial_energy))
            #energies[i, 0] = current_states[i].energy

        #print("n_steps // n_steps_between_exchange:", n_steps // n_steps_between_exchange)
        n_steps_between_exchange = n_steps_between_exchange//2  # Adjust for odd-even exchange scheme
        # Main parallel tempering loop
        for n in trange((n_steps // (n_steps_between_exchange))+1, desc="Running QePT", leave=False, disable = not verbose):

            # Update each replica in parallel for n_steps_between_exchange steps
            
            #updated_states = Parallel(n_jobs=-1)(
            #    delayed(self.update_n)(mcmc, current_states[i], n_steps_between_exchange)
            #    for i, mcmc in enumerate(self.mcmcs)
            #)

            # Not actually in parallel...
            if n < n_steps // n_steps_between_exchange:
                updated_states = [
                    self.update_n(mcmc, current_states[i], n_steps_between_exchange)
                    for i, mcmc in enumerate(self.mcmcs)
                ]
            else:
                remaining_steps = n_steps % n_steps_between_exchange
                updated_states = [
                    self.update_n(mcmc, current_states[i], remaining_steps)
                    for i, mcmc in enumerate(self.mcmcs)
                ]
            current_states = updated_states

            # Attempt replica exchanges in an odd-even manner

            
            if n % 2 == 0:
                #Odd swaps
                for i in range(1, self.m_replicas - 1, 2):  # Odd pairs
                    if self.swap_accept(current_states[i].bitstring, current_states[i + 1].bitstring, temps[i], temps[i + 1]):
                        current_states[i], current_states[i + 1] = current_states[i + 1], current_states[i]
                #print(f"Completed (odd) exchange step {n+1}, at {(n+1) * n_steps_between_exchange + 1}")
            else:
                #Even swaps
                for i in range(0, self.m_replicas - 1, 2):  # Even pairs
                    if self.swap_accept(current_states[i].bitstring, current_states[i + 1].bitstring, temps[i], temps[i + 1]):
                        current_states[i], current_states[i + 1] = current_states[i + 1], current_states[i]
                #print(f"Completed (even) exchange step {n+1}, at {(n+1) * n_steps_between_exchange + 1}")


        return current_states
