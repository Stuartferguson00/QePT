"""
Quantum-Enhanced Parallel Tempering (QePT) Implementation

This module implements a parallel tempering algorithm that can utilize quantum-enhanced
MCMC (QeMCMC) alongside classical MCMC methods for enhanced sampling of glassy boltzmann distribtuions. 
Primarilty built for optimisation, but can easily be refactored for sampling applications. 
"""

# from qemcmc.MCMC import MCMC
# from qemcmc.QeMCMC_ import QeMCMC
# from qemcmc.ClassicalMCMC import ClassicalMCMC
# from qemcmc.helpers import get_random_state
# from qemcmc.helpers import MCMCState 
# from qemcmc.energy_models import EnergyModel

from qemcmc.model.energy_model import EnergyModel

from qemcmc.utils.helpers import get_random_state
from qemcmc.utils.helpers import MCMCState, MCMCChain

from qemcmc.sampler.runners import Runner, MCMCRunner
from qemcmc.sampler.qe_proposal import QeProposal
from qemcmc.sampler.proposal import Proposal
from qemcmc.sampler.classical_proposal import ClassicalProposal

from typing import List, Dict, Optional
import numpy as np
from tqdm import trange
from tqdm import tqdm



class MCMCRunnerAlt(Runner):
    """
    Orchestrates a standard MCMC run loop. Altered from the original, as have to be able to iterate one step at a time.

    This runner uses a given proposal sampler and energy model to generate a Markov chain
    of states. It manages state updates, energy evaluations, and Metropolis acceptance tests.
    The sampler targets the Boltzmann distribution p(s) ∝ exp(-E(s) / T).

    Parameters
    ----------
    model : EnergyModel
        The energy model defining the system.
    temp : float
        The temperature for the Metropolis acceptance criterion.
    """

    def __init__(self, model: EnergyModel, temp: float, proposer: Proposal):
        super().__init__()
        self.model = model
        self.temp = temp
        self.proposer = proposer

    def run(
        self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name: Optional[str] = None,
        verbose: bool = False,
        sample_frequency: int = 1,
    ) -> MCMCChain:
        """
        Run the MCMC simulation.

        Parameters
        ----------
        proposer : Proposal
            The proposal engine for generating new states.
        n_hops : int
            The number of MCMC steps to perform.
        initial_state : str, optional
            The starting bitstring for the chain. If None, a random state is generated.
        name : str, optional
            A name for the MCMC chain.
        verbose : bool, optional
            Enable progress bar and print statements.
        sample_frequency : int, optional
            The frequency at which to sample states for the chain. Default is ``1`` (every step).

        Returns
        -------
        MCMCChain
            The generated Markov chain of states.
        """
        if name is None:
            name = getattr(self.proposer, "method", "Standard") + " MCMC"

        if initial_state is None:
            initial_state_obj = MCMCState(get_random_state(self.model.n), accepted=True, position=0)
        else:
            initial_state_obj = MCMCState(initial_state, accepted=True, position=0)

        current_state = initial_state_obj
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state_obj.energy = energy_s

        if verbose:
            print(f"Starting with: {current_state.bitstring} with energy: {energy_s}")

        mcmc_chain = MCMCChain([current_state], name=name)

        for i in tqdm(range(0, n_hops), desc="Run " + name, disable=not verbose):
            s_prime = self.proposer.update(current_state.bitstring)
            energy_sprime = self.model.get_energy(s_prime)
            accepted = self.is_accepted(energy_s, energy_sprime, temperature=self.temp)

            if accepted:
                energy_s = energy_sprime
            current_state = self.update_once(current_state)


            if i % sample_frequency == 0 and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, True, current_state.energy, position=i))

        return mcmc_chain
    
    def update_once(self, current_state: MCMCState) -> MCMCState: 
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
        s_prime = self.proposer.update(s)
        # Calculate energy of the proposed state
        energy_sprime = self.model.get_energy(s_prime)
        
        # Apply Metropolis acceptance criterion
        accepted = self.is_accepted(energy_s, energy_sprime, temperature=self.temp)
        
        # Update current_state if the proposal was accepted
        if accepted:
            current_state = MCMCState(s_prime, accepted, energy_sprime)
            # Note: energy_s assignment below is redundant since we return current_state
            energy_s = energy_sprime  
        return current_state

class QePT(Runner):
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
                mcmc_proposal = ClassicalProposal(model, method = "local")
                mcmc_runner = MCMCRunnerAlt( model, np.nan, mcmc_proposal)
                
            elif proposal == "uniform":
                # Classical MCMC with uniform proposal
                mcmc_proposal = ClassicalProposal(model, method = "uniform")
                mcmc_runner = MCMCRunnerAlt( model, np.nan, mcmc_proposal)
            elif proposal == "qemcmc":
                # Quantum-enhanced MCMC - requires additional parameters
                try:
                    gamma = quantum_args_dict['gamma']
                    time = quantum_args_dict['time']
                    delta_time = quantum_args_dict.get('delta_time', 0.8)  # Default to 0.8 if not provided
                    m = quantum_args_dict.get('m', 1)  # Default to 1 if not provided
                except KeyError as e:
                    raise ValueError(f"Missing required quantum argument: {e}")
                except TypeError:
                    raise ValueError("quantum_args_dict must be provided for 'qemcmc' proposals")
                proposal = QeProposal(model, gamma, time, None, delta_time, m = m)
                mcmc_runner =  MCMCRunnerAlt(model, np.nan, proposal)
                #print("sample_sizes: ", self.sample_sizes)  # Debug output (commented)
            else:
                raise ValueError(f"Invalid proposal method: {proposal}. Choose from 'local', 'uniform', 'qemcmc'")
            mcmcs.append(mcmc_runner)

        self.mcmcs = mcmcs

    def update_n(self, mcmc: MCMCRunner, current_state: MCMCState, n_: int, temper_index: int, current_n: int, n_steps_between_exchange_: int) -> MCMCState:
        """
        Perform n MCMC update steps on a single replica.
        
        This method applies the MCMC update procedure n times sequentially
        to evolve the current state of a replica.
        
        Args:
            mcmc (MCMC): The MCMC object for this replica
            current_state (MCMCState): Current state of the replica
            n_ (int): Number of update steps to perform
            temper_index (int): Index of the temperature for the replica
            current_n (int): Current iteration number (used for energy history indexing)
            steps_per_update (int): Number of steps between replica exchanges (used for energy history indexing)
        Returns:
            MCMCState: The updated state after n steps
        """
        for m in range(1, n_ + 1):
            current_state = mcmc.update_once(current_state)
            self.energy_history[temper_index, (current_n*n_steps_between_exchange_) + m] = current_state.energy  # Record energy history after updates
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
    
    def run(self, n_steps: int, temps: np.ndarray, n_steps_between_exchange: int, verbose: bool = False) -> np.ndarray:  # Fixed return type
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
            np.ndarray: current_states (list): Final states of all replicas
            np.ndarray: energy_history: Recorded energy history for all replicas across all steps (number of replicas x n_steps)
                

        """        
        self.energy_history = np.zeros((self.m_replicas, n_steps+1)) # Initialize energy history array to record energies at each step for all replicas
        # Initialize random starting configurations for all replicas
        current_states = []
        for i, mcmc in enumerate(self.mcmcs):
            mcmc.temp = temps[i]  # Assign temperature to each replica
            # Generate random initial state - Note: calling get_random_state twice may be inefficient
            random_state = get_random_state(self.model.n_spins)
            initial_energy = mcmc.model.get_energy(random_state)
            current_states.append(MCMCState(random_state, True, initial_energy))
            self.energy_history[i, 0] = initial_energy
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
                    self.update_n(mcmc, current_states[i], n_steps_between_exchange, i,n,  n_steps_between_exchange)
                    for i, mcmc in enumerate(self.mcmcs)
                ]
            else:
                remaining_steps = n_steps % n_steps_between_exchange
                updated_states = [
                    self.update_n(mcmc, current_states[i], remaining_steps, i, n, n_steps_between_exchange)
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
                #energy_history[:, n] = [state.energy for state in current_states]  # Record energy history after odd swaps
            else:
                #Even swaps
                for i in range(0, self.m_replicas - 1, 2):  # Even pairs
                    if self.swap_accept(current_states[i].bitstring, current_states[i + 1].bitstring, temps[i], temps[i + 1]):
                        current_states[i], current_states[i + 1] = current_states[i + 1], current_states[i]
                #print(f"Completed (even) exchange step {n+1}, at {(n+1) * n_steps_between_exchange + 1}")
                #energy_history[:, n] = [state.energy for state in current_states]  # Record energy history after even swaps
            #energy_history[:, n] = [state.energy for state in current_states]  # Record energy history after updates
            

        return current_states, self.energy_history
