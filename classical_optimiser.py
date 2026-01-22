
import gurobipy as gp
import numpy as np
import pickle
import os

class Classical_Solver:
    def __init__(self):
        pass

    def solve(self, model):
        """
        Solve the Ising model using Gurobi.
        :param model: The Ising model to solve.
        :return: The optimal objective value and the spin configuration.
        """
        n_spins = model.num_spins
        m = gp.Model()

        # Create variables
        spins = m.addVars(n_spins, vtype=gp.GRB.BINARY, name='spins')

        # Objective: Ising Hamiltonian
        ising_expr = gp.quicksum(
            -model.J[i, j] * 0.5 * (2 * spins[i] - 1) * (2 * spins[j] - 1)
            for i in range(n_spins) for j in range(n_spins)
        ) + gp.quicksum(
            -model.h[i] * (2 * spins[i] - 1)
            for i in range(n_spins)
        )

        m.setObjective(ising_expr, gp.GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)

        # Solve it!
        m.optimize()
        optimal_value = m.objVal
        spin_configuration = [int(spins[i].X) for i in range(n_spins)]

        self.print_results(optimal_value, spin_configuration)
        
        return optimal_value, spin_configuration
    
    def print_results(self, optimal_value, spin_configuration):
        """
        Print the results of the optimization.
        :param model: The Ising model.
        :param optimal_value: The optimal objective value.
        :param spin_configuration: The spin configuration.
        """
        print(f"Optimal Value: {optimal_value}")
        print("Spin Configuration:", spin_configuration)
        #for i, spin in enumerate(spin_configuration):
        #    print(f"Spin {i}: {spin}")

    def solve_brute(self, model):
        """
        Solve the Ising model using brute force.
        :param model: The Ising model to solve.
        :return: The optimal objective value and the spin configuration.
        """
        
        n_spins = model.num_spins
        
        if n_spins > 20:
            raise ValueError("Brute force method is not feasible for n_spins > 20 due to combinatorial explosion.")
        all_energies = model.get_all_energies()
        min_energy_arg = np.argmin(all_energies)
        ground_state_bitstring = model.S[min_energy_arg]

        ground_state = all_energies[min_energy_arg]
        ground_state = model.get_lowest_energies(1)[0]

        return ground_state, ground_state_bitstring

