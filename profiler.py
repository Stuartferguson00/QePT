import cProfile
import pstats
from qept.utils import get_models
from qept.qept import QePT
from pathlib import Path


def main():
    """
    Example script to calculate the spectral gap of a Parallel Tempering chain.
    """
    n_spins = 10

    try:
        models = get_models(n_spins, Path(__file__).resolve().parent/"models")
        model = models[0]
    except FileNotFoundError:
        print(f"Models for {n_spins} spins not found. Please generate them first.")
        return

    proposals = ["local", "local", "local", "qemcmc"]







    
    quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': 0.8}

    n_hops = 100
    n_steps_between_exchange = 10
    temps = [10,1,0.1,0.01]


    PT = QePT(model, proposals, quantum_args_dict=quantum_args_dict)
    current_states, energy_history = PT.run(n_hops, temps, n_steps_between_exchange=n_steps_between_exchange)

    
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats()