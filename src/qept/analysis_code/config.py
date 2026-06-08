"""
Configuration file for the QOptimizer package.
"""

# General settings
REPS = 100
HIGH_TEMP = 10.0
LOW_TEMP = 0.01
SUCCESS_PROBABILITY = 0.99

# Problem sizes
N_SPINS_LIST = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35]

# Bayesian optimization settings
N_RANDOM_STARTS = 20
N_CALLS = 30

# Search space ranges for n_hops and m_replicas
# These are defined as a function to allow for dynamic range selection based on n_spins.
def get_sa_search_space(n_spins: int) -> tuple[int, int]:
    """Returns the (min_n, max_n) search space for SA."""
    if 3 <= n_spins < 8:
        return 3, 100
    elif 8 <= n_spins < 12:
        return 10, 300
    elif 12 <= n_spins < 15:
        return 10, 500
    elif 15 <= n_spins < 20:
        return 15, 1000
    elif 20 <= n_spins < 25:
        return 20, 2000
    elif 25 <= n_spins <= 30:
        return 25, 5000
    elif n_spins > 30:
        return 30, 10000
    return 3, 100  # Default

def get_pt_search_space(n_spins: int) -> tuple[int, int]:
    """Returns the (min_n_hops, max_n_hops) search space for PT."""
    if 3 <= n_spins < 8:
        return 3, 50
    elif 8 <= n_spins < 10:
        return 10, 100
    elif 10 <= n_spins < 12:
        return 10, 200
    elif 12 <= n_spins < 15:
        return 10, 300
    elif 15 <= n_spins < 20:
        return 15, 500
    elif 20 <= n_spins < 25:
        return 20, 800
    elif 25 <= n_spins <= 30:
        return 25, 1000
    elif n_spins > 30:
        return 30, 2000
    return 3, 100  # Default