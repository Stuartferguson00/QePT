import os
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from qemcmc import EnergyModel
from qept.utils import get_models, get_effort_p
from pathlib import Path
from skopt.plots import plot_gaussian_process
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
import qept.analysis_code.config as config

class Optimizer(ABC):
    """
    An abstract base class for optimizers.
    """

    def __init__(self, n_spins: int, models: List[EnergyModel], proposal: str="local", quantum_args_dict: dict = None):
        self.n_spins = n_spins
        self.models = models
        self.proposal = proposal
        self.quantum_args_dict = quantum_args_dict

        self.savers = []

    @abstractmethod
    def run_one(self, model: Any) -> float:
        pass

    def run_optimization(self,num_models:int =1, plot:bool = False) -> Tuple[List[float], List[float], List[List[Any]]]:
        optimal_params = []
        optimal_efforts = []
        for model in tqdm(self.models[0:num_models]):
            def func_to_optimise(params):
                
                return self.run_one(model, params=[params])

            result = gp_minimize(
                func_to_optimise,
                dimensions=self.get_search_space(),
                n_random_starts=config.N_RANDOM_STARTS,
                n_calls=config.N_CALLS,
                initial_point_generator="halton",
                random_state=2,
            )
            
            if plot:
                _ = plot_gaussian_process(result)
                plt.title(f"Bayesian Optimization Objective Plot for {self.proposal} with {self.n_spins} Spins")
                plt.savefig(f"bayesian_objective_plot_{self.proposal}_{self.n_spins}_{self.tag}.png")
                plt.close()

            x_vals = np.linspace(self.get_search_space()[0].low, self.get_search_space()[0].high, 500).reshape(-1, 1)
            mod = result.models[-1]
            mod = mod.fit(result.x_iters, result.func_vals)
            y_mean, y_std = mod.predict(x_vals, return_std=True)
            x_best_expected = x_vals[np.argmin(y_mean)][0]


            optimal_params.append(np.exp(x_best_expected))
            optimal_efforts.append(self.run_one(model, params=[x_best_expected,], reps_overide=config.REPS*10))

        
        
        results_dict = {
                "optimal_nhops": [int(x) for x in optimal_params],
                "mean_optimal_nhops": float(np.mean(optimal_params, axis=0)),
                "sem_optimal_nhops": float(np.std(optimal_params, axis=0, ddof=1) / np.sqrt(len(optimal_params))),
                "mean_optimal_efforts": float(np.mean(optimal_efforts)),
                "sem_optimal_efforts": float(np.std(optimal_efforts, ddof=1) / np.sqrt(len(optimal_efforts))),
                "n_spins": self.models[-1].n_spins,
                "reps": config.REPS,
                "high_temp": config.HIGH_TEMP,
                "low_temp": config.LOW_TEMP,
                "data_save": np.array(self.savers), 
            }
        if self.tag == "PT":
            results_dict["m_replicas"] = self.m_replicas
        self.save_results(results_dict)
        return np.array(optimal_params), np.array(optimal_efforts), np.array(self.savers)
    
    def run_gridsearch(self, num_models: int = 1) -> Tuple[List[float], List[float], List[List[Any]]]:
        optimal_params = []
        optimal_efforts = []
        for model in tqdm(self.models[0:num_models], desc="Models"):
            


            highest_hops = self.get_search_space()[0].high
            
            func_vals = self.run_one(model, params=[highest_hops])
            best_idx = np.argmin(func_vals)
            #print("func_vals:", func_vals)
            #print("best_idx:", best_idx)
            
            x_best = func_vals[best_idx]


            optimal_params.append(best_idx)
            optimal_efforts.append(x_best)
            
            #print("grid:", [ int(np.exp(g)) for g in grid])
        results_dict = {
            "optimal_nhops": [int(x) for x in optimal_params],
            "mean_optimal_nhops": float(np.mean(optimal_params, axis=0)),
            "sem_optimal_nhops": float(np.std(optimal_params, axis=0, ddof=1) / np.sqrt(len(optimal_params))),
            "mean_optimal_efforts": float(np.mean(optimal_efforts)),
            "sem_optimal_efforts": float(np.std(optimal_efforts, ddof=1) / np.sqrt(len(optimal_efforts))),
            "n_spins": self.models[-1].n_spins,
            "reps": config.REPS,
            "high_temp": config.HIGH_TEMP,
            "low_temp": config.LOW_TEMP,
            "data_save": np.array(self.savers),
        }
        if hasattr(self, "tag") and self.tag == "PT":
            results_dict["m_replicas"] = self.m_replicas
        self.save_results(results_dict)
        
        #print(results_dict)
        
        return np.array(optimal_params), np.array(optimal_efforts), np.array(self.savers)
    

    @abstractmethod
    def get_search_space(self) -> List:
        pass

    def save_results(self, results: Dict):
        #results_path = Path("C:/Users/Stuart Ferguson/OneDrive - University of Edinburgh/Documents/PhD/CODE/QeHO/QeHO/")
        results_path = Path(__file__).resolve().parent.parent.parent.parent
        results_dir = results_path / f"results_{self.tag}"
        results_dir.mkdir(exist_ok=True)
        results_dir = results_dir / f"opt_results_{self.proposal}"
        results_dir.mkdir(exist_ok=True)
        if self.tag == "PT" or self.tag == "PT_gridsearch":
            # Build a tag string for replicas, e.g., "lll", "llq"
            
            replica_tag = ""
            for proposal in self.proposals:
                replica_tag += proposal[0]
            if replica_tag == "l"*len(self.proposals):
                res_path = results_dir / f"{str(self.n_spins).zfill(3)}_{self.m_replicas}.pkl"
            else:
                res_path = results_dir / f"{str(self.n_spins).zfill(3)}_{self.m_replicas}_{replica_tag}.pkl"
        else:
            res_path = results_dir / f"{str(self.n_spins).zfill(3)}.pkl"
        res_path = res_path.resolve()
        with open(res_path, "wb") as f:
            pickle.dump(results, f)
        print("Results saved to:", res_path)