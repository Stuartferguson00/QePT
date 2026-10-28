import pickle
import numpy as np
import joblib
from skopt.space import Real
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from qemcmc import EnergyModel
from qept.qept import QePT
from qept.utils import get_effort_p
import qept.analysis_code.config as config
from tqdm import tqdm

class PTParamAnalyzer():
    """
    A class for optimizing the parallel tempering algorithm.
    """


    def __init__(self, n_spins: int, m_replicas: int, models: List[EnergyModel], proposals: Union[str,List[str]], quantum_args_dict: Optional[Dict[str, Any]] = None):
        
        self.n_spins = n_spins
        self.models = models

        self.savers = []
        self.results_dict = {}

        self.m = quantum_args_dict["m"] if quantum_args_dict is not None and "m" in quantum_args_dict else 1
        self.m_replicas = m_replicas
        self.tag = "PT"
        self.quantum_args_dict = quantum_args_dict
        self.proposals = proposals
        
        if type(proposals) is not list:
            if type(proposals) is str:
                self.proposals = [proposals]
                
            else:
                raise TypeError("proposals must be a string or a list of strings")

        for proposal in proposals:
            if proposal == "local" or proposal == "uniform":
                pass
            elif proposal == "qemcmc":
                # Quantum-enhanced MCMC - requires additional parameters
                if quantum_args_dict is None:
                    raise ValueError("quantum_args_dict must be provided for 'qemcmc' proposals")
            else:
                raise ValueError(f"Invalid proposal method: {proposal}. Choose from 'local', 'uniform', 'qemcmc'")

    
    



    def run(self, num_models: int = 1) -> Tuple[List[float], List[float], List[List[Any]]]:
        optimal_params = []
        optimal_efforts = []

        # loop through models
        for model in tqdm(self.models[0:num_models], desc="Models",miniters = 20):
        
            highest_hops = self.get_search_space()[0].high
            
            # find effort at each step
            func_vals = self.run_effort_calc(model, params=[highest_hops])
            best_idx = np.argmin(func_vals)           
            x_best = func_vals[best_idx]

            optimal_params.append(best_idx)
            optimal_efforts.append(x_best)
        
        # save results
        self.results_dict = {
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
            self.results_dict["m_replicas"] = self.m_replicas
        self.save_results(self.results_dict)
        
        
        return np.array(optimal_params), np.array(optimal_efforts), np.array(self.savers)
    


    def save_results(self, results: Dict):
        #results_path = Path("C:/Users/Stuart Ferguson/OneDrive - University of Edinburgh/Documents/PhD/CODE/QeHO/QeHO/")
        results_path = Path(__file__).resolve().parent.parent.parent.parent
        results_dir = results_path / f"results_{self.tag}"
        results_dir.mkdir(exist_ok=True)
        results_dir = results_dir / f"opt_results"
        results_dir.mkdir(exist_ok=True)
        if self.tag == "PT":
            # Build a tag string for replicas, e.g., "lll", "llq"
    
            replica_tag = ""
            for proposal in self.proposals:
                replica_tag += proposal[0]
            if replica_tag == "l"*len(self.proposals):
                res_path = results_dir / f"{str(self.n_spins).zfill(3)}_{self.m_replicas}.pkl"
            else:
                res_path = results_dir / f"{str(self.n_spins).zfill(3)}_{self.m_replicas}_{replica_tag}_{self.m}.pkl"
        else:
            res_path = results_dir / f"{str(self.n_spins).zfill(3)}.pkl"
        res_path = res_path.resolve()
        with open(res_path, "wb") as f:
            pickle.dump(results, f)
        print("Results saved to:", res_path)














    
    def get_search_space(self) -> List:
        min_n_hops, max_n_hops = config.get_pt_search_space(self.n_spins)
        return [Real(np.log(min_n_hops), np.log(max_n_hops), name="n_hops")]

    def run_effort_calc(self, model: Any, params: List, reps_overide:int = None) -> float:
        n_hops = int(np.exp(params[0]))
        temps = np.logspace(np.log10(config.HIGH_TEMP), np.log10(config.LOW_TEMP), self.m_replicas)
        n_steps_between_exchange = 2#model.n_spins
        
        if reps_overide is not None:
            reps = reps_overide
        else:
            reps = config.REPS

        jobs = (
            joblib.delayed(self.do_pt)(
                model, n_hops, self.m_replicas, temps, n_steps_between_exchange, self.proposals, self.quantum_args_dict
            )
            for i in range(reps)
            )       
        result_list = list(tqdm(
            joblib.Parallel(n_jobs=-2, return_as="generator")(jobs), 
            total=reps,
            desc="PT Iterations", disable = True
            ))
        
        # result_list = joblib.Parallel(n_jobs=-2)(
        #     joblib.delayed(self.do_pt)(
        #         model, n_hops, self.m_replicas, temps, n_steps_between_exchange, self.proposals, self.quantum_args_dict
        #     )
        #     for i in range(reps)
        # ) # returns a list of tuples (final_energy, energy_history)


        final_energies = np.array([res[0] for res in result_list if res is not None])
        energy_history = np.array([res[1] for res in result_list if res is not None]) #(number of runs x number of replicas x number of steps)
        # find the energy of the lowest T chain for each number of hops
        energies_lowest_T = energy_history[:, np.argmin(temps), :]
        
        evals_per_run = self.m_replicas * np.arange(0, energy_history.shape[2])
        
        all_efforts = []
        all_p = []
        for column in range(len(energies_lowest_T[1])):
            #print(f"Processing column {column}")
            effort, p = get_effort_p(model.lowest_energy, energies_lowest_T[:,column], config.SUCCESS_PROBABILITY, evals_per_run[column])
            all_efforts.append(effort)
            all_p.append(p)
        for i_ in range(len(energies_lowest_T[1])):
            self.savers.append([i_, self.m_replicas, all_efforts[i_], all_p[i_]])

        return all_efforts

    @staticmethod
    def do_pt(
        model: Any,
        n_hops: int,
        m_replicas: int,
        temps: np.ndarray,
        n_steps_between_exchange: int,
        proposals: List[str],
        quantum_args_dict: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Perform a single parallel tempering (PT) run using the specified model and proposal.
        """
        if len(proposals) == 1:
            proposals = proposals * m_replicas
        else:
            proposals = proposals
        PT = QePT(model, proposals, quantum_args_dict=quantum_args_dict)
        current_states, energy_history = PT.run(n_hops, temps, n_steps_between_exchange=n_steps_between_exchange, verbose = False, early_stop_energy = model.lowest_energy)
        
        
        # if energy_history is ending in Zeros, early stopping has occured, so pad results
        early_stop_point = energy_history.shape[1] - np.sum(energy_history[-1] == 0)
        if early_stop_point < energy_history.shape[1]:
            energy_history[-1,early_stop_point:] = energy_history[-1,early_stop_point-1]

        #energies = [cs.energy for cs in current_states]
        final_energy = current_states[np.argmin(temps)].energy

        return final_energy, energy_history
