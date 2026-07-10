from tqdm import tqdm
import warnings

from qept.utils import get_models
from qept.analysis_code.pt_analyser import PTParamAnalyzer#do_pt, get_effort_p, get_models
from pathlib import Path
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from qemcmc.coarse_grain import CoarseGraining
    

if __name__ == "__main__":
    num_models = 5  # Number of models to optimize
    dir_ = Path(__file__).resolve().parent
    n_spins = [4,6,8,10,12,14,16,18,20,24]
    m_replicas = [4,]#, 11, 12, 13, 14, 15]
    m_quantum_replicas = [1,]#[1,1,1,1,2,2,2,2]
    m = 2
    for m_idx, m_replica in tqdm(list(enumerate(m_replicas)), desc="m_replicas"):
        m_quantum_replica = m_quantum_replicas[m_idx]
        print(f"Running PT optimization for m_replicas={m_replica}")
        print(f"The lowest {m_quantum_replica} of which are quantum")
        
        


        for n_spin in tqdm(n_spins, desc="n_spins", leave=False):


            if m_quantum_replica >0:
                gamma = (0.25,0.6)
                time_ = (2,20)
                delta_time = 0.8
                
                quantum_args_dict = {'gamma': (0.25, 0.6), 'time': (2, 20), 'delta_time': delta_time, "m":m}
            else:
                quantum_args_dict = None



            PTan = PTParamAnalyzer(
                n_spins=n_spin,
                m_replicas=m_replica,
                models=get_models(n_spin, dir_ / "models"),
                proposals = ["local",]*(m_replica-m_quantum_replica) + ["qemcmc"]*m_quantum_replica,
                quantum_args_dict=quantum_args_dict
            )

            PTan.tag = "PT"
            optimal_params, optimal_efforts, savers = PTan.run(num_models=num_models)
        
            
