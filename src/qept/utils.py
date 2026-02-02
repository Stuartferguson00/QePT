import pickle
from typing import List
from qemcmc.model.energy_model import EnergyModel
from pathlib import Path


    
def get_models(n_spins: int, models_path: str = 'models') -> List[EnergyModel]:
    """
    Load models from a pickle file.
    """
    str_nspins = str(n_spins).zfill(3)
    
    model_dir = Path(models_path) / f'{str_nspins}.obj'
    model_dir = model_dir.resolve()
    with open(model_dir, 'rb') as f:
        models = pickle.load(f)
    return models

def save_models(n_spins, models):
    """
    Save models to a pickle file.
    """
    str_nspins = str(n_spins).zfill(3)
    model_dir = Path('models') / f'{str_nspins}.obj'
    model_dir = model_dir.resolve()
    with open(model_dir, 'wb') as f:
        pickle.dump(models, f)
        
        