import itertools
import os
from tqdm import tqdm
import pickle
import qemcmc
from qemcmc.utils.helpers import *
from qemcmc.model.model_maker import ModelMaker
from qemcmc.model.energy_model import EnergyModel
from typing import List
from classical_optimiser import Classical_Solver
import numpy as np
import dimod
import time

# Basic helper code to initialise a list Ising models of type required by cgqemcmc
# Once created, Models are pickled so they can be easily accessed later.
<<<<<<< HEAD
<<<<<<< HEAD
for n_spins in [11,13,17,19]:#np.arange(4,20):
=======
for n_spins in [50]:#np.arange(4,20):
>>>>>>> 5f9d5b9aa35733bc0d084e6cbe5f414ae73849d2
=======
for n_spins in [50]:#np.arange(4,20):
>>>>>>> 5f9d5b9aa35733bc0d084e6cbe5f414ae73849d2

    reps = 100

    dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = dir+'/models/'


    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #change file names for easy file organisation
    str_nspins = str(n_spins).zfill(3)

    model_dir = model_dir + str_nspins + '.obj'

    models = []
    for i in tqdm(range(0,reps)):
        

        

        model = ModelMaker(n_spins, "Fully Connected Ising", f"{str_nspins}_rep:{i}").model
        models.append(model)
        model.lowest_energy = Classical_Solver().solve(model)[0]
        #model.lowest_energy = Classical_Solver().solve_brute(model)[0][0]
        

    print("saving models to: ", model_dir)

    fileObj = open(model_dir, 'wb')
    pickle.dump(models,fileObj)
    fileObj.close()
