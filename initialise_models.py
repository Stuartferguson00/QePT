import os
from tqdm import tqdm
import pickle
import qemcmc
from qemcmc.helpers import *
from qemcmc.ModelMaker import ModelMaker
from classical_optimiser import Classical_Solver



# Basic helper code to initialise a list Ising models of type required by cgqemcmc
# Once created, Models are pickled so they can be easily accessed later.
for n_spins in [7,]:#np.arange(4,20):

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
        MK = ModelMaker(n_spins, "Fully Connected Ising", str(n_spins) +" number: " +str(i), cost_function_signs = [-1,-1])
        model = MK.model
        model.lowest_energy = Classical_Solver().solve(model)[0]
        models.append(model)
        
    print("saving models to: ", model_dir)

    fileObj = open(model_dir, 'wb')
    pickle.dump(models,fileObj)
    fileObj.close()
