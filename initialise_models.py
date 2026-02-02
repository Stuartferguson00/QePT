import itertools
import os
from tqdm import tqdm
import pickle
import qemcmc
from qemcmc.utils.helpers import *
from qemcmc.utils.model_maker import ModelMaker
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
        # MK = ModelMaker(n_spins, "Fully Connected Ising Generic", str(n_spins) +" number: " +str(i), cost_function_signs = [-1,-1])
        # model = MK.model
        # model.lowest_energy = Classical_Solver().solve(model)[0]
        # models.append(model)



        subgroups = list(itertools.combinations(range(n_spins), n_spins))
        shape_of_J = (n_spins, n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu

        h = np.round(np.random.normal(0, 1, n_spins), decimals=4)

        couplings = [h, J]
        # why does the user have to calculate their own alpha? At least we should do it in model maker. Ask the user to input max_number of qubits, and we enumerate all combinations internally.
        alpha = np.sqrt(n_spins) / np.sqrt(sum([J[i][j] ** 2 for i in range(n_spins) for j in range(i)]) + sum([h[j] ** 2 for j in range(n_spins)]))

        model = qemcmc.EnergyModel(n=n_spins, couplings=couplings, subgroups=subgroups, subgroup_probs=np.ones(len(subgroups)) / len(subgroups), alpha=alpha)
        # Gurobi broken for some reason
        #model.lowest_energy = Classical_Solver().solve(model)[0]
        models.append(model)
        model.lowest_energy = Classical_Solver().solve_brute(model)[0][0]

        
    print("saving models to: ", model_dir)

    fileObj = open(model_dir, 'wb')
    pickle.dump(models,fileObj)
    fileObj.close()
