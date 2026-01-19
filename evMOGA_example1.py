# ev-MOGA example 1
# 2 objectives

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
import time

import evmoga as ev
from evmoga import utilities as evu

import evMOGA_user_functions as uf

# --------------------------------------------------------------------------------------- 
# Description and other information about the data:

description = "Example 1"
objectives = ["J1", "J2"]
signs = [+1, +1] # -1: maximize, +1: minimize

# --------------------------------------------------------------------------------------- 
# Options:

save_results = True
mat_folder = "./mat_files/"
mat_filename = mat_folder + "evMOGApy_example1.mat" 

# ---------------------------------------------------------------------------------------
# Setting-up and running ev-MOGA

n_obj = 2 # Number of objectives
n_var = 2 # Number of decision variables

eMOGA = {
    
    'description': description,
    'objectives': objectives,
    
    'objfun': uf.mop31,
    'iterationfun': uf.fun_iteration,
    'resultsfun': uf.fun_results,

    'objfun_dim': n_obj,
    'searchSpace_dim': n_var,
    'searchspaceUB': +np.pi * np.ones(n_var),
    'searchspaceLB': -np.pi * np.ones(n_var),
    
    'Nind_P': int(500),
    'Generations': int(200),    
    # 'Nind_GA': int(200),
    'n_div': [200 for i in range(n_obj)],

    'param': {
        'signs': signs,
    },

    'Nit': 10,
}

eMOGA = ev.MOGA(eMOGA)

# ---------------------------------------------------------------------------------------
# Plotting results

dot_size_2D = 20
opt_dot_size_factor_2D = 4

eMOGA['signs'] = np.array(eMOGA['param']['signs'])
cmap = 'winter'

if 'graph options' not in eMOGA:
    eMOGA['graph options'] = dict()
eMOGA['graph options']['obj_ord_2D'] = [0, 1]
eMOGA['graph options']['dot_size_2D'] = dot_size_2D
eMOGA['graph options']['opt_dot_size_factor_2D'] = opt_dot_size_factor_2D

fig = evu.plot_Pareto_Front(eMOGA, plot_optims=True)
evu.plot_Level_Diagrams(eMOGA, plot_params_LD=False)

# ---------------------------------------------------------------------------------------
# Comparing with the exact solution:

mop3aux = loadmat("./evMOGAtoolbox/evMOGAtoolbox/mop3aux.mat")
pfrontaux = mop3aux['pfront']

fig = plt.figure(figsize=(7, 7))
plt.scatter(pfrontaux[:,0], pfrontaux[:,1],
            marker='.', s=0.25, c='r', 
            label='True Pareto Front')

plt.scatter(eMOGA['coste_A'][:,0], eMOGA['coste_A'][:,1], 
            marker='o', s=25, facecolors='none', edgecolors='b', linewidths=0.5,
            label='ev-MOGA Pareto Front')

plt.xlabel('J1')
plt.legend()
plt.grid()
plt.show()