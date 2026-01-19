# Obtaining the bi-objetive Pareto Front (Markowitz) using ev-MOGA
# 2 objectives

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import evmoga as ev
from evmoga import utilities as evu

import portfolio_selection_user_functions as uf

# ------------------------------------------------------------------------------------------
# Description and other information about the data:

dataset = "Eurostoxx50"
# dataset = "DowJones"
# dataset = "FTSE100"
# dataset = "NASDAQ100"
# dataset = "SP500"

excel_input_data = "./Datasets/Datasets_Sustainability/" + dataset + ".xlsx"
price_sheet_name = "AssetPrices"
esg_sheet_name = "ESG"

description = 'Mean-Variance (2 objectives) - ' + dataset
objectives = ["Mean return", "Variance"]
signs = [-1, +1] # -1: maximize, +1: minimize

# ------------------------------------------------------------------------------------------
# Options:

save_results = True
mat_folder = "./mat_files/"
mat_filename = mat_folder + "evMOGApy_" + dataset + "_2obj.mat" 

# ------------------------------------------------------------------------------------------
# Import data to define the optimization problem

df = pd.read_excel(excel_input_data, sheet_name=price_sheet_name, index_col=0).dropna(axis=1)
df = df.pct_change(axis=0).dropna(axis=0) # Daily returns:
# display(df)

returns = df.values.T
print(f"Returns shape = {returns.shape}")

mean_r = returns.mean(axis=1) # Mean returns
print(f"Shape of Mean returns = {mean_r.shape}")
cov_Mtrx = np.cov(returns)
print(f"Shape of Covariance matrix = {cov_Mtrx.shape}")

# ------------------------------------------------------------------------------------------
# Setting-up and running ev-MOGA

n_obj = len(signs) # Number of objectives
n_var = mean_r.shape[0] # Number of decision variables

eMOGA = {
    
    'description': description,
    'objectives': objectives,
    
    'objfun': uf.objective_function_2obj,
    'iterationfun': uf.fun_iteration,
    'resultsfun': uf.fun_results,
    'constraintfun': uf.sum1,

    'objfun_dim': n_obj,
    'searchSpace_dim': n_var,
    'searchspaceUB': np.ones(n_var),
    'searchspaceLB': np.zeros(n_var),
    
    'Nind_P': int(5000),
    'Generations': int(200),    
    'Nind_GA': int(200),
    'n_div': [200 for i in range(n_obj)],

    'param': {
        'signs': signs,
        'mean_r': mean_r,
        'ret': returns,
        'cov_Mtrx': cov_Mtrx,
    },

    'Sigma_Pm_ini': 20.0,
    'Sigma_Pm_fin': 0.1,

    'Pm': 0.5,
    'randseed': 12345,

    'precision_onoff': True,
    'precision': 0.001,

    'save_results': save_results,
    'mat_file': mat_filename,

    'Nit': 10,
    'time_Nit_gen': [],
    'Nind_A_Nit_gen': [],
}

eMOGA = uf.generate_P0(eMOGA)

eMOGA = ev.MOGA(eMOGA)

# ------------------------------------------------------------------------------------------
# Plotting results

dot_size_2D = 20
opt_dot_size_factor_2D = 4

eMOGA['signs'] = np.array(eMOGA['param']['signs'])
cmap = 'winter'

if 'graph options' not in eMOGA:
    eMOGA['graph options'] = dict()
eMOGA['graph options']['obj_ord_3D'] = [1, 2, 0]
eMOGA['graph options']['dot_size_2D'] = dot_size_2D
eMOGA['graph options']['opt_dot_size_factor_2D'] = opt_dot_size_factor_2D

evu.plot_Pareto_Front(eMOGA, plot_optims=True)
# evu.plot_2D_projections(eMOGA)
evu.plot_Level_Diagrams(eMOGA, plot_params_LD=False)

plt.show()

# ------------------------------------------------------------------------------------------
# The End