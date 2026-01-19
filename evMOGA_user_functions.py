
import numpy as np 
from scipy.io import savemat

import time
from tqdm import tqdm

# ---------------------------------------------------------------
# Objective function:

def mop31(theta, params):

    a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2);
    a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2);
    b1 = 0.5 * np.sin(theta[0]) - 2 * np.cos(theta[0]) + np.sin(theta[1]) - 1.5 * np.cos(theta[1]);
    b2 = 1.5 * np.sin(theta[0]) - np.cos(theta[0]) + 2 * np.sin(theta[1]) - 0.5 * np.cos(theta[1]);

    J1 = 1 + (a1 - b1)**2 + (a2 - b2)**2;
    J2 = (theta[0] + 3)**2 + (theta[1] + 1)**2;
    
    return np.array([J1, J2]) * params['signs']

# ---------------------------------------------------------------
# Objective function:

def mop31_p(theta, params):

    k = params['const']

    a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - k * np.cos(2);
    a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2);
    b1 = 0.5 * np.sin(theta[0]) - 2 * np.cos(theta[0]) + np.sin(theta[1]) - 1.5 * np.cos(theta[1]);
    b2 = 1.5 * np.sin(theta[0]) - np.cos(theta[0]) + 2 * np.sin(theta[1]) - 0.5 * np.cos(theta[1]);

    J1 = 1 + (a1 - b1)**2 + (a2 - b2)**2;
    J2 = (theta[0] + 3)**2 + (theta[1] + 1)**2;
    
    return np.array([J1, J2]) * params['signs']

# ---------------------------------------------------------------
# Iteration function:

def fun_iteration(eMOGA):

    if 'Nit' in eMOGA.keys():

        if np.mod(eMOGA['gen_counter'], eMOGA['Nit']) == 0:

            eMOGA['time_Nit_gen'].append(time.perf_counter() - eMOGA['t0'])       
            eMOGA['Nind_A_Nit_gen'].append(eMOGA['Nind_A'])

            if len(eMOGA['time_Nit_gen']) > 1:
                elapsed_time = eMOGA['time_Nit_gen'][-1] - eMOGA['time_Nit_gen'][-2]
            else:
                elapsed_time = eMOGA['time_Nit_gen'][-1]

            print(f"Generations: {eMOGA['gen_counter']} out of {eMOGA['Generations']}. Nind_A = {eMOGA['Nind_A']} (elapsed time: {elapsed_time:.1f} s)")

            if eMOGA['save_results']:
                savemat(eMOGA['mat_file'], eMOGA)
    else:
        print(f"Generations: {eMOGA['gen_counter']} out of {eMOGA['Generations']}. Nind_A = {eMOGA['Nind_A']}")        

    return(eMOGA)

# ---------------------------------------------------------------
# Results function:

def fun_results(eMOGA):
    """
    This function is called at the end of the optimization process.
    
    """
    
    if eMOGA['save_results']:
        savemat(eMOGA['mat_file'], eMOGA)
    
    return(eMOGA)

# ---------------------------------------------------------------
# The end
