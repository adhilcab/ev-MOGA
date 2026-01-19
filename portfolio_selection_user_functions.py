import numpy as np 
from scipy.io import savemat

import time
from tqdm import tqdm

# ---------------------------------------------------------------
# Objective function (2obj)

def objective_function_2obj(x, param): # Markowitz mean-variance model

    z1 = np.dot(param['mean_r'], x)
    z2 = np.dot(x, np.dot(param['cov_Mtrx'], x))

    return np.array([z1, z2]) * np.array(param['signs'])

# ---------------------------------------------------------------
# Objective function (3obj)

def objective_function_3obj(x, param): # Markowitz + ESG

    signs = np.array(param['signs'])

    z1 = np.dot(param['mean_r'], x) * signs[0]
    z2 = np.dot(x, np.dot(param['cov_Mtrx'], x)) * signs[1]
    z3 = np.dot(param['esg'], x) * signs[2]

    z = np.array([z1, z2, z3])

    if 'Md_1' in param:
        Md_1 = param['Md_1']
        z = np.dot(Md_1, z)

    return z

# ---------------------------------------------------------------
# Iteration function (2obj):

def fun_iteration_2obj(eMOGA):

    """
    This function is called at the end of each generation.
    It is used, for example, to store the best solutions found so far.

    """

    if np.mod(eMOGA['gen_counter'], 10) == 0:

        print(f"Generations: {eMOGA['gen_counter']} out of {eMOGA['Generations']}. Nind_A = {eMOGA['Nind_A']}")
    
    return(eMOGA)

# ---------------------------------------------------------------
# Iteration function (3obj):

def fun_iteration(eMOGA):

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
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# User constraints:

def discretize(x, precision, LB, UB):

    y = np.round(x / precision) * precision
    rest = np.round((1.0 - np.sum(y)) / precision) * precision

    if np.abs(rest) > precision / 2:
        idxsort = np.argsort(y)[::-1] # Sort indices in descending order     
        idxcond = np.where((y[idxsort] + rest <= UB) & (y[idxsort] + rest >= LB))[0]
        y[idxsort[idxcond[0]]] += rest

    return y

# -------------------------------------------------------------------

def sat(x, LB, UB):

    y = x.copy()
    y = np.clip(y, 0.0, UB)
    idx = np.where((y < LB) & (y > 0.0))[0]
    if len(idx) > 0:
        for i in idx:
            y[i] = LB[i] if y[i] > LB[i] / 2 else 0.0

    return y

# -------------------------------------------------------------------

def sum1_increasing(x, LB, UB):

    y = x.astype(float).copy()

    idx = np.where(y > 0)[0] # Is there any non-zero element?
    from_zero = True if len(idx) == 0 else False

    while True:

        if from_zero:
            idx = range(len(y))
        else:
            idx = np.where(y > 0)[0]
            if np.sum(UB[idx]) < 1.0: # Si la suma dels límits superiors és menor que 1.0 no podem assolir la suma 1 dels actius.
                jdx = np.setdiff1d(np.arange(len(y)), idx)
                if len(jdx) == 0:
                    raise ValueError("Cannot achieve the sum of 1 with the given bounds.")
                jdx = np.random.choice(jdx, len(jdx), replace=False)
            while np.sum(UB[idx]) < 1.0: # Cal afegir nous actius fins que la suma dels límits superiors sigui >= 1.0
                if np.sum(LB[idx]) + LB[jdx[0]] > 1.0:
                    # Si la suma dels límits inferiors incloent el de l'actiu a afegir és major que 1.0:
                    if len(jdx) == 1: # Si només en queda un per afegir i no ens serveix, error:
                        raise ValueError("Cannot achieve the sum of 1 with the given bounds.")
                    else: # L'actual no ens serveix però en queden més per provar:
                        jdx = np.delete(jdx, 0)
                        continue
                else: # Podem afegir l'actiu jdx[0] a la solució
                    y[jdx[0]] = LB[jdx[0]]
                    idx = np.append(idx, jdx[0])
                    jdx = np.delete(jdx, 0)
                    if np.sum(y[idx]) > 1.0: # La suma és major que 1.0 en afegir el nou actiu, els posem tots a LB
                        y[idx] = LB[idx]


        current_sum = float(y.sum())
        remaining_diff = np.abs(1.0 - current_sum)
                
        idx = np.random.choice(idx, len(idx), replace=False)
        capacity = UB[idx] - y[idx]

        for j in range(len(idx) - 1):
            if remaining_diff <= 1e-6:
                return y
            delta = np.random.rand() * np.min([remaining_diff, capacity[j]])
            y[idx[j]] += delta
            if y[idx[j]] < LB[idx[j]]: # Ensure ytemp is either 0.0 or >= LB
                if (y[idx[j]] > LB[idx[j]] / 2) and (LB[idx[j]] <= remaining_diff):
                    y[idx[j]] = LB[idx[j]]
                else:
                    y[idx[j]] = 0.0
                delta = np.abs(y[idx[j]]) # Update delta to the value of y[idx[j]]
            current_sum += delta
            remaining_diff = np.abs(1.0 - current_sum)

        if (capacity[-1] >= remaining_diff):
            if (remaining_diff + y[idx[-1]] > LB[idx[-1]]): # We found a feasible solution
                y[idx[-1]] += remaining_diff
                break
        from_zero = False if np.sum(y) > 0.0 else True

    return y

# -------------------------------------------------------------------

def sum1_decreasing(x, LB, UB):

    y = x.astype(float).copy()
    current_sum = float(y.sum())

    # El primer que hi ha que fer és eliminar actius fins que sum(LBi) <= 1.0
    # L'estratègia serà eliminar els actius amb valors més baixos
    idx = np.where(y > 0)[0]
    idx = idx[np.argsort(y[idx])]  # Sort idx according to the sorted values of y[idx]
    tol = 1e-6 if np.random.rand() < 0.5 else 0.0 # Randomly choose if we allow sum(LBi) to be strictly greater than 1.0 or not
    while np.sum(LB[idx]) > 1.0 - tol:
        y[idx[0]] = 0.0
        idx = np.delete(idx, 0)
        if np.sum(y[idx]) >= 1.0:
            continue
        else:
            y = sum1_increasing(y, LB, UB)
            return y
        
    current_sum = float(y.sum())

    while True:
        remaining_diff = np.abs(1.0 - current_sum)
        if remaining_diff <= 1e-6:
            break
        idx = np.where(y > 0.0)[0]
        idx = np.random.choice(idx, len(idx), replace=False)
        capacity = y[idx] - LB[idx]

        for j in range(len(idx) - 1):
            delta = np.random.rand() * np.min([remaining_diff, capacity[j]])
            y[idx[j]] -= delta
            current_sum -= delta
            remaining_diff = np.abs(1.0 - current_sum)

        if capacity[-1] >= remaining_diff: # We found a feasible solution
            y[idx[-1]] -= remaining_diff
            break
        else: # We add/remove a random portion of capacity to/from the last one and perform another round
            delta = np.random.rand() * capacity[-1]
            y[idx[-1]] -= delta
            current_sum -= delta
            continue

    return y

# -------------------------------------------------------------------

def sum1(x, eMOGA):
    """
    Adjusts the input array x so that its elements sum to 1.0, while respecting the bounds defined by LB and UB.
    Elements can be zero, but if they are non-zero, they must be within the specified bounds.
    """
    if x.ndim != 1:
        raise ValueError("The array must be one-dimensional")

    LB = eMOGA['searchspaceLB']
    UB = eMOGA['searchspaceUB']

    y = x.astype(float).copy()
    y = sat(y, LB, UB)
    diff = 1.0 - float(y.sum())

    if diff < 0:
        y = sum1_decreasing(y, LB, UB)        
    elif diff > 0:
        y = sum1_increasing(y, LB, UB)

    if eMOGA['precision_onoff']:
        y = discretize(y, eMOGA['precision'], LB, UB)

    return y

# -------------------------------------------------------------------

def sum1_no_aleat(x, eMOGA):

    LB = eMOGA['searchspaceLB']
    UB = eMOGA['searchspaceUB']

    y = x.astype(float).copy()
    y = sat(y, LB, UB)
    
    return y / np.sum(y)


# -------------------------------------------------------------------

def generate_P0_method(eMOGA):

    print('\n--------------------------------------------------------\n')
    print(f"Generating user-defined initial population P(0) with {eMOGA['Nind_P']} individuals\n")

    eMOGA['ele_P'] = np.empty((0, eMOGA['searchSpace_dim']))

    match eMOGA['Generating_P0_method']:

        case 0:  # 0: zero initialization and aleatory sum1
            print("Method: Zero initialization and aleatory sum1()\n")
            x = np.zeros(eMOGA['searchSpace_dim'])
            for i in tqdm(range(eMOGA['Nind_P'])):
                y = sum1(x, eMOGA)
                eMOGA['ele_P'] = np.vstack((eMOGA['ele_P'], y))

        case 1:  # 1: random initial values and aleatory sum1
            print("Method: Random initialization and aleatory sum1()\n")
            for i in tqdm(range(eMOGA['Nind_P'])):
                x = eMOGA['searchspaceLB'] + (eMOGA['searchspaceUB'] - eMOGA['searchspaceLB']) * np.random.rand(eMOGA['searchSpace_dim'])
                y = sum1(x, eMOGA)
                eMOGA['ele_P'] = np.vstack((eMOGA['ele_P'], y))

        case 2:  # 2: random initial values and non-aleatory sum1 (sum1_no_aleat)
            print("Method: Random initialization and non-aleatory sum1_no_aleat()\n")            
            for i in tqdm(range(eMOGA['Nind_P'])):
                x = eMOGA['searchspaceLB'] + (eMOGA['searchspaceUB'] - eMOGA['searchspaceLB']) * np.random.rand(eMOGA['searchSpace_dim'])
                y = sum1_no_aleat(x, eMOGA)
                eMOGA['ele_P'] = np.vstack((eMOGA['ele_P'], y))

        case _:
            raise ValueError("Unknown Generating_P0_method")

    return eMOGA

# -------------------------------------------------------------------

def generate_P0(eMOGA):

    print('\n--------------------------------------------------------\n')
    print(f"Generating user-defined initial population P(0) with {eMOGA['Nind_P']} individuals\n")

    eMOGA['ele_P'] = np.empty((0, eMOGA['searchSpace_dim']))
    
    x = np.zeros(eMOGA['searchSpace_dim'])

    for i in tqdm(range(eMOGA['Nind_P'])):
        y = sum1(x, eMOGA)
        eMOGA['ele_P'] = np.vstack((eMOGA['ele_P'], y))

    return eMOGA

# -------------------------------------------------------------------
