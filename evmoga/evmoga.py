import numpy as np # type: ignore
from tqdm import tqdm

# -------------------------------------------------------------------

def initialize(eMOGA):

    print('Initializing ev-MOGA algorithm...\n')

    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Setting parameters

    if 'Nind_P' not in eMOGA.keys():
        eMOGA['Nind_P'] = 1000

    if 'n_div' not in eMOGA.keys():
        eMOGA['n_div'] = 100

    if 'Pm' not in eMOGA.keys():
        eMOGA['Pm'] = 0.2

    if 'precision_onoff' not in eMOGA.keys():
        eMOGA['precision_onoff'] = False
    elif 'precision' not in eMOGA.keys():
        eMOGA['precision'] = 0.0001

    if 'Sigma_Pm_ini' not in eMOGA.keys():
        eMOGA['Sigma_Pm_ini'] = 20.0
    if 'Sigma_Pm_fin' not in eMOGA.keys():
        eMOGA['Sigma_Pm_fin'] = 0.1
    if 'dd_ini' not in eMOGA.keys():
        eMOGA['dd_ini'] = 0.25
    if 'dd_fin' not in eMOGA.keys():   
        eMOGA['dd_fin'] = 0.10

    if 'Generations' not in eMOGA.keys():
        eMOGA['Generations'] = 100
        print(f"### Default value assigned: Generations = {eMOGA['Generations']}.")

    if eMOGA['Generations'] < 0:
        eMOGA['Generations'] = 0
        print("### Number of generations should be non negative. Value set to 0.")

    if eMOGA['Generations'] == 1:
        eMOGA['dd_fin_aux'] = np.inf()
        eMOGA['Sigma_Pm_fin_aux'] = np.inf()
    else:
        eMOGA['dd_fin_aux'] = ((eMOGA['dd_ini'] / eMOGA['dd_fin'])**2 - 1) / (eMOGA['Generations'] - 1)
        eMOGA['Sigma_Pm_fin_aux'] = ((eMOGA['Sigma_Pm_ini'] / eMOGA['Sigma_Pm_fin'])**2 - 1) / (eMOGA['Generations'] - 1)

    if 'randseed' not in eMOGA.keys(): # AH: Comprovar i repassar
        np.random.seed()
        auxseed = np.random.get_state()
        eMOGA['randseed'] = auxseed[1][0]
    else:
        np.random.seed(np.abs(np.int32(eMOGA['randseed'])))

    if not 'random_update_P' in eMOGA.keys():
        eMOGA['random_update_P'] = False
    elif not isinstance(eMOGA['random_update_P'], bool):
        eMOGA['random_update_P'] = False
    
    if eMOGA['random_update_P']:
        if not 'tries' in eMOGA.keys():
            eMOGA['tries'] = int(np.round(0.2 * eMOGA['Nind_P']))
        print(f"### Only {eMOGA['tries']} solutions of P will be checked for update.")

    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

    eMOGA['gen_counter'] = 0

    # ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    # Working np.arrays

    # eMOGA['ele_P'] = np.empty((0, eMOGA['searchSpace_dim']))
    eMOGA['coste_P'] = np.empty((0, eMOGA['objfun_dim']))
    eMOGA['rndidxP'] = np.random.permutation(eMOGA['Nind_P'])

    eMOGA['mod'] = np.zeros(eMOGA['Nind_P'])
    """
    eMOGA['mod'][i]
        - is 0 if not yet inspected,
        - is 1 if the solution 'i' is dominated
        - is 2 if the solution 'i' is non-dominated
    """

    eMOGA['ele_A'] = np.empty((0, eMOGA['searchSpace_dim']))
    eMOGA['coste_A'] = np.empty((0, eMOGA['objfun_dim']))
    eMOGA['box_A'] = np.empty((0, eMOGA['objfun_dim']))
    eMOGA['Nind_A'] = 0

    if 'Nind_GA' not in eMOGA.keys():
        eMOGA['Nind_GA'] = np.int32(0.1 * eMOGA['Nind_P'])
    if np.mod(eMOGA['Nind_GA'], 2) != 0:
        eMOGA['Nind_GA'] = np.ceil(eMOGA['Nind_GA'] / 2) * 2

    eMOGA['ele_GA'] = np.empty((eMOGA['Nind_GA'], eMOGA['searchSpace_dim']))
    eMOGA['coste_GA'] = np.empty((eMOGA['Nind_GA'], eMOGA['objfun_dim']))
    eMOGA['box_GA'] = np.empty((eMOGA['Nind_GA'], eMOGA['objfun_dim']))

    return eMOGA

# -------------------------------------------------------------------

def create_solution(eMOGA):
    
    x = eMOGA['searchspaceLB'] + (eMOGA['searchspaceUB'] - eMOGA['searchspaceLB']) * np.random.rand(eMOGA['searchSpace_dim'])

    if 'constraintfun' in eMOGA.keys() and eMOGA['constraintfun'] is not None:
        x = eMOGA['constraintfun'](x, eMOGA)

    return x
    

# -------------------------------------------------------------------

def generate_P0(eMOGA):

    if 'ele_P' in eMOGA.keys():
        Nind_P = eMOGA['ele_P'].shape[0]
    else:
        Nind_P = 0
        print(f"No initial population P(0) found")

    if Nind_P >= eMOGA['Nind_P']:
        print(f"Initial population P(0) with {Nind_P} individuals found")
        print("No new solutions needed")
        eMOGA['Nind_P'] = Nind_P
    else: 
        if Nind_P > 0:
            print(f"Initial population P(0) with {Nind_P} individuals found")
            print(f"Generating the {eMOGA['Nind_P'] - Nind_P} remaining individuals to complete initial population P(0)")
        else:
            print(f"Generating initial population P(0) with {eMOGA['Nind_P']} individuals")
            eMOGA['ele_P'] = np.empty((0, eMOGA['searchSpace_dim']))

        for i in tqdm(range(eMOGA['Nind_P'] - Nind_P), bar_format='\t{l_bar}{bar}|'):
            x = create_solution(eMOGA)
            eMOGA['ele_P'] = np.vstack((eMOGA['ele_P'], x))
    
        print(f"\nFinal size of P(0): {eMOGA['ele_P'].shape[0]} individuals")

    return eMOGA

# -------------------------------------------------------------------

def MutRealDGausPond2(eMOGA, i):

    padre = eMOGA['ele_GA'][i].copy()
    sigma = eMOGA['Sigma_Pm_ini'] / np.sqrt(1 + eMOGA['gen_counter'] * eMOGA['Sigma_Pm_fin_aux'])

    hijo = padre
    '''
    index = np.random.permutation(eMOGA['searchSpace_dim'])
    for j in range(eMOGA['searchSpace_dim']):
        #genanchu[index[j]] += gauss(1, 1, 0, (eMOGA['searchspaceUB'][index[j]] - eMOGA['searchspaceLB'][index[j]]) * sigma / 100.0)
        genanchu[index[j]] += (eMOGA['searchspaceUB'][index[j]] - eMOGA['searchspaceLB'][index[j]]) * sigma / 100.0 * np.random.randn()
    '''
    # Mutació en una sola ordre:
    hijo += (np.array(eMOGA['searchspaceUB']) - np.array(eMOGA['searchspaceLB'])) * sigma / 100.0 * np.random.randn(eMOGA['searchSpace_dim'])
    
    if 'constraintfun' in eMOGA.keys() and eMOGA['constraintfun'] is not None:
        hijo = eMOGA['constraintfun'](hijo, eMOGA)

    eMOGA['ele_GA'][i] = hijo.copy()

    return eMOGA

# -------------------------------------------------------------------

def Lxov(eMOGA, i1, i2):
    
    padre1 = eMOGA['ele_GA'][i1].copy()
    padre2 = eMOGA['ele_GA'][i2].copy()

    # Factor que se va reduciendo al aumentar el número de generaciomes
    # Produce cruces con mayor "cambio" al principio
    dd = eMOGA['dd_ini'] / np.sqrt(1 + eMOGA['gen_counter'] * eMOGA['dd_fin_aux'])
    
    beta = (1.0 + 2.0 * dd) * np.random.rand() - dd
    hijo1 = beta * padre1 + (1.0 - beta) * padre2
    hijo2 = (1.0 - beta) * padre1 + beta * padre2
        
    # Aplicamos restriccciones definidas por el usuario
    if 'constraintfun' in eMOGA.keys() and eMOGA['constraintfun'] is not None:
        hijo1 = eMOGA['constraintfun'](hijo1, eMOGA)
        hijo2 = eMOGA['constraintfun'](hijo2, eMOGA)

    eMOGA['ele_GA'][i1] = hijo1.copy()
    eMOGA['ele_GA'][i2] = hijo2.copy()

    return eMOGA

# -------------------------------------------------------------------

def fill_GA(eMOGA):
    """
    Generates Auxiliary Matrix 'GA' and evaluates the objetive function for each new solution

    """

    for i in range(0, eMOGA['Nind_GA'], 2):

        a = np.random.randint(eMOGA['Nind_P'])
        eMOGA['ele_GA'][i] = eMOGA['ele_P'][a].copy() # From P
        
        b = np.random.randint(eMOGA['Nind_A'])
        eMOGA['ele_GA'][i+1] = eMOGA['ele_A'][b].copy() # From A

        pm = np.random.rand()

        if pm > eMOGA['Pm']:
            #print('\nCreuament')
            eMOGA = Lxov(eMOGA, i, i+1)
        else:
            #print('\nMutacions')
            eMOGA = MutRealDGausPond2(eMOGA, i)
            eMOGA = MutRealDGausPond2(eMOGA, i+1)


    for i, g in enumerate(eMOGA['ele_GA']):
        eMOGA['coste_GA'][i] = eMOGA['objfun'](g, eMOGA['param'])

    return eMOGA

# -------------------------------------------------------------------

def dominance(f: np.ndarray, g: np.ndarray) -> int:
    """
    Checks the dominance relationship between f and g.

    Dominance relationship:
        - Returns 0 if neither dominates the other.
        - Returns 1 if f dominates g.
        - Returns 2 if g dominates f.
        - Returns 3 if g and f are the same.
    
    Args:
        f (np.ndarray): solution 1.
        g (np.ndarray): solution 2.
    
    Returns:
        int: The dominance relationship between f an g.
    """
    a, b = 0, 0
    for fi, gi in zip(f, g):
        if fi < gi:
            a = 1
        elif fi > gi:
            b = 1
    return  3 - a * 2 - b

# -------------------------------------------------------------------

def box_dominance(box_f: np.ndarray, box_g: np.ndarray) -> int:
    """
    Checks the e-dominance relationship between box_f and box_g.

    Dominance relationship:
        - Returns 0 if neither dominates the other.
        - Returns 1 if box_f dominates box_g.
        - Returns 2 if box_g dominates box_f.
        - Returns 3 if box_g and box_f are the same.
    
    Args:
        box_f (np.ndarray): box 1.
        box_g (np.ndarray): box 2.
    
    Returns:
        int: The e-dominance relationship between box_f an box_g.
    """
    a, b = 0, 0
    for box_fi, box_gi in zip(box_f, box_g):
        if box_fi < box_gi:
            a = 1
        elif box_fi > box_gi:
            b = 1
    return  3 - a * 2 - b

# -------------------------------------------------------------------

def calculate_box(eMOGA, cost):
    box = np.int16(np.zeros(eMOGA['objfun_dim']))
    idx = np.where(eMOGA['epsilon'] > 0.0)[0]
    box[idx] = np.ceil(np.round(1e6 * (cost[idx] - eMOGA['min_f'][idx]) / eMOGA['epsilon'][idx]) / 1e6)
    # ATENCIÓ a la forma de fer el ceil...
    return np.int16(box)

# -------------------------------------------------------------------

def archive(eMOGA, x, cost, box):
    j = 0    
    while j <= eMOGA['Nind_A'] - 1:
        a = box_dominance(box, eMOGA['box_A'][j])
        #match a:
        if a == 0: # Nothing to do 
            pass
        elif a == 2: # Can't archive in A because it is box-dominated                
            return eMOGA
        elif a == 3: # They are in the same box, but one of them have to be removed
            b = dominance(cost, eMOGA['coste_A'][j])
            match b:
                case 0: # The one nearest of the coordinates of the box is matained
                    dist1 = np.linalg.norm((cost - ((box - 0.5) * eMOGA['epsilon'] + eMOGA['min_f'])) / eMOGA['epsilon'], ord=2)
                    dist2 = np.linalg.norm((eMOGA['coste_A'][j] - ((box - 0.5) * eMOGA['epsilon'] + eMOGA['min_f'])) / eMOGA['epsilon'], ord=2)
                    if dist1 < dist2:
                        eMOGA['ele_A'][j] = x.copy()
                        eMOGA['coste_A'][j] = cost.copy()
                    return eMOGA
                case 1: # g substitutes f
                    eMOGA['ele_A'][j] = x.copy()
                    eMOGA['coste_A'][j] = cost.copy()
                    return eMOGA
                case 2 | 3: # I They are identical or the one in A dominates the new one, then one in A is mantained
                    return eMOGA
        else: # The new one box-dominates an individual of A, then, this individual is removed
            if j < eMOGA['Nind_A']: #If it is in the last position there is nothing to move, only modifying Nind_A
                eMOGA['ele_A'] = np.delete(eMOGA['ele_A'], j, axis=0)
                eMOGA['box_A'] = np.delete(eMOGA['box_A'], j, axis=0)
                eMOGA['coste_A'] = np.delete(eMOGA['coste_A'], j, axis=0)
            j -= 1
            eMOGA['Nind_A'] -= 1
        j += 1

    # Adding at the end

    if eMOGA['ele_A'].shape[0] <= eMOGA['Nind_A']:
        eMOGA['ele_A'] = np.vstack((eMOGA['ele_A'], x))
    else:
        eMOGA['ele_A'][eMOGA['Nind_A']] = x.copy()

    if eMOGA['coste_A'].shape[0] <= eMOGA['Nind_A']:
        eMOGA['coste_A'] = np.vstack((eMOGA['coste_A'], cost))
    else:
        eMOGA['coste_A'][eMOGA['Nind_A']] = cost.copy()

    if eMOGA['box_A'].shape[0] <= eMOGA['Nind_A']:
        eMOGA['box_A'] = np.vstack((eMOGA['box_A'], box))
    else:
        eMOGA['box_A'][eMOGA['Nind_A']] = box.copy()

    eMOGA['Nind_A'] += 1

    return eMOGA


# -------------------------------------------------------------------

def evalua_objetivos_poblacion_inicial(eMOGA):
    
    print('\nEstimating value functions of P(0)')
    
    coste_P = np.zeros((eMOGA['Nind_P'], eMOGA['objfun_dim']))
    for i, x in enumerate(eMOGA['ele_P']):
        coste_P[i] = eMOGA['objfun'](x, eMOGA['param'])
    eMOGA['coste_P'] = coste_P
    return eMOGA

# -------------------------------------------------------------------

def generate_A0(eMOGA):

    print('\nEstimating the initial e-Pareto front A(0):')

    # Mark all the solutions in P:
    print('      - Marking non-dominated solutions...')

    # mod:
    #   - is 1 if the solution 'i' is dominated
    #   - is 2 if the solution 'i' is non-dominated
    mod = np.int8(2 * np.ones(eMOGA['Nind_P'])) # Initialized as non-dominated
    for i in range(len(mod)):
        if mod[i] == 2: # If it is not marked as dominated
            dominated = np.all(eMOGA['coste_P'] > eMOGA['coste_P'][i], axis=1)
            idx = np.where(dominated)[0]
            if len(idx) > 0:
                mod[idx] = int(1)

    # Non-dominated solutions:
    coste_ND = eMOGA['coste_P'][(mod == 2)]

    # All non-dominated individuals have been marked
    # Next steps are: obtaining bounds, grid and inserting non-dominated if required
    eMOGA['max_f'] = np.max(coste_ND, axis=0)
    eMOGA['min_f'] = np.min(coste_ND, axis=0)
    eMOGA['epsilon'] = (eMOGA['max_f'] - eMOGA['min_f']) / eMOGA['n_div']

    print('      - Selecting epsilon-non-dominated solutions...')
    for i, _ in enumerate(tqdm(coste_ND, bar_format='\t{l_bar}{bar}|')):
        box = calculate_box(eMOGA, coste_ND[i])
        eMOGA = archive(eMOGA, eMOGA['ele_P'][i], coste_ND[i], box)
    
    # Only for debugging purposes:
    eMOGA['coste_NDA0'] = coste_ND # coste_A0 non-epsilon

    return eMOGA

# -------------------------------------------------------------------

def iteracion_MOEA(eMOGA):
    """
    1. Obtain the zone wher lays every solution in GA
    2. Depending on the zone calculated:
        Z1: The new solution is within the hypercube containing A, then we must check the dominance 
        Z2: The new solution dominates all the solutions in A. 
        Z3: The new solution is clearly dominated by all the solutions in A
        Z4: Otherwise, and we must check the dominance with respect to A
    
    Strategy:

        With the solutions of GA located in Z3, do nothing
    
        If there are any solution of GA located in Z2:
            Recalculate the new matrix A from solutions of GA located in Z2
            Obtain the new solutions of GA located in Z4
        
        Else, if there were no solution of GA located in Z2
            Evaluate solutions of GA located in Z1

        Eavaluate solutions of GA located in Z4        

    """

    a = np.sum(eMOGA['coste_GA'] >= eMOGA['max_f'], axis=1)
    b = np.sum(eMOGA['coste_GA'] <= eMOGA['min_f'], axis=1)

    Z1 = np.where(np.logical_and(a==0, b==0))[0]
    Z2 = np.where(np.logical_and(a==0, b==eMOGA['objfun_dim']))[0] # Complete substitution of matrix A
    Z3 = np.where(np.logical_and(a==eMOGA['objfun_dim'], b==0))[0] # Nothing to do with them
    Z4 = np.setdiff1d(np.array([i for i in range(eMOGA['Nind_GA'])]), np.concatenate((Z1, Z2, Z3), axis=0), assume_unique=True)

    # -----------------------------------------------------------------------
    # If there are any solution of GA located in Z2:

    if len(Z2) > 0: # There are solution in GA that dominate all solutions in A

        # 1. Recalculate the new matrix A from solutions of GA located in Z2:
        #   a. Obtain the non-dominated solutions
        #   b. Recalculate the limits of the new matrix A
        #   c. Obtain boxes
        #   d. Archive solutions
        #
        # 2. Obtain the new solutions of GA located in Z4

        # -----------------------------------------------------------------------
        # 1. Recalculate the new matrix A from solutions of GA located in Z2:
        
        # Initializing matrix A:
        eMOGA['ele_A'] = np.empty((0, eMOGA['searchSpace_dim']))
        eMOGA['coste_A'] = np.empty((0, eMOGA['objfun_dim']))
        eMOGA['box_A'] = np.empty((0, eMOGA['objfun_dim']))
        eMOGA['Nind_A'] = 0

        # mod:
        #   - is 1 if the solution 'i' is dominated
        #   - is 2 if the solution 'i' is non-dominated
        mod = np.int8(2 * np.ones(Z2.shape)) # Initialized as non-dominated
        for i in range(len(mod)):
            if mod[i] == 2: # If it is not marked as dominated
                dominated = np.all(eMOGA['coste_GA'][Z2] > eMOGA['coste_GA'][Z2[i]], axis=1)
                idx = np.where(dominated)[0]
                if len(idx) > 0:
                    mod[idx] = int(1)

        # Non-dominated solutions:
        coste_Z2_ND = eMOGA['coste_GA'][Z2[(mod == 2)]]
        ele_Z2_ND = eMOGA['ele_GA'][Z2[(mod == 2)]]

        # All non-dominated individuals in Z2 have been marked
        # Next steps are: obtaining bounds, grid and inserting non-dominated if required
        eMOGA['max_f'] = np.max(coste_Z2_ND, axis=0)
        eMOGA['min_f'] = np.min(coste_Z2_ND, axis=0)
        eMOGA['epsilon'] = (eMOGA['max_f'] - eMOGA['min_f']) / eMOGA['n_div']

        # Selecting epsilon-non-dominated solutions...
        for i, _ in enumerate(coste_Z2_ND):
            box = calculate_box(eMOGA, coste_Z2_ND[i])
            eMOGA = archive(eMOGA, ele_Z2_ND[i], coste_Z2_ND[i], box)

        # -----------------------------------------------------------------------
        # 2. Obtain the new solutions of GA located in Z4:

        a = np.sum(eMOGA['coste_GA'] >= eMOGA['max_f'], axis=1)
        b = np.sum(eMOGA['coste_GA'] <= eMOGA['min_f'], axis=1)

        Z1 = np.where(np.logical_and(a==0, b==0))[0] # AH: No caldria, passen a Z3
        Z2 = np.where(np.logical_and(a==0, b==eMOGA['objfun_dim']))[0] # AH: Ja estan avaluades
        Z3 = np.where(np.logical_and(a==eMOGA['objfun_dim'], b==0))[0]
        Z4 = np.setdiff1d(np.array([i for i in range(eMOGA['Nind_GA'])]), np.concatenate((Z1, Z2, Z3), axis=0), assume_unique=True)

    # -----------------------------------------------------------------------
    # Else, if there were no solution of GA located in Z2

    else:
        # Evaluate solutions of GA located in Z1

        for i in Z1: # Bounds of A are unchanged and checking if it is e-non-dominated
            box = calculate_box(eMOGA, eMOGA['coste_GA'][i])
            eMOGA = archive(eMOGA, eMOGA['ele_GA'][i], eMOGA['coste_GA'][i], box)

    # -----------------------------------------------------------------------
    # Anyway, evaluate solutions of GA located in Z4

    # Filtering nondominated soltutions in Z4. Deleting dominated solutions
    # AH: fa falta en realitat...??? Podríem passar directament al següent proceiment
    to_remove = list()
    for i, iz4 in enumerate(Z4): 
        dominate = np.all(eMOGA['coste_A'] <= eMOGA['coste_GA'][iz4], axis=1)
        idx = np.argmax(dominate)
        if dominate[idx]:
            to_remove.append(i)

    Z4 = np.delete(Z4, to_remove)

    # Working with non-dominated solutions in Z4
    for i in Z4: 

        # Deleting the solutions in A dominated by solution in Z4
        dominated = np.all(eMOGA['coste_A'] > eMOGA['coste_GA'][i], axis=1)
        idx = np.where(dominated)[0]
        eMOGA['ele_A'] = np.delete(eMOGA['ele_A'], idx, axis=0)
        eMOGA['box_A'] = np.delete(eMOGA['box_A'], idx, axis=0)
        eMOGA['coste_A'] = np.delete(eMOGA['coste_A'], idx, axis=0)
        eMOGA['Nind_A'] -= len(idx)

        # Updating bounds
        eMOGA['max_f'] = np.max(np.concatenate([eMOGA['coste_A'], np.array([eMOGA['coste_GA'][i]])], axis=0), axis = 0)
        eMOGA['min_f'] = np.min(np.concatenate([eMOGA['coste_A'], np.array([eMOGA['coste_GA'][i]])], axis=0), axis = 0)
        # Obtaining epsilon
        eMOGA['epsilon'] = (eMOGA['max_f'] - eMOGA['min_f']) / eMOGA['n_div']

        # Recalculating boxes of A
        for j, _ in enumerate(eMOGA['coste_A']): # Recalculating boxes of A
            eMOGA['box_A'][j] = calculate_box(eMOGA, eMOGA['coste_A'][j])

        # Updating A
        Nind_A_temp = eMOGA['Nind_A']
        copyele = eMOGA['ele_A'].copy()
        copycos = eMOGA['coste_A'].copy()
        copybox = eMOGA['box_A'].copy()
        eMOGA['Nind_A'] = 1
        for j in range(1, Nind_A_temp):
            eMOGA = archive(eMOGA, copyele[j], copycos[j], copybox[j])

        # Storing in A the individual checked
        box = calculate_box(eMOGA, eMOGA['coste_GA'][i])
        eMOGA = archive(eMOGA, eMOGA['ele_GA'][i], eMOGA['coste_GA'][i], box)

    return eMOGA

# -------------------------------------------------------------------

def update_P(eMOGA):
    
    '''    
    for i in range(eMOGA['Nind_GA']):
        dominats = np.all(eMOGA['coste_P'][eMOGA['rndidxP']] >= eMOGA['coste_GA'][i], axis=1)    
        idx = np.argmax(dominats) # np.argmax(condicio) s'atura en trobar la primera coincidència
        if dominats[idx]:
            eMOGA['ele_P'][eMOGA['rndidxP'][idx]] = eMOGA['ele_GA'][i].copy()
            eMOGA['coste_P'][eMOGA['rndidxP'][idx]] = eMOGA['coste_GA'][i].copy()
    '''
    
    coste_P_rand = eMOGA['coste_P'][eMOGA['rndidxP']]

    for xga, zga in zip(eMOGA['ele_GA'], eMOGA['coste_GA']):
        dominats = np.all(coste_P_rand >= zga, axis=1)    
        idx = np.argmax(dominats) # np.argmax(condicio) s'atura en trobar la primera coincidència
        if dominats[idx]:
            eMOGA['ele_P'][eMOGA['rndidxP'][idx]] = xga
            eMOGA['coste_P'][eMOGA['rndidxP'][idx]] = zga

    return eMOGA

# -------------------------------------------------------------------

def MOGA(eMOGA):

    print('\n--------------------------------------------------------\n')

    eMOGA = initialize(eMOGA)
    eMOGA = generate_P0(eMOGA)
    eMOGA = evalua_objetivos_poblacion_inicial(eMOGA)
    eMOGA = generate_A0(eMOGA)

    print('\nIterating process searching for e-Pareto front...\n')

    if 'iterationfun' in eMOGA.keys(): # User iteration function
            eMOGA = eMOGA['iterationfun'](eMOGA)

    # -------------------------------------------------
        
    while eMOGA['gen_counter'] < eMOGA['Generations']:
        
        # k \in [0, N - 1]; 
        # N = eMOGA['Generations']
        # k = eMOGA['gen_counter']

        eMOGA = fill_GA(eMOGA)
        eMOGA = iteracion_MOEA(eMOGA)
        eMOGA = update_P(eMOGA)

        eMOGA['gen_counter'] += 1

        if 'iterationfun' in eMOGA.keys(): # User iteration function
            eMOGA = eMOGA['iterationfun'](eMOGA)

    if 'resultsfun' in eMOGA.keys(): # User final results function
        eMOGA = eMOGA['resultsfun'](eMOGA)

    return eMOGA

# -------------------------------------------------------------------
