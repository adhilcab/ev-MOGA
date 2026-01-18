
# ------------------------------------------------------------------

import scipy.io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------

def show_fig(block=False, pause=0.001):
    """
    Displays the figure without blocking execution, and gives the backend GUI time 
    to process events for the window to appear.
    """
    plt.show(block=block)
    if not block:
        plt.pause(pause)

# ------------------------------------------------------------------

def load_evMOGA_OOS_results(nom_fitxer_mat: str, verbose=False) -> dict:

    oos = scipy.io.loadmat(nom_fitxer_mat)

    oos['description'] = oos['description'][0]
    oos['signs'] = oos['signs'][0]

    oos['returns_opt'] = oos['returns_opt'][0]
    oos['esgs_opt'] = oos['esgs_opt'][0]
    oos['returns_sharpe'] = oos['returns_sharpe'][0]
    oos['esgs_sharpe'] = oos['esgs_sharpe'][0]

    if verbose:
        print('Ouput is a dictionary with fields:')
        print([k for k in oos.keys() if '__' not in k])

        print(f"Description: '{oos['description']}'")
        print(f"\nObjectives: '{oos['objectives']}'")
        print(f"\nSigns: '{oos['signs']}'")

        print("\n'x_opt' contains all portfolios obtained for the 'optimum'")
        print("'z_opt' contains the corresponding objective values")
        print("'returns_opt' contains a vector with the N*HP out-of-sample values for the return of that portfolio")
        print("'esgs_opt' contains a vector with the N*HP out-of-sample values for the ESG Score of that portfolio")

        print("\n'x_sharpe' contains all portfolios obtained for the 'Max. Sharpe Ratio' portfolio")
        print("'z_sharpe' contains the corresponding objective values")
        print("'returns_sharpe' contains a vector with the N*HP out-of-sample values for the return of that portfolio")
        print("'esgs_opt' contains a vector with the N*HP out-of-sample values for the ESG Score of that portfolio")

    return oos

# ------------------------------------------------------------------

def load_evMOGA_results(nom_fitxer_mat: str, mri=False) -> dict:

    eMOGA = scipy.io.loadmat(nom_fitxer_mat)

    # Convert eMOGA['params'] to dictionary
    param_fields = list(eMOGA['param'].dtype.names)
    param = dict()
    for i, field in enumerate(param_fields):
        exec('param["' + field + '"] = ' + f'eMOGA["param"][0,0][{i}]')
    eMOGA['param'] = param

    # Scalar values are stored as 2D arrays, convert them to scalars
    eMOGA['Nind_A'] = eMOGA['Nind_A'][0,0]
    eMOGA['Generations'] = eMOGA['Generations'][0,0]
    eMOGA['gen_counter'] = eMOGA['gen_counter'][0,0]

    if mri:
        print(nom_fitxer_mat, '\n')
        print(f"Se han analizado {eMOGA['gen_counter']} de {eMOGA['Generations']} generaciones planificadas")
        print(f"La frontera contiene {eMOGA['coste_A'].shape[0]} solucions")
        print(f"Nind_A = {eMOGA['Nind_A']} individuos",)
        print(eMOGA)    

    return eMOGA

# ------------------------------------------------------------------

def load_eMOGA(mat_file_name: str, verbose=True) -> dict:

    eMOGA = scipy.io.loadmat(mat_file_name)

    # Convert eMOGA['params'] to dictionary
    if 'param' in eMOGA:
        param = dict()
        param_fields = list(eMOGA['param'].dtype.names)
        for i, field in enumerate(param_fields):
            param[field] = eMOGA['param'][field][0,0]
            if param[field].shape[0] == 1:
                param[field] = param[field][0]
        eMOGA['param'] = param

    # Convert eMOGA['profiles'] to dictionary
    if 'profiles' in eMOGA:
        profiles = list()
        for i, p in enumerate(eMOGA['profiles'][0]):
            profile_dict = dict()
            profile_fields = list(p.dtype.names)
            for i, field in enumerate(profile_fields):
                profile_dict[field] = p[field][0,0][0]
                if field == 'idx':
                    profile_dict[field] = profile_dict[field][0]
            profiles.append(profile_dict)
        eMOGA['profiles'] = profiles

    # Convert eMOGA['graph options'] to dictionary
    if 'graph options' in eMOGA:
        graph_options = dict()
        graph_fields = list(eMOGA['graph options'].dtype.names)
        for i, field in enumerate(graph_fields):
            graph_options[field] = eMOGA['graph options'][field][0,0][0]
        eMOGA['graph options'] = graph_options

    if 'time_Nit_gen' in eMOGA:
        eMOGA['total_elapsed_minutes'] = eMOGA['time_Nit_gen'][-1] // 60
        eMOGA['total_elapsed_seconds'] = eMOGA['time_Nit_gen'][-1] % 60

    # Scalar values are stored as 2D arrays, convert them to scalars
    scalars = ['Nind_P', 'Nind_A', 'n_div', 'Generations', 'gen_counter',
            'Pm', 'precision_onoff', 'precision', 'randseed']
    for scalar in scalars:
        if scalar in eMOGA:
            eMOGA[scalar] = eMOGA[scalar][0,0]

    # Srings are stored as 1D arrays, convert them to strings
    strings = ['description']
    for s in strings:
        if s in eMOGA:
            eMOGA[s] = eMOGA[s][0]

    # nd.arrays of shape (1, n) are stored as 2D arrays, convert them to 1D arrays
    arrays = ['objfun_dim', 'signs', 'z_ideal', 'z_nadir', 'color']
    for array in arrays:
        if array in eMOGA:
            eMOGA[array] = eMOGA[array].ravel()
        
    # Print information about the loaded file
    if verbose:
        print(f"\nFile loaded: '{mat_file_name}'")
        print(f"\nDescription: '{eMOGA['description']}'")

        print(f"\nThe Pareto frontier contains {eMOGA['coste_A'].shape[0]} optimal solutions after {eMOGA['gen_counter']} generations (of {eMOGA['Generations']} planned)")
        # print(f"\nNind_A = {eMOGA['Nind_A']} efficient solutions",)

        if 'profiles' in eMOGA:
            print(f"\nNumber of profiles: {len(eMOGA['profiles'])}:\n") 
            for i, profile in enumerate(eMOGA['profiles']):
                for key, value in profile.items():
                    if key.lower() == 'name':
                        print(f"  Profile {i+1}: '{profile[key]}'")
        else:
            print("\nNo profiles defined yet")

        print("\n\n")

    return eMOGA


# ------------------------------------------------------------------

def normalize_frontier(pfront):
    
    return (pfront - pfront.min(axis=0)) / (pfront.max(axis=0) - pfront.min(axis=0))

# ------------------------------------------------------------------

def get_optimum(pfront, preferences=[]):
    
    idx, _, _ = preference_directions(pfront, preferences=preferences)
    return idx[0]

# ------------------------------------------------------------------

def herfindahl(x, cov_Mtrx):
    n_var = cov_Mtrx.shape[0]
    RCR = x * (np.dot(cov_Mtrx, x)) / (np.dot(x, np.dot(cov_Mtrx, x)))
    HI = np.dot(RCR, RCR)
    NHI = (1 - HI) / (1 - 1 / n_var)
    return [HI, NHI]

# ------------------------------------------------------------------

def sharpe(returns, rfr=0, af=1):
    
    return (np.mean(returns) - rfr) / np.std(returns) * af**0.5
    

# ------------------------------------------------------------------

def get_preference_profile_solutions(eMOGA, verbose=True):
        
    n_obj = int(eMOGA['objfun_dim'])

    # Ideal and Nadir
    eMOGA['z_ideal'] = np.min(eMOGA['coste_A'], axis=0) * eMOGA['param']['signs'] # Ideal
    eMOGA['z_nadir'] = np.max(eMOGA['coste_A'], axis=0) * eMOGA['param']['signs'] # Nadir

    idx, d, Md = preference_directions(eMOGA['coste_A'])
    
    # Color for visualization corresponding to the distance to the ideal
    nums = np.arange(len(d), 0, -1) / len(d)
    color = np.zeros(len(d))
    color[idx] = nums**3 # Cubic for better visualization
    eMOGA['color'] = color
    
    if 'profiles' not in eMOGA:
        eMOGA['profiles'] = [{
            'Name': 'Optimum',
            'Method': 'Preference Directions',
            'Label': 'OPT',
            'Mp': get_Mp(n_obj),
            'idx': idx[0],
            'x': eMOGA['ele_A'][idx[0], :],
            'z': eMOGA['coste_A'][idx[0], :] * eMOGA['param']['signs']
        },]
    else:
        if 'optimum' not in [p['Name'].lower() for p in eMOGA['profiles']]:
            eMOGA['profiles'].append({
                'Name': 'Optimum',
                'Method': 'Preference Directions',
                'Label': 'OPT',
                'Mp': get_Mp(n_obj),
            })
        for p in eMOGA['profiles']:
            if 'Label' not in p:
                p['Label'] = p['Name'][:3].upper()
            if 'idx' not in p:
                if p['Method'].lower() == 'preference directions':
                    idx, _, _ = preference_directions(eMOGA['coste_A'], Mp=p['Mp'])
                    p['idx'] = idx[0]
                    p['x'] = eMOGA['ele_A'][idx[0], :]
                    p['z'] = eMOGA['coste_A'][idx[0], :] * eMOGA['param']['signs']

    if verbose:
        
        print(f"\nFrontier: '{eMOGA['description']}'\n")
        print(f"Nind_A = {eMOGA['Nind_A']} (Generations: {eMOGA['gen_counter']} out of {eMOGA['Generations']})")
        
        print(f"\nz_ideal = {np.array2string(eMOGA['z_ideal'], formatter={'float_kind': lambda x: f'{x:.3g}'})}")
        print(f"\nz_nadir = {np.array2string(eMOGA['z_nadir'], formatter={'float_kind': lambda x: f'{x:.3g}'})}")
        
        for p in eMOGA['profiles']:

            print("\n------------------------------------------------")

            print(f"\nProfile: '{p['Name']}'\nLabel: '{p['Label']}'")
            print(f"\nx =\n {np.array2string(p['x'], formatter={'float_kind': lambda x: f'{x:.6g}'})}")
            print(f"\nz = {np.array2string(p['z'], formatter={'float_kind': lambda x: f'{x:.3g}'})}")

    return eMOGA


# ------------------------------------------------------------------

def dominanceCone(Mp: np.matrix) -> np.matrix:
    
    if np.linalg.det(Mp) == 0:
        raise ValueError("The input matrix Mp must be non-singular (det(Mp) != 0). Please redefine your preference directions.")

    n = Mp.shape[0]
    Md = np.zeros(Mp.shape)

    for i in range(n):
        
        Maux = Mp.copy() 
        Maux = np.delete(Mp, i, axis=1) # Remove i column
        v = Mp[:, i].T # Extract i column

        # Building system of linear equation
        for j in range(n): # to avoid det(A)=0
            A = np.concatenate([Maux.T, np.zeros((1, n))], axis=0)
            A[-1, j] = 1
            if np.linalg.det(A) != 0: break
            
        b = np.zeros([n, 1])
        b[-1] = 1

        # Solve system of linear equations Av = b for v
        vaux1 = np.linalg.solve(A, b).T
        
        # Compute inverse vector (opposite direction)
        vaux2 = -vaux1.copy();

        # Checking convenient cone (v*vi>0)
        if (vaux1 * v.T) > (vaux2 * v.T):
            Md[:, i] = (vaux1.T/np.linalg.norm(vaux1)).ravel()
        else:
            Md[:, i] = (vaux2.T/np.linalg.norm(vaux2)).ravel()

    return Md

# ------------------------------------------------------------------

def get_Mp(n_obj, preferences=[]):
    
    if len(preferences) == 0: # No preferences
        preferences = [1 for i in range(n_obj * (n_obj - 1) // 2)]
    elif len(preferences) != n_obj * (n_obj - 1) // 2:
        print('Error: Wrong number of preferences. len(preferencies) must be equal to (n_obj * (n_obj - 1) / 2)')
        return  [-1, -1, -1]

    lower = np.zeros(len(preferences))
    upper = np.zeros(len(preferences))

    for i in range(len(preferences)):
        if preferences[i] < 1:
            lower[i] = 1/preferences[i] - 1
        else:
            upper[i] = preferences[i] - 1

    Mp = np.matrix(np.eye(n_obj))
    lower_tri_indices = np.tril_indices(Mp.shape[0], -1)
    Mp[lower_tri_indices] = -1

    idx = np.where(Mp == 0.0)
    unique_indices = np.ravel_multi_index(idx, Mp.shape)
    np.put(Mp, unique_indices, upper)

    Mp2 = Mp.copy().T
    np.put(Mp2, unique_indices, lower)
    Mp = Mp2.T

    return Mp

# ------------------------------------------------------------------

def preference_directions(pfront, preferences=[], Mp=[]): # pf is an ndarray

    n_obj = pfront.shape[1]
    
    Mp = get_Mp(n_obj, preferences=preferences) if len(Mp) == 0 else Mp
    Md = dominanceCone(Mp)
    Md_1 = np.linalg.inv(Md) 
    G = np.matrix(np.matmul(Md_1.T, Md_1))

    # Normalized Pareto Front
    delta = normalize_frontier(pfront) # d = z_norm - z_norm_ideal = z_norm 

    d = np.zeros([1, delta.shape[0]]) # La distància en la nova base de dominància
    delta = np.matrix(delta)
    for i, delta_i in enumerate(delta):
        delta_i = np.matrix(delta_i)
        d[0, i] = np.sqrt(delta_i * G * delta_i.T) # Redefinició de la distància en la nova base de dominància

    idx = np.argsort(d)
    
    return [idx[0].tolist(), d[0].tolist(), Md]

# ------------------------------------------------------------------

def notacio_cientifica(ax, axis='x'):
    # Configurar el formatter perquè mostri la notació científica a un dels eixos
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))  # Controla quan aplicar notació científica
    if 'x' in axis:
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)
    return

# ------------------------------------------------------------------

def plot_2D_projections(eMOGA, pairings=((1, 0), (2, 0), (1, 2)), 
                        plot_optims=True, cmap='winter'):

    if eMOGA['objfun_dim'] != 3: 
        print("We cannot plot 2D projections of a non-3D Pareto front")
        return

    if 'profiles' not in eMOGA or len(eMOGA['profiles']) == 0:
        eMOGA = get_preference_profile_solutions(eMOGA, verbose=False)

    pfront = eMOGA['coste_A'].copy() * eMOGA['param']['signs']
    objectives = eMOGA['objectives'].copy()
    pfront_norm = normalize_frontier(eMOGA['coste_A'])

    dot_size = 1
    opt_size = dot_size * 16
    color_opt = 'red'

    if 'graph options' in eMOGA:
        graph_options = eMOGA['graph options'].copy()
        if 'dot_size_2D' in graph_options:
            dot_size = graph_options['dot_size_2D']
        if 'opt_dot_size_factor_2D' in graph_options:
            opt_size = dot_size * graph_options['opt_dot_size_factor_2D']
        if 'opt_color' in eMOGA['graph options']:
            color_opt = eMOGA['graph options']['opt_color']

    fontSize_labels = 9

    active_style = plt.rcParams["axes.facecolor"]
    if active_style in ["black", "#000000"]:
        mode_fosc = True
        color_opt_text = 'white'        
    else:
        mode_fosc = False
        color_opt_text = 'black'
    
    d = np.sqrt(np.sum((pfront_norm)**2, axis=1))  # Euclidean distance to the normalized ideal with no preferences
    color = eMOGA['color']

    if 'description' in eMOGA:
        description = eMOGA['description']
    else:
        description = ''

    if len(description) > 0:
        superior_title = description + ' - 2D projections'
    else:
        superior_title = '2D projections'
    
    fig = plt.figure(figsize=(15,4))
    gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.25, hspace=0.3)
    fig.suptitle(superior_title, fontsize=12, y=0.975, va="top")

    z0 = eMOGA['z_ideal']
    znadir = eMOGA['z_nadir']

    for k, parella in enumerate(pairings):

        i, j = parella[0], parella[1]

        ax = fig.add_subplot(gs[k])            
        ax.grid(color=(0.65, 0.65, 0.65), linestyle='--', linewidth=0.25, zorder=0)
        ax.set_xlabel(objectives[i].replace('-','$-$')); ax.set_ylabel(objectives[j].replace('-','$-$'))

        ax.scatter(x=pfront[:, i], y=pfront[:, j], s=dot_size, c=color, edgecolor='None', marker='.', cmap=cmap, zorder=3)
        notacio_cientifica(ax, axis='x')
        notacio_cientifica(ax, axis='y')

        if plot_optims:

            xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
            if k == 1: 
                incx = -0.02 * (xmax - xmin)
                ha = 'right'
            else:
                incx = +0.02 * (xmax - xmin)
                ha = 'left'
            ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
            incy = 0.01 * (ymax - ymin)

            ax.scatter(x=z0[i], y=z0[j], s=opt_size, c='black', edgecolor='None', marker='.', zorder=3)
            ax.text(z0[i] + incx, z0[j], 'Ideal', fontsize=fontSize_labels, color='black', ha=ha, va='center', fontweight='light', zorder=3)

            ax.scatter(x=znadir[i], y=znadir[j], s=.25*opt_size, c='white', edgecolor='black', marker='o', zorder=3)
            ax.text(znadir[i] + incx, znadir[j], 'Nadir', fontsize=fontSize_labels, color='black', ha=ha, va='center', fontweight='light', zorder=3)

            for p in eMOGA['profiles']:
                
                idx = p['idx']
                if 'color' in p: 
                    color_opt_i = p['color']
                else:
                    color_opt_i = color_opt
                
                ax.scatter(x=pfront[idx, i], y=pfront[idx, j], s=opt_size, c='red', edgecolor='None', marker='.', zorder=3)
                ax.text(pfront[idx, i] + np.abs(incx), pfront[idx, j], p['Label'], fontsize=fontSize_labels, color='black', ha='left', va='center', fontweight='light', zorder=3)
                
                if p['Name'].lower() == 'optimum':
                    optims = np.concatenate([np.array([z0]), np.array([pfront[idx, :]])], axis=0)
                    ax.plot(optims[:, i], optims[:, j], lw=0.2, c='black')
                    optims = np.concatenate([np.array([znadir]), np.array([pfront[idx, :]])], axis=0)
                    ax.plot(optims[:, i], optims[:, j], lw=0.2, c='black')

        if 'limits_2D' in graph_options:            
            if 'x_limits' in graph_options['limits_2D']:
                ax.set_xlim(graph_options['limits_2D']['x_limits'][k])
            if 'y_limits' in graph_options['limits_2D']:
                ax.set_ylim(graph_options['limits_2D']['y_limits'][k])

        # print(f"parella: {k}, xlim: {ax.get_xlim():.1e}, ylim: {ax.get_ylim():.1e}")

    show_fig(block=False)

    return fig


# ------------------------------------------------------------------

def plot_3D_Pareto_Front(eMOGA, plot_optims=True, cmap='winter'):

    if eMOGA['objfun_dim'] != 3: 
        print(f"This function only works for 3D Pareto fronts. The number of objectives is {eMOGA['objfun_dim']}")
        return eMOGA

    if 'profiles' not in eMOGA or len(eMOGA['profiles']) == 0:
        eMOGA = get_preference_profile_solutions(eMOGA, verbose=False)

    pfront = eMOGA['coste_A'].copy() * eMOGA['param']['signs']
    objectives = eMOGA['objectives'].copy()
    pfront_norm = normalize_frontier(eMOGA['coste_A'])

    dot_size = 1
    opt_size = dot_size * 16
    color_opt = 'red'

    if 'graph options' in eMOGA:

        graph_options = eMOGA['graph options'].copy()

        if 'dot_size_3D' in graph_options:
            dot_size = graph_options['dot_size_3D']
        if 'opt_dot_size_factor_3D' in graph_options:
            opt_size = dot_size * graph_options['opt_dot_size_factor_3D']
        if 'opt_color' in graph_options:
            color_opt = graph_options['opt_color']
        
        if 'obj_ord_3D' in graph_options:
            obj_ord = graph_options['obj_ord_3D']
        else:
            obj_ord = [0, 1, 2]

    z0 = eMOGA['z_ideal']
    znadir = eMOGA['z_nadir']
    color = eMOGA['color']

    if 'description' in eMOGA:
        description = eMOGA['description']
    else:
        description = ''

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pfront[:, obj_ord[0]], pfront[:, obj_ord[1]], pfront[:, obj_ord[2]], s=dot_size, c=color, edgecolor='None', marker='.', cmap=cmap)

    if plot_optims:
        ax.scatter(z0[obj_ord[0]], z0[obj_ord[1]], z0[obj_ord[2]], s=opt_size, edgecolor='None', marker='.', c='black', zorder=3)
        ax.text(z0[obj_ord[0]], z0[obj_ord[1]], z0[obj_ord[2]], 'Ideal', fontsize=9, color='black', ha='left', va='center', fontweight='light', zorder=3)
        ax.scatter(znadir[obj_ord[0]], znadir[obj_ord[1]], znadir[obj_ord[2]], s=opt_size, edgecolor='None', marker='.', c='black', zorder=3)
        ax.text(znadir[obj_ord[0]], znadir[obj_ord[1]], znadir[obj_ord[2]], 'Nadir', fontsize=9, color='black', ha='left', va='center', fontweight='light', zorder=3)

        for p in eMOGA['profiles']:
            idx = p['idx']
            ax.scatter(pfront[idx, obj_ord[0]], pfront[idx, obj_ord[1]], pfront[idx, obj_ord[2]], s=opt_size, edgecolor='None', marker='.', c=color_opt, zorder=3)
            ax.text(pfront[idx, obj_ord[0]], pfront[idx, obj_ord[1]], pfront[idx, obj_ord[2]], p['Label'], fontsize=9, color='black', ha='left', va='center', fontweight='light', zorder=3)

            if p['Name'].lower() == 'optimum':
                optims = np.concatenate([np.array([z0]), np.array([pfront[idx, :]])], axis=0)
                optims[:] = optims[:, obj_ord]
                ax.plot(*optims.T, lw=0.2, c='black')
                optims = np.concatenate([np.array([znadir]), np.array([pfront[idx, :]])], axis=0)
                optims[:] = optims[:, obj_ord]
                ax.plot(*optims.T, lw=0.2, c='black')

    if 'graph options' in eMOGA:
        if '3D_view' in graph_options:
            ax.view_init(
                elev=graph_options['3D_view']['elevation'],
                azim=graph_options['3D_view']['azimuth']
                ) 
        else:
            ax.view_init(elev=20, azim=-70, roll=0) 

    ax.set_xlabel(objectives[obj_ord[0]].replace('-','$-$')); 
    ax.set_ylabel(objectives[obj_ord[1]].replace('-','$-$')) 
    ax.set_zlabel(objectives[obj_ord[2]].replace('-','$-$')) 

    # Color de panell 3D
    cpan = 0.980
    ax.xaxis.set_pane_color((cpan, cpan, cpan, 1.0))
    ax.yaxis.set_pane_color((cpan, cpan, cpan, 1.0))
    ax.zaxis.set_pane_color((cpan, cpan, cpan, 1.0))
    
    # Estil de les línies del grid
    color3Dgrid = "#e0e0e0"
    ax.xaxis._axinfo["grid"]['color'] = color3Dgrid
    #ax.zaxis._axinfo["grid"]['linestyle'] = "--"
    ax.yaxis._axinfo["grid"]['color'] = color3Dgrid
    ax.zaxis._axinfo["grid"]['color'] = color3Dgrid
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    #ax.tick_params(axis='both', which='minor', labelsize=6)
    
    if len(description) > 0:
        superior_title = description + ' - 3D Pareto Front'
    else:
        superior_title = '3D Pareto Front'
    
    fig.suptitle(superior_title, fontsize=12, y=0.85, va="top")

    show_fig(block=False)

    return fig

# ------------------------------------------------------------------

def plot_3D_Pareto_Front_plotly(eMOGA, plot_optims=True, cmap='winter'):

    n_obj = len(eMOGA['objectives'])
    if n_obj != 3:
        print(f"This function only works for 3D Pareto fronts. The number of objectives is {n_obj}")
        return

    if 'profiles' not in eMOGA or len(eMOGA['profiles']) == 0:
        eMOGA = get_preference_profile_solutions(eMOGA, verbose=False)

    color_opt = 'red'

    if 'graph options' in eMOGA:
        graph_options = eMOGA['graph options'].copy()
    else:
        graph_options = dict()

    if 'obj_ord_3D' in graph_options:
        obj_ord = graph_options['obj_ord_3D']
    else:
        obj_ord = [0, 1, 2]
    if 'dot_size_3D' in graph_options:
        dot_size = graph_options['dot_size_3D']
    else:
        dot_size = 1
    if 'opt_dot_size_factor_3D' in graph_options:
        opt_size = dot_size * graph_options['opt_dot_size_factor_3D']
    else:
        opt_size = dot_size * 4        
    if 'opt_color' in graph_options:
        color_opt = graph_options['opt_color']

    cmap = plt.colormaps[cmap]
    colors = cmap(np.linspace(0, 1, 256))  # Extract RGBA values
    # Convert to Plotly format: (normalized value, 'rgb(r,g,b)')
    plotly_winter_scale = [(i / 255, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})") for i, (r, g, b, _) in enumerate(colors)]

    df = pd.DataFrame(eMOGA['coste_A'] * eMOGA['param']['signs'], columns=eMOGA['objectives'])
    df["Color"] = eMOGA['color']

    fig = px.scatter_3d(df, 
                         x=eMOGA['objectives'][obj_ord[0]], 
                         y=eMOGA['objectives'][obj_ord[1]], 
                         z=eMOGA['objectives'][obj_ord[2]], 
                         size_max=12, 
                         color="Color",
                         color_continuous_scale=plotly_winter_scale, 
                         width=800, 
                         height=800,
                         title=f"3D Scatter plot of the Pareto frontier of {eMOGA['description']}"
                        )

    camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=-1.95, y=-1.00, z=0.25)
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene_camera=camera)

    marker_size = dot_size
    showscale_visibility = False
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(coloraxis_showscale=showscale_visibility)

    grid_alpha = 0.25
    bg_alpha = 0.025

    fig.update_layout(
        scene = dict(
            xaxis = dict(
                backgroundcolor=f"rgba(0, 0, 0, {bg_alpha})",
                gridcolor=f"rgba(0, 0, 0, {grid_alpha})",
                showbackground=True,
                zerolinecolor=f"rgba(0, 0, 0, {bg_alpha})",
            ),
            yaxis = dict(
                backgroundcolor=f"rgba(0, 0, 0, {bg_alpha})",
                gridcolor=f"rgba(0, 0, 0, {grid_alpha})",
                showbackground=True,
                zerolinecolor=f"rgba(0, 0, 0, {bg_alpha})",
            ),
            zaxis = dict(
                backgroundcolor=f"rgba(0, 0, 0, {bg_alpha})",
                gridcolor=f"rgba(0, 0, 0, {grid_alpha})",
                showbackground=True,
                zerolinecolor=f"rgba(0, 0, 0, {bg_alpha})",
            ),
        )
    )

    if plot_optims:
        
        # Nadir
        z_nadir = eMOGA['z_nadir']
        fig.add_trace(go.Scatter3d(
            x=np.array(z_nadir[obj_ord[0]]),
            y=np.array(z_nadir[obj_ord[1]]),
            z=np.array(z_nadir[obj_ord[2]]),
            mode='markers',
            marker=dict(size=opt_size, color='black', symbol='circle-open'), 
            name='Nadir',
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=np.array(z_nadir[obj_ord[0]]),
            y=np.array(z_nadir[obj_ord[1]]),
            z=np.array(z_nadir[obj_ord[2]]),
            mode='text',
            text=["Nadir"],
            textposition="middle right",
            showlegend=False
        ))

        # Ideal
        z_ideal = eMOGA['z_ideal']
        fig.add_trace(go.Scatter3d(
            x=np.array(z_ideal[obj_ord[0]]),
            y=np.array(z_ideal[obj_ord[1]]),
            z=np.array(z_ideal[obj_ord[2]]),
            mode='markers',
            marker=dict(size=opt_size, color='black', symbol='circle'),
            name='Ideal',
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=np.array(z_ideal[obj_ord[0]]),
            y=np.array(z_ideal[obj_ord[1]]),
            z=np.array(z_ideal[obj_ord[2]]),
            mode='text',
            text=["Ideal"],
            textposition="middle right",
            showlegend=False
        ))

        for p in eMOGA['profiles']:
            z = p['z']
            fig.add_trace(go.Scatter3d(
                x=np.array(z[obj_ord[0]]),
                y=np.array(z[obj_ord[1]]),
                z=np.array(z[obj_ord[2]]),
                mode='markers',
                marker=dict(size=opt_size, color=color_opt, symbol='circle'),
                name=p['Label'],
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=np.array(z[obj_ord[0]]),
                y=np.array(z[obj_ord[1]]),
                z=np.array(z[obj_ord[2]]),
                mode='text',
                text=[p['Label']],
                textposition="middle right",
                showlegend=False
            ))

            # Lines

            if p['Name'].lower() == 'optimum':
                z_opt = z
                points = np.concatenate([np.array([z_ideal]), np.array([z_opt])], axis=0).T
                points[:] = points[obj_ord, :]
                fig.add_trace(go.Scatter3d(
                    x=points[0, :], y=points[1, :], z=points[2, :],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))
                points = np.concatenate([np.array([z_nadir]), np.array([z_opt])], axis=0).T
                points[:] = points[obj_ord, :]
                fig.add_trace(go.Scatter3d(
                    x=points[0, :], y=points[1, :], z=points[2, :],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ))

    if 'graph options' in eMOGA:
        if '3D_view' in eMOGA['graph options']:
            azimuth_rad = np.radians(eMOGA['graph options']['3D_view']['azimuth'])
            elevation_rad = np.radians(eMOGA['graph options']['3D_view']['elevation'])
            r = eMOGA['graph options']['3D_view']['distance']
        else:
            azimuth_rad = np.radians(45)
            elevation_rad = np.radians(20)
            r = 2

        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(
                        x=r * np.cos(elevation_rad) * np.cos(azimuth_rad),
                        y=r * np.cos(elevation_rad) * np.sin(azimuth_rad),
                        z=r * np.sin(elevation_rad),
                    )
                )
            ),
        )

    show_fig(block=False)

    return fig

# ------------------------------------------------------------------

def plot_2D_Pareto_Front(eMOGA, plot_optims=True, cmap='winter'):

    if 'profiles' not in eMOGA or len(eMOGA['profiles']) == 0:
        eMOGA = get_preference_profile_solutions(eMOGA, verbose=False)

    pfront = eMOGA['coste_A'].copy() * eMOGA['param']['signs']
    objectives = eMOGA['objectives'].copy()

    color_opt = 'red'

    if 'graph options' in eMOGA:
        graph_options = eMOGA['graph options'].copy()
        if 'obj_ord_2D' in graph_options:
            obj_ord = graph_options['obj_ord_2D']
        else:
            obj_ord = [1, 0]
        if 'dot_size_2D' in graph_options:
            dot_size = graph_options['dot_size_2D']
        if 'opt_dot_size_factor_2D' in graph_options:
            opt_size = dot_size * graph_options['opt_dot_size_factor_2D']
        if 'opt_color' in eMOGA['graph options']:
            color_opt = eMOGA['graph options']['opt_color']

    color = eMOGA['color']
    z0 = eMOGA['z_ideal'].copy()
    znadir = eMOGA['z_nadir'].copy()

    if 'description' in eMOGA:
        description = eMOGA['description']
    else:
        description = ''

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot()
    ax.scatter(pfront[:, obj_ord[0]], pfront[:, obj_ord[1]], s=dot_size, c=color, edgecolor='None', marker='.', cmap=cmap)
    notacio_cientifica(ax, axis='x')
    notacio_cientifica(ax, axis='y')

    fontSize_labels = 9

    if plot_optims:

        xmin, xmax = ax.get_xlim()[0], ax.get_xlim()[1]
        incx = +0.02 * (xmax - xmin)
        ha = 'left'
        ymin, ymax = ax.get_ylim()[0], ax.get_ylim()[1]
        incy = 0.01 * (ymax - ymin)

        ax.scatter(x=z0[obj_ord[0]], y=z0[obj_ord[1]], s=opt_size, c='black', edgecolor='None', marker='.', zorder=3)
        ax.text(z0[obj_ord[0]] + incx, z0[obj_ord[1]], 'Ideal', fontsize=fontSize_labels, color='black', ha=ha, va='center', fontweight='light', zorder=3)
        
        ax.scatter(x=znadir[obj_ord[0]], y=znadir[obj_ord[1]], s=.25*opt_size, c='white', edgecolor='black', marker='o', zorder=3)
        ax.text(znadir[obj_ord[0]] + incx, znadir[obj_ord[1]], 'Nadir', fontsize=fontSize_labels, color='black', ha=ha, va='center', fontweight='light', zorder=3)
        
        for p in eMOGA['profiles']:            
            z = p['z']
            if 'color' in p: 
                color_opt_i = p['color']
            else:
                color_opt_i = color_opt
            ax.scatter(z[obj_ord[0]], z[obj_ord[1]], s=opt_size, edgecolor='None', marker='.', c=color_opt_i, zorder=3)
            ax.text(z[obj_ord[0]] - incx, z[obj_ord[1]], p['Label'], fontsize=8, ha='right', va='bottom')
            if p['Name'].lower() == 'optimum':
                optims = np.concatenate([np.array([z0]), np.array([z])], axis=0)
                optims[:] = optims[:, obj_ord]
                ax.plot(*optims.T, lw=0.2, c='black')
                optims = np.concatenate([np.array([znadir]), np.array([z])], axis=0)
                optims[:] = optims[:, obj_ord]
                ax.plot(*optims.T, lw=0.2, c='black')

    ax.set_xlabel(objectives[obj_ord[0]].replace('-','$-$')); 
    ax.set_ylabel(objectives[obj_ord[1]].replace('-','$-$')) 
    
    if len(description) > 0:
        superior_title = description + ' - 2D Pareto Front'
    else:
        superior_title = '2D Pareto Front'
    
    fig.suptitle(superior_title, fontsize=12, y=0.925, va="top")

    ax.grid(True)
    show_fig(block=False)

    return fig


# ------------------------------------------------------------------

def plot_Pareto_Front(eMOGA, plot_optims=True, cmap='winter', plotly=True):
    
    n_obj = len(eMOGA['objectives'])
    
    if n_obj == 3:

        if plotly:
            fig = plot_3D_Pareto_Front_plotly(eMOGA, plot_optims=plot_optims, cmap='winter')
        else:
            fig = plot_3D_Pareto_Front(eMOGA, plot_optims=plot_optims, cmap=cmap)
    
    elif n_obj == 2:
        fig = plot_2D_Pareto_Front(eMOGA, plot_optims=plot_optims, cmap=cmap)    
    
    return fig


# ------------------------------------------------------------------

def plot_Level_Diagrams(eMOGA, plot_optims=True,
                        plot_params_LD=True, kpld=2, subplot_size=4.25, cmap='winter'):

    if 'profiles' not in eMOGA or len(eMOGA['profiles']) == 0:
        eMOGA = get_preference_profile_solutions(eMOGA, verbose=False)

    pfront = eMOGA['coste_A'] * eMOGA['param']['signs']
    pset = eMOGA['ele_A']
    objectives = eMOGA['objectives']
    pfront_norm = normalize_frontier(eMOGA['coste_A'])

    dot_size = 1
    opt_size = dot_size * 16
    color_opt = 'red'
    opt_text_bgcolor = 'white'

    if 'graph options' in eMOGA:
        graph_options = eMOGA['graph options'].copy()
        if 'dot_size_2D' in graph_options:
            dot_size = graph_options['dot_size_2D']
        if 'opt_dot_size_factor_2D' in graph_options:
            opt_size = dot_size * graph_options['opt_dot_size_factor_2D']
        if 'opt_color' in eMOGA['graph options']:
            color_opt = eMOGA['graph options']['opt_color']
        if 'opt_text_background_color' in eMOGA['graph options']:
            opt_text_bgcolor = eMOGA['graph options']['opt_text_background_color']

    active_style = plt.rcParams["axes.facecolor"]
    if active_style in ["black", "#000000"]:
        mode_fosc = True
        color_opt_text = 'white'        
    else:
        mode_fosc = False
        color_opt_text = 'black'

    d = np.sqrt(np.sum((pfront_norm)**2, axis=1))  # Euclidean distance to the normalized ideal with no preferences
    color = eMOGA['color']

    if 'description' in eMOGA:
        description = eMOGA['description']
    else:
        description = ''

    n_obj = len(objectives)

    fig, axes = plt.subplots(nrows=1, ncols=n_obj, 
                            figsize=(subplot_size * n_obj, 1.1 * subplot_size)
                            )

    for i in range(n_obj):

        axes[i].scatter(pfront[:, i], d, s=dot_size, c=color, edgecolor='None', marker='.', cmap=cmap)
        notacio_cientifica(axes[i], axis='x')
        notacio_cientifica(axes[i], axis='y')

        if plot_optims:

            xmin, xmax = axes[i].get_xlim()[0], axes[i].get_xlim()[1]
            incx = 0.01 * (xmax - xmin)
            ymin, ymax = axes[i].get_ylim()[0], axes[i].get_ylim()[1]
            ymin = 0.95 * ymin
            axes[i].set_ylim(ymin, ymax)
            incy = 0.01 * (ymax - ymin)

            for p in eMOGA['profiles']:
                idx = p['idx']
                if 'color' in p: 
                    color_opt_i = p['color']
                else:
                    color_opt_i = color_opt
                axes[i].scatter(pfront[idx,i], d[idx], s=opt_size, color=color_opt_i, edgecolor='None', marker='.', label='z0')
                # axes[i].text(pfront[idx,i] + incx, d[idx] - incy, p['Label'], fontsize=9, color=color_opt_text, backgroundcolor=opt_text_bgcolor, ha='left', va='top', fontweight='light')
                axes[i].text(
                    pfront[idx, i] + incx,
                    d[idx] - incy,
                    p['Label'],
                    fontsize=9,
                    color=color_opt_text,
                    ha='left', va='top',
                    fontweight='light',
                    bbox=dict(
                        boxstyle='square,pad=0.05',   # prova 0.00–0.15
                        facecolor=opt_text_bgcolor,
                        edgecolor='none'
                    )
                )
                if mode_fosc: axes[i].grid(color="gray", linestyle="--", linewidth=0.75)


        axes[i].set_xlabel(objectives[i].replace('-', '$-$'))
        if i == 0: 
            axes[i].set_ylabel('Normalized distance')
        else:
            axes[i].set_yticklabels([])

        axes[i].grid(True)

    if len(description) > 0:
        superior_title = description + ' - Level Diagrams'
    else:
        superior_title = 'Level Diagrams'
    
    fig.suptitle(superior_title, fontsize=12, y=0.975, va="top")

    # Adjust the space between the plots
    plt.tight_layout()
    show_fig(block=False)

    # ------------------------------------------------------------------
    # Level Diagrams for parameters

    if plot_params_LD:

        T = pset.shape[1]
        k = kpld # Number of columns more than rows

        nrows = np.int16(np.ceil(-k/2 + (1/2) * np.sqrt(k**2 + (4*T))))
        ncols = nrows + k

        fig2 = plt.figure(figsize=(0.65 * subplot_size * ncols, 0.65 * subplot_size * nrows))
        for i in range(nrows):
            for j in range(ncols):
                if i * ncols + j + 1 <= T:
                    ax = plt.subplot(nrows, ncols, i * ncols + j + 1)
                    ax.scatter(pset[:, i * ncols + j], d, s=dot_size, c=color, edgecolor='None', marker='.', cmap=cmap)
                    if plot_optims:
                        idx = [p['idx'] for p in eMOGA['profiles'] if p['Name'].lower() == 'optimum'][0]
                        ax.scatter(pset[idx, i * ncols + j], d[idx], s=opt_size, c='red', edgecolor='None', marker='.', label='z0')
                    ax.set_xlabel(f"$x_{{{i * ncols + j + 1}}}$")
                    ax.grid(True)
                    if 'xmax' in eMOGA:
                        ax.set_xlim(0, eMOGA['xmax'])
                    else:
                        ax.set_xlim(0, 1)
                    if j == 0: 
                        ax.set_ylabel('Normalized distance')
                    else:
                        ax.set_yticklabels([])

        fig2.suptitle(superior_title, fontsize=12, y=0.995, va="top")

        plt.tight_layout()
        show_fig(block=False)

    return fig, axes

# ------------------------------------------------------------------

def plot_evMOGA_evolution(eMOGA):

    if 'time_Nit_gen' in eMOGA:

        Nit = eMOGA['Nit'][0][0]

        time_Nit_gen = eMOGA['time_Nit_gen'].flatten()
        gens = np.arange(0, time_Nit_gen.size) * Nit
        incr = np.diff(time_Nit_gen, prepend=time_Nit_gen[0])

        plots = [
            {
                'y': eMOGA['Nind_A_Nit_gen'].flatten(),
                'ylabel': "Number of solutions",
                'title': f"Solutions on the Pareto front",
                'color': '#1f77b4'
            },
            {
                'y': incr,
                'ylabel': 'Time (s)',
                'title': f"Elapsed time every {Nit} generations",
                'color': 'green'
            }
        ]

        nplots = len(plots)
        fig, axes = plt.subplots(nplots, 1, figsize=(7, 2.25 * nplots), sharex=True)
        if nplots == 1:
            axes = [axes]

        for ax, plot in zip(axes, plots):
            ax.plot(gens, plot['y'], linewidth=0.5, marker='o', markersize=3, color=plot['color'])
            ax.set_ylabel(plot['ylabel'])
            ax.set_title(plot['title'])
            ax.grid(True)
        axes[-1].set_xlabel('Generations')

        plt.tight_layout()
        show_fig(block=False)
    else:
        print("No evolution data found in eMOGA.")

    return fig, axes

# ------------------------------------------------------------------
# Fi
