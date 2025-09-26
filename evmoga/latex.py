
# Funcions per exportar variables de l'eMOGA a LaTeX

import numpy as np

# --------------------------------------------------------------------------------------------

def newcommand_LaTeX(command, value, format=None):
    if format:
        value = f"{value:{format}}"
    return f"\\ifundef{{\\{command}}}\n\t{{\\newcommand{{\\{command}}}{{{value}}}}}\n\t{{\\renewcommand{{\\{command}}}{{{value}}}}}\n\n"

# --------------------------------------------------------------------------------------------

def newcommand_LaTeX_num(command, value):
    return f"\\ifundef{{{command}}}\n\t{{\\newcommand{{{command}}}{{\\num{{{value}}}}}}}\n\t{{\\renewcommand{{{command}}}{{\\num{{{value}}}}}}}\n\n"

# --------------------------------------------------------------------------------------------

def export_to_LaTeX_evMOGA(eMOGA, filename, parametres=["Nind_P",
                                                 "Generations", 
                                                 #"gen_counter", 
                                                 "Nind_GA", 
                                                 "n_div",
                                                 "searchspaceUB", 
                                                 "searchspaceLB", 
                                                 "Sigma_Pm_ini", 
                                                 "Sigma_Pm_fin", 
                                                 "dd_ini",
                                                 "dd_fin",
                                                 "Pm", 
                                                 "Nit",
                                                 #"Nind_A",
                                                 "precision_onoff",
                                                 "precision"]):

    with open(filename, "w") as f:
        for p in parametres:
            command = f"\\{p.replace('_', '')}Value"
            if isinstance(eMOGA[f"{p}"], (list, np.ndarray)):
                f.write(newcommand_LaTeX_num(command=command, value=eMOGA[f"{p}"][0,0]))
            else:
                f.write(newcommand_LaTeX_num(command=command, value=eMOGA[f"{p}"]))
    # --------------------------------------------------------------------------------------------
    # Final
