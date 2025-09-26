import numpy as np

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