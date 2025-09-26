import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------

def Average_Nearest_Neighbor_Distance(A: np.ndarray, B: np.ndarray, metric: str = 'euclidean') -> tuple[float, np.ndarray]:
    """
    1) Min–max normalització (0–1) per columna tenint en compte A i B.
    2) Distància mitjana dels mínims: mean_i( min_j d(A[i], B[j]) ).
    
    Args:
      A: np.ndarray de forma (n1, m)
      B: np.ndarray de forma (n2, m)
      metric: nom de la mètrica per scipy.spatial.distance.cdist
      
    Retorn:
      float amb la distància mitjana mínima sobre les files de A.
    """
    # 1) Min i max per columna
    #    apilem A i B per càlcul conjunt
    C = np.vstack((A, B))
    mins = C.min(axis=0)      # forma (m,)
    maxs = C.max(axis=0)      # forma (m,)
    
    # 2) Normalització amb control de divisió per zero
    range_ = maxs - mins
    # si alguna columna té range_==0, no la canviem (denom=1 evita NaNs)
    range_[range_ == 0] = 1.0
    
    A_norm = (A - mins) / range_
    B_norm = (B - mins) / range_
    
    # 3) Matriu de distàncies normalitzades
    D = cdist(A_norm, B_norm, metric=metric)
    # mínim per punt d'A i després mitjana
    mins_per_point = D.min(axis=1)

    return mins_per_point.mean(), mins_per_point

# ---------------------------------------------------------------------------

def Hausdorff_Distance(A: np.ndarray, B: np.ndarray, metric: str = 'euclidean') -> tuple[np.ndarray, float, float]:
    """
    Càlcula la distància de Hausdorff entre dos conjunts de punts.

    Args:
      A: np.ndarray de forma (n1, m)
      B: np.ndarray de forma (n2, m)
      metric: nom de la mètrica per scipy.spatial.distance.cdist

    Retorn:
      np.ndarray amb les distàncies de Hausdorff entre A i B.
    """
    # 1) Min i max per columna
    #    apilem A i B per càlcul conjunt
    C = np.vstack((A, B))
    mins = C.min(axis=0)      # forma (m,)
    maxs = C.max(axis=0)      # forma (m,)

    # 2) Normalització amb control de divisió per zero
    range_ = maxs - mins
    # si alguna columna té range_==0, no la canviem (denom=1 evita NaNs)
    range_[range_ == 0] = 1.0

    A_norm = (A - mins) / range_
    B_norm = (B - mins) / range_

    # 3) Distàncies mínimes de cada punt de A a B i de B a A
    Dab = np.min(cdist(A_norm, B_norm, metric=metric), axis=1)  # distància mínima de cada punt de B a A
    Dba = np.min(cdist(B_norm, A_norm, metric=metric), axis=1)  # distància mínima de cada punt de A a B

    dab = Dab.max()  # distància mínima de cada punt de A a B
    dba = Dba.max()  # distància mínima de cada punt

    # 4) Distància de Hausdorff
    return np.max([dab, dba]), dab, dba