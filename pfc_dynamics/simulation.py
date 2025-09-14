import numpy as np
from scipy.stats import multivariate_normal
from .utils import kronmult

def sim_weights(n, T, P, rP, len_scale, rho):
    """
    Generate random weights and basis functions
    
    Args:
        n: number of neurons
        T: number of time points  
        P: number of task variables
        rP: ranks for each task variable (length P-1, last is condition-independent)
        len_scale: length scale parameters
        rho: variance parameters
    
    Returns:
        W: list of weight matrices
        S: list of basis function matrices
        BB: concatenated coefficient matrix
    """
    BB = []
    W = []
    S = []
    
    # first P-1 components have specified ranks
    for p in range(P-1):
        # generate covariance matrix for smooth time components
        time_diff = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
        Scov = np.exp(-(time_diff / len_scale[p])**2 / 2)
        
        # random weights
        Wp = rho[p] * np.random.randn(n, rP[p])
        W.append(Wp)
        
        # sample basis functions
        if rP[p] == 1:
            Sp = multivariate_normal.rvs(mean=np.zeros(T), cov=Scov, size=1).reshape(-1, 1)
        else:
            Sp = multivariate_normal.rvs(mean=np.zeros(T), cov=Scov, size=rP[p]).T
        Sp = np.flip(Sp, axis=0)  # flip for some reason
        S.append(Sp)
        
        # concatenate coefficients - Wp is n x rP[p], Sp is T x rP[p], so Wp*Sp' is n x T
        BB.append(Wp @ Sp.T)
    
    # last component is condition-independent (full rank)
    p = P - 1
    time_diff = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :])
    Scov = np.exp(-(time_diff / len_scale[p])**2 / 2)
    
    # condition-independent component has full rank T
    Wp = rho[p] * np.random.randn(n, T)
    W.append(Wp)
    
    # identity basis for condition-independent component
    Sp = np.eye(T)
    S.append(Sp)
    
    # concatenate coefficients
    BB.append(Wp @ Sp.T)
    
    BB = np.vstack(BB)
    return W, S, BB

def sim_conditions(var_uniq, N):
    """
    Generate experimental conditions
    
    Args:
        var_uniq: list of unique values for each variable
        N: number of trials to sample
    
    Returns:
        Xsamp: sampled conditions [N x P]
        Xcond: all possible conditions
    """
    P = len(var_uniq)
    levP = [len(var_uniq[p]) for p in range(P)]
    
    # create all possible combinations
    if P == 1:
        Xcond = np.meshgrid(var_uniq[0])[0].flatten()
    elif P == 2:
        x1, x2 = np.meshgrid(var_uniq[0], var_uniq[1])
        Xcond = np.column_stack([x1.ravel(), x2.ravel()])
    elif P == 3:
        x1, x2, x3 = np.meshgrid(var_uniq[0], var_uniq[1], var_uniq[2])
        Xcond = np.column_stack([x1.ravel(), x2.ravel(), x3.ravel()])
    elif P == 4:
        x1, x2, x3, x4 = np.meshgrid(var_uniq[0], var_uniq[1], var_uniq[2], var_uniq[3])
        Xcond = np.column_stack([x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel()])
    elif P == 5:
        x1, x2, x3, x4, x5 = np.meshgrid(var_uniq[0], var_uniq[1], var_uniq[2], var_uniq[3], var_uniq[4])
        Xcond = np.column_stack([x1.ravel(), x2.ravel(), x3.ravel(), x4.ravel(), x5.ravel()])
    
    # sample N conditions
    n_conditions = np.prod(levP)
    sample_idx = np.random.choice(n_conditions, size=N, replace=True)
    Xsamp = Xcond[sample_idx]
    
    return Xsamp, Xcond

def sim_pop_data(Xsamp, BB, d, n, T, N):
    """
    Generate population data from BTDR model
    
    Args:
        Xsamp: [N x P] matrix of regressors
        BB: [P x 1] list of n x T coefficient matrices
        d: [n x 1] noise variances
        n: number of neurons
        T: number of time points
        N: number of trials
    
    Returns:
        Y: [n x T x N] array of observations
    """
    Y = np.zeros((n, T, N))
    
    for k in range(N):
        # add noise
        if np.isscalar(d):
            noisek = np.random.randn(n, T) / np.sqrt(d)
        else:
            noisek = np.diag(1.0 / np.sqrt(d)) @ np.random.randn(n, T)
        
        # signal + noise
        Y[:, :, k] = kronmult([np.eye(n), Xsamp[k:k+1, :]], BB) + noisek
    
    return Y
