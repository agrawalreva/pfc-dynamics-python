import numpy as np

def vec(x):
    """Vectorize input matrix"""
    return x.flatten()

def slow_mult(A, B):
    """Slow matrix multiply - placeholder"""
    return A @ B

def slow_backslash(A, B):
    """Slow backslash - placeholder"""
    return np.linalg.solve(A, B)

def slow_chol(A):
    """Slow cholesky - placeholder"""
    return np.linalg.cholesky(A)

def kronmult(Amats, x, ii=None):
    """
    Multiply matrix (.... A{3} kron A{2} kron A{1})(:,ii) by x
    
    Args:
        Amats: list of matrices [A1, ..., An]
        x: matrix to multiply with kronecker matrix
        ii: binary vector indicating sparse locations (optional)
    
    Returns:
        y: result matrix
    """
    ncols = x.shape[1]
    
    # handle sparse indices if provided
    if ii is not None:
        x0 = np.zeros((len(ii), ncols))
        x0[ii, :] = x
        x = x0
    
    nrows = x.shape[0]
    nA = len(Amats)
    
    if nA == 1:
        # single matrix multiply
        y = Amats[0] @ x
    else:
        # multiple matrices - apply in sequence
        y = x.copy()
        for jj in range(nA):
            ni, nj = Amats[jj].shape
            y = Amats[jj] @ y.reshape(nj, -1)
            # reshape and permute
            y = y.reshape(ni, nrows // nj, -1)
            y = np.transpose(y, (1, 0, 2))
            nrows = ni * nrows // nj
        
        y = y.reshape(nrows, ncols)
    
    return y
