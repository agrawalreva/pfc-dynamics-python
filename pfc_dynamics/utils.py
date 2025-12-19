import numpy as np

def vec(x):
    """Vectorize input matrix"""
    return x.flatten()

def slow_mult(A, B):
    """Slow matrix multiply for 3D arrays"""
    na = A.shape
    nb = B.shape
    
    if len(na) == 2 and len(nb) == 2:
        # 2D case - regular matrix multiply
        return A @ B
    elif len(na) == 3 and len(nb) == 3:
        # 3D case - multiply along third dimension
        if na[2] != nb[2]:
            raise ValueError('Third dimension of matrices must match')
        C = np.zeros((na[0], nb[1], na[2]))
        for ii in range(na[2]):
            C[:, :, ii] = A[:, :, ii] @ B[:, :, ii]
        return C
    else:
        # Mixed case - try to handle it
        return A @ B

def slow_backslash(A, B):
    """Slow backslash (solve) for 3D arrays"""
    na = A.shape
    nb = B.shape
    
    if len(na) == 2 and len(nb) == 2:
        # 2D case - regular solve
        return np.linalg.solve(A, B)
    elif len(na) == 3:
        # 3D case - solve along third dimension
        if len(nb) == 2:
            # B is 2D, broadcast
            C = np.zeros((na[0], nb[1], na[2]))
            for ii in range(na[2]):
                C[:, :, ii] = np.linalg.solve(A[:, :, ii], B)
        elif len(nb) == 3:
            if na[2] != nb[2]:
                raise ValueError('Third dimension of matrices must match')
            C = np.zeros((na[0], nb[1], na[2]))
            for ii in range(na[2]):
                C[:, :, ii] = np.linalg.solve(A[:, :, ii], B[:, :, ii])
        else:
            raise ValueError('Incompatible dimensions')
        return C
    else:
        # 2D case
        return np.linalg.solve(A, B)

def slow_chol(A):
    """Slow cholesky for 3D arrays"""
    na = A.shape
    
    if len(na) == 2:
        # 2D case - regular cholesky
        return np.linalg.cholesky(A)
    elif len(na) == 3:
        # 3D case - cholesky along third dimension
        C = np.zeros(na)
        for ii in range(na[2]):
            C[:, :, ii] = np.linalg.cholesky(A[:, :, ii])
        return C
    else:
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
