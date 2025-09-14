import numpy as np

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

def kronmulttrp(Amats, *args):
    """
    Multiply matrix (A{2} kron A{1})^T times vector x
    
    Args:
        Amats: list of matrices
        *args: additional arguments
    
    Returns:
        y: result
    """
    # transpose all matrices
    Amats_trp = [A.T for A in Amats]
    return kronmult(Amats_trp, *args)

def multikron(A):
    """
    Form matrix (A{n} kron ... A{2} kron A{1})
    
    Args:
        A: list of matrices
    
    Returns:
        B: kronecker product matrix
    """
    if len(A) == 1:
        return A[0]
    elif len(A) == 2:
        return np.kron(A[1], A[0])
    else:
        return np.kron(multikron(A[1:]), A[0])

def kroncovdiag(BB, C, ii=None):
    """
    Compute diagonal of BCB^T, where C is symmetric and B is kronecker matrix
    
    Args:
        BB: list of basis matrices
        C: symmetric matrix
        ii: binary vector indicating sparse locations (optional)
    
    Returns:
        mdiag: diagonal of B*C*B'
    """
    nB = len(BB)
    nr = [B.shape[0] for B in BB]
    nc = [B.shape[1] for B in BB]
    nctot = np.prod(nc)
    
    # expand C if necessary
    if ii is not None:
        Cbig = np.zeros((nctot, nctot))
        Cbig[np.ix_(ii, ii)] = C
        C = Cbig
    
    # compute diagonal
    if nB == 1:
        mdiag = np.sum(BB[0] * (BB[0] @ C), axis=1)
    else:
        mdiag = C.copy()
        for jj in range(nB):
            mdiag = BB[jj] @ mdiag.reshape(nc[jj], -1)
            mdiag = mdiag.reshape(nr[jj], nctot // nc[jj], nc[jj], -1)
            mdiag = np.transpose(mdiag, (0, 2, 1, 3))
            mdiag = np.sum(mdiag * BB[jj][:, :, None, None], axis=1)
            mdiag = np.transpose(mdiag, (1, 2, 0))
            nctot = nctot // nc[jj]
        mdiag = mdiag.flatten()
    
    return mdiag
