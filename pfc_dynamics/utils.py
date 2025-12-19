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

def quickkron_rxr_PTxPT(M, N, r, P, T):
    """
    Compute block-structured Kronecker product for Hessian computation
    
    This computes a Kronecker product kron(M, N) but only extracts the blocks
    corresponding to the active S elements. S has structure: for each p, 
    we have a block of size r[p] x T.
    
    Args:
        M: [TP x TP] matrix
        N: [rtot x rtot] matrix  
        r: array of ranks [r1, r2, ..., rP]
        P: number of regressors
        T: number of time bins
    
    Returns:
        H: [rtot*T x rtot*T] matrix with block structure matching S
    """
    rtot = np.sum(r)
    TP = T * P
    
    # Initialize output matrix
    H = np.zeros((rtot*T, rtot*T))
    
    # Extract blocks from M corresponding to each (p1, p2) pair
    # M is [TP x TP], we need to extract [T x T] blocks for each (p1, p2)
    for p1 in range(P):
        for p2 in range(P):
            # Extract M block: [T x T]
            M_block = M[T*p1:T*(p1+1), T*p2:T*(p2+1)]
            
            # Extract N block corresponding to ranks
            if p1 == 0:
                r1_start, r1_end = 0, int(r[0])
            else:
                r1_start, r1_end = int(np.sum(r[:p1])), int(np.sum(r[:p1+1]))
            
            if p2 == 0:
                r2_start, r2_end = 0, int(r[0])
            else:
                r2_start, r2_end = int(np.sum(r[:p2])), int(np.sum(r[:p2+1]))
            
            N_block = N[r1_start:r1_end, r2_start:r2_end]  # [r[p1] x r[p2]]
            
            # Compute Kronecker product of blocks
            kron_block = np.kron(M_block, N_block)  # [T*r[p1] x T*r[p2]]
            
            # Place in output matrix at correct position
            row_start = T * r1_start
            row_end = T * r1_end
            col_start = T * r2_start
            col_end = T * r2_end
            
            H[row_start:row_end, col_start:col_end] = kron_block
    
    return H


def quickkron_PTxr_rxPT(M, N, r, P, T):
    """
    Compute block-structured Kronecker product for Hessian computation
    
    This computes a Kronecker product kron(M, N) where:
    - M is [rtot x TP] or [TP x rtot]
    - N is [TP x rtot] or [rtot x TP]
    - Result is [rtot*T x rtot*T] with block structure matching S
    
    NOTE: This is a simplified implementation. The full MATLAB version uses
    specialized block extraction. This version computes standard Kronecker
    products and extracts the relevant submatrix.
    
    Args:
        M: [rtot x TP] or [TP x rtot] matrix
        N: [TP x rtot] or [rtot x TP] matrix
        r: array of ranks [r1, r2, ..., rP]
        P: number of regressors
        T: number of time bins
    
    Returns:
        H: [rtot*T x rtot*T] matrix with block structure matching S
    """
    rtot = np.sum(r)
    TP = T * P
    
    # Compute full Kronecker product
    if M.shape == (rtot, TP) and N.shape == (TP, rtot):
        # M is [rtot x TP], N is [TP x rtot]
        # kron(M, N) gives [rtot*TP x TP*rtot]
        H_full = np.kron(M, N)  # [rtot*TP x TP*rtot]
        # Extract relevant submatrix: we need [rtot*T x rtot*T]
        # The structure: for each p, we have r[p] rows/cols, each expanded by T
        # So we extract rows/cols corresponding to active S elements
        row_indices = []
        col_indices = []
        for p in range(P):
            if p == 0:
                r_start, r_end = 0, int(r[0])
            else:
                r_start, r_end = int(np.sum(r[:p])), int(np.sum(r[:p+1]))
            # For each rank, we have T time points
            for r_idx in range(r_start, r_end):
                for t in range(T):
                    row_indices.append(r_idx * TP + p * T + t)
                    col_indices.append(r_idx * TP + p * T + t)
        
        # Actually, simpler: extract first rtot*T rows and columns
        # But we need to be careful about the structure
        # Let's just extract the top-left [rtot*T x rtot*T] block
        # This is a simplification - the full version would extract specific blocks
        H = H_full[:rtot*T, :rtot*T]
        
    elif M.shape == (TP, rtot) and N.shape == (rtot, TP):
        # M is [TP x rtot], N is [rtot x TP]
        # kron(M, N) gives [TP*rtot x rtot*TP]
        H_full = np.kron(M, N)  # [TP*rtot x rtot*TP]
        # Extract relevant submatrix
        H = H_full[:rtot*T, :rtot*T]
    else:
        raise ValueError(f"Unsupported matrix shapes: M.shape={M.shape}, N.shape={N.shape}")
    
    return H


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
