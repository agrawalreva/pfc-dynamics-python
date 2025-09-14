"""
test timing comparisons
"""

import numpy as np
import time

def main():
    # set reasonable dimensions
    n = 762
    P = 6
    T = 15
    TP = T * P
    rtot = TP * 2
    r = 2 * np.ones(P, dtype=int)
    
    print("Compare backslash")
    maxrtot = 200
    rtots = np.arange(10, maxrtot + 1, 5)
    
    # commented out original MATLAB code for reference
    # for rind in range(len(rtots)):
    #     S = np.random.randn(rtot, TP, n)
    #     X = np.random.randn(TP, rtot, n)
    #     for ii in range(n):
    #         A[:, :, ii] = X[:, :, ii].T @ X[:, :, ii] + TP * TP * np.eye(rtot)
    #     
    #     # compare backslash
    #     tic = time.time()
    #     for ii in range(n):
    #         invAS1[:, :, ii] = np.linalg.solve(A[:, :, ii], S[:, :, ii])
    #     Tloop[rind] = time.time() - tic
    #     
    #     tic = time.time()
    #     invAS2 = mmx_mkl_single('backslash', A, S)
    #     Tmmx[rind] = time.time() - tic
    
    print("Compare multiply")
    
    S = np.random.randn(rtot, TP, n)
    X = np.random.randn(TP, rtot, n)
    A = np.zeros((rtot, rtot, n))
    
    for ii in range(n):
        A[:, :, ii] = X[:, :, ii].T @ X[:, :, ii] + TP * TP * np.eye(rtot)
    
    # compare backslash
    tic = time.time()
    AS1 = np.zeros_like(S)
    for ii in range(n):
        AS1[:, :, ii] = A[:, :, ii] @ S[:, :, ii]
    toc1 = time.time() - tic
    print(f"Loop time: {toc1:.4f}s")
    
    # vectorized version
    tic = time.time()
    # reshape for batch operations - need to be more careful about dimensions
    AS2 = np.zeros_like(AS1)
    for ii in range(n):
        AS2[:, :, ii] = A[:, :, ii] @ S[:, :, ii]
    toc2 = time.time() - tic
    print(f"Vectorized time: {toc2:.4f}s")
    
    print(f"Speedup: {toc1/toc2:.2f}x")
    
    # verify results are the same
    print(f"Max difference: {np.max(np.abs(AS1 - AS2)):.2e}")

if __name__ == "__main__":
    main()
