"""
Estimation functions
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from .utils import vec, kronmult, slow_mult, slow_backslash, slow_chol

def mk_suff_stats_bilin_reg_sims(Z, X, hk):
    """
    Calculate sufficient statistics for bilinear regression
    
    Args:
        Z: [n x T x N] data array
        X: [N x P] design matrix
        hk: [N x n] observation matrix
    
    Returns:
        XX, XY, Yn, allstim, n, T
    """
    n, T, N = Z.shape
    P = X.shape[1]
    
    allstim = []
    ntrials = np.sum(hk, axis=0)
    Yn = []
    
    for ii in range(n):
        # stim vals for each trial
        stim_ii = np.zeros((int(ntrials[ii]), P))
        for jj in range(P):
            smps = X[hk[:, ii] == 1, jj]
            stim_ii[:, jj] = smps
        allstim.append(stim_ii)
        Yn.append(Z[ii, :, hk[:, ii] == 1])
    
    # sparse identity matrix
    It = sparse.eye(T)
    XXperneuron = []
    XYperneuron = np.zeros((T * P, n))
    
    for jx in range(n):
        stm = allstim[jx]
        XXperneuron.append(np.kron(stm.T @ stm, It.toarray()))
        # MATLAB: vec(reshape(Yn{jx},T,ntrials(jx))*stm)
        # Yn[jx] is T x ntrials, stm is ntrials x P, so Yn[jx] @ stm is T x P
        ntrials_jx = int(ntrials[jx])
        Yn_reshaped = Yn[jx].reshape(T, ntrials_jx)  # Ensure T x ntrials
        XYperneuron[:, jx] = vec(Yn_reshaped @ stm)
    
    # construct block diagonal XX matrix
    XX = sparse.block_diag(XXperneuron).toarray()
    XY = XYperneuron.flatten()
    
    # permute indices so coefficients grouped by matrix instead of by neuron
    nwtot = P * T * n
    iiperm = np.arange(nwtot).reshape(T, P, n)
    iiperm = vec(np.transpose(iiperm, (0, 2, 1)))
    XX = XX[iiperm, :][:, iiperm]
    XY = XY[iiperm]
    
    return XX, XY, Yn, allstim, n, T

def mk_suff_stats_btdr_incomp_obs_uneqvar_s_fast(X, Z, hk):
    """
    Calculate sufficient statistics for BTDR with incomplete observations
    
    Args:
        X: [N x P] regressor matrix
        Z: [n x T x N] data array
        hk: [N x n] observation matrix
    
    Returns:
        Ri, Ai, zzi, ni, Xi, Xzetai, zi
    """
    n, T, N = Z.shape
    P = X.shape[1]
    ni = np.sum(hk, axis=0)
    
    zi = []
    Xi = []
    Ai = np.zeros((P, P, n))
    Ri = np.zeros((P * T, P * T, n))
    zzi = np.zeros(n)
    Xzetai = np.zeros((T * P, n))
    
    for i in range(n):
        Xi.append(X[hk[:, i] == 1, :])
        Ai[:, :, i] = Xi[i].T @ Xi[i]
        zi.append(Z[i, :, hk[:, i] == 1])
        zetai = vec(zi[i])
        Xzetai[:, i] = kronmult([np.eye(T), Xi[i].T], zetai.reshape(-1, 1)).flatten()
        Ri[:, :, i] = Xzetai[:, i:i+1] @ Xzetai[:, i:i+1].T
        zzi[i] = zetai.T @ zetai
    
    return Ri, Ai, zzi, ni, Xi, Xzetai, zi

def svd_regress_s_vdata(XX, XY, Yn, allstim, T, n, r, ridgeparam, opts):
    """
    SVD regression with data
    
    Args:
        XX, XY: sufficient statistics
        Yn, allstim: data
        T, n: dimensions
        r: ranks
        ridgeparam: ridge parameter
        opts: options
    
    Returns:
        pars, b0
    """
    Bfullhat, Bthat, Bxhat = svd_regress_b(XX, XY, [T, n], r, ridgeparam, opts)
    shatsvd = []
    B = []  # Will be list of transposed matrices
    
    P = len(r)
    for p in range(P):
        shatsvd.extend(vec(Bthat[p]))
        # MATLAB: B = [B Bfullhat(:,:,p)'] - concatenate horizontally
        # Bfullhat is nt x nx x nmats, so Bfullhat(:,:,p) is nt x nx
        # Bfullhat(:,:,p)' is nx x nt
        B.append(Bfullhat[:, :, p].T)  # nx x nt
    
    # Concatenate B matrices horizontally: MATLAB B becomes n x (T*P)
    # In Python, we'll build it as a list and then access by row
    # Actually, let's build B as n x (T*P) matrix like MATLAB
    B_matrix = np.hstack(B)  # n x (T*P)
    
    # estimate lambdas
    lambhat = np.zeros(n)
    for ii in range(n):
        # MATLAB: Bi = vec(B(ii,:)) - get row ii and vectorize
        Bi = vec(B_matrix[ii, :])  # T*P x 1
        # MATLAB: ri = vec(Yn{ii}')- kronmult({speye(T),allstim{ii}},Bi)
        ri = vec(Yn[ii].T) - kronmult([sparse.eye(T), allstim[ii]], Bi.reshape(-1, 1)).flatten()
        lambhat[ii] = allstim[ii].shape[0] * T / (ri.T @ ri)
    
    pars = np.concatenate([lambhat, shatsvd])
    b0 = Bfullhat[:, :, P-1]
    
    return pars, b0

def svd_regress_b(xx, xy, wdims, ps, lambda_reg, opts):
    """
    SVD regression
    
    Args:
        xx, xy: sufficient statistics
        wdims: [T, n] dimensions
        ps: ranks
        lambda_reg: ridge parameter
        opts: options
    
    Returns:
        wsvd, wt, wx
    """
    if lambda_reg is not None and lambda_reg != 0:
        xx = xx + lambda_reg * np.eye(xx.shape[0])
    
    if opts is None:
        opts = {'MaxIter': 25, 'TolFun': 1e-6, 'Display': 'iter'}
    
    nwtot = len(xy)
    nt = wdims[0]  # height
    nx = wdims[1]  # width
    nw = nt * nx   # coeffs 
    nmats = len(ps)  # distinct low rank matrices
    
    if nwtot != nw * nmats:
        raise ValueError('Mismatch in size of data and params')
    
    # initialize estimate by linear regression and SVD
    w0 = np.linalg.solve(xx, xy)
    wt = []
    wx = []
    wsvd = np.zeros((nt, nx, nmats))
    
    for jj in range(nmats):
        iistrt = nw * jj
        iiend = nw * (jj + 1)
        wmat = w0[iistrt:iiend].reshape(nt, nx)
        U, s, Vt = np.linalg.svd(wmat, full_matrices=False)
        wt.append(U[:, :ps[jj]] @ np.diag(np.sqrt(s[:ps[jj]])))
        wx.append(Vt[:ps[jj], :].T @ np.diag(np.sqrt(s[:ps[jj]])))
        wsvd[:, :, jj] = wt[jj] @ wx[jj].T
    
    return wsvd, wt, wx

def est_pars_coord_ascent_lambi_s_b(lambhat0, shat0, bhat0, r, Ai, Xi, zetai, ni, xbari, Ybar, Xzetai0):
    """
    Coordinate ascent estimation
    
    Args:
        lambhat0, shat0, bhat0: initial parameters
        r: ranks
        Ai, Xi, zetai: sufficient statistics
        ni, xbari, Ybar: data
        Xzetai0: cross terms
    
    Returns:
        lambhat, shat, Shat, Shatblock, bhat
    """
    P = len(r)
    rtot = np.sum(r)
    T, n = Ybar.shape
    TP = T * P
    maxsteps = 10
    stopvar = True
    stopcrit = 1e-4
    k = 1
    
    while stopvar:
        # recalculate sufficient stats
        Ri, zzi, _ = ecm_suff_stat(zetai, Xi, bhat0)
        
        # estimate S conditioned on b0 and lambda
        options = {'disp': False}
        loglikS_lamb = lambda s: neg_log_lik_btdr_incomp_obs_uneqvar_s_only(
            s, lambhat0, Ai, Ri, zzi, r, ni, 0)
        result = minimize(loglikS_lamb, shat0, method='BFGS', options=options)
        shat = result.x
        
        Shat = []
        for p in range(P):
            start_idx = np.sum(r[:p]) * T
            end_idx = np.sum(r[:p+1]) * T
            Shat.append(shat[start_idx:end_idx].reshape(r[p], T))
        Shatblock = sparse.block_diag(Shat).toarray()
        
        # estimate lambda conditioned on new S and b0
        loglikfunLambda = lambda lamb: neg_log_lik_btdr_incomp_obs_uneqvar_lamb_only(
            lamb, Shatblock, Ai, Ri, zzi, r, ni, 0)
        result = minimize(loglikfunLambda, lambhat0, method='BFGS')
        lambhat = result.x
        
        # estimate b0 conditioned on new S and new lambda
        S2 = Shatblock.reshape(-1, P)
        AiIS = np.transpose(S2 @ Ai.reshape(P, P*n), (1, 0, 2))
        SAiIS = Shatblock @ AiIS.reshape(TP, rtot*n)
        SAiIS = SAiIS.reshape(rtot, rtot, n)
        lambiSAiS = SAiIS * lambhat[None, None, :]
        Ci = lambiSAiS + np.eye(rtot)[:, :, None]
        bhat = mmle_b(Ci, Shatblock, lambhat, ni, xbari, Ybar, Xzetai0, r)
        
        # convergence check
        newpars = np.concatenate([lambhat, shat, vec(bhat)])
        oldpars = np.concatenate([lambhat0, shat0, vec(bhat0)])
        parerr = np.max((newpars - oldpars)**2 / oldpars**2)
        
        if parerr < stopcrit or k >= maxsteps:
            stopvar = False
        
        lambhat0 = lambhat
        shat0 = shat
        bhat0 = bhat
        k += 1
    
    return lambhat, shat, Shat, Shatblock, bhat

def ecm_suff_stat(zetai, Xi, bhat0):
    """Calculate ECM sufficient statistics"""
    n = len(zetai)
    T = zetai[0].shape[0]
    P = Xi[0].shape[1]
    
    Ri = np.zeros((P*T, P*T, n))
    zzi = np.zeros(n)
    Xzetai = np.zeros((T*P, n))
    
    for i in range(n):
        # MATLAB: zi = vec(bsxfun(@minus,zetai{ii},b(:,ii)))
        # Subtract b from zetai before computing statistics
        zi = zetai[i] - bhat0[:, i:i+1]  # T x ntrials - T x 1 -> T x ntrials
        zi_vec = vec(zi)  # Vectorize
        zi_reshaped = zi_vec.reshape(T, -1)  # Reshape to T x ntrials
        # MATLAB: Xzetai(:,ii) = vec(zi*Xi{ii})
        Xzetai[:, i] = vec(zi_reshaped @ Xi[i])
        Ri[:, :, i] = Xzetai[:, i:i+1] @ Xzetai[:, i:i+1].T
        zzi[i] = vec(zi).T @ vec(zi)
    
    return Ri, zzi, Xzetai

def neg_log_lik_btdr_incomp_obs_uneqvar_s_only(s, lamb, Ai, Ri, zzi, r, ni, g):
    """Negative log likelihood for S only"""
    P = len(r)
    T = Ri.shape[0] // P
    TP = T * P
    n = len(ni)
    rtot = np.sum(r)
    
    # Make block diagonal S matrix
    Shat = []
    for p in range(P):
        start_idx = np.sum(r[:p]) * T
        end_idx = np.sum(r[:p+1]) * T
        Shat.append(s[start_idx:end_idx].reshape(r[p], T))
    S = sparse.block_diag(Shat).toarray()
    
    # lambAiIS := lambi*kron(I,Ai)*S'
    # MATLAB: lambAi = reshape(bsxfun(@times,Ai,permute(lambi,[3 2 1])),P,P*n)
    lambAi = Ai * lamb[None, None, :]  # [P x P x n]
    lambAi = lambAi.reshape(P, P*n)  # [P x P*n]
    # MATLAB: S2 = reshape(S,[],P)
    S2 = S.reshape(-1, P)  # [rtot*T x P] -> reshape to [rtot*T/P x P]? Actually, S is rtot x TP, so reshape(-1, P) gives rtot*TP/P x P = rtot*T x P
    # Wait, S is rtot x TP, so S2 = reshape(S, [], P) in MATLAB means reshape to have P columns
    # In MATLAB, reshape(S, [], P) with S being rtot x TP gives (rtot*TP/P) x P = rtot*T x P
    # But we want rtot x TP reshaped to have P columns, so it's (rtot*TP/P) x P
    # Actually, let me check: S is rtot x (T*P), so reshape(S, [], P) gives (rtot*T*P/P) x P = rtot*T x P
    # In Python, S.reshape(-1, P) with S being rtot x TP gives (rtot*TP/P) x P = rtot*T x P ✓
    # MATLAB: lambAiIS = permute(reshape(S2*lambAi,rtot,TP,n),[2 1 3])
    lambAiIS = S2 @ lambAi  # [rtot*T x P] @ [P x P*n] = [rtot*T x P*n]
    lambAiIS = lambAiIS.reshape(rtot, TP, n)  # Reshape to [rtot x TP x n]
    lambAiIS = np.transpose(lambAiIS, (1, 0, 2))  # [TP x rtot x n]
    
    # Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
    Ci = S @ lambAiIS.reshape(TP, rtot*n)
    Ci = Ci.reshape(rtot, rtot, n)
    Ci = Ci + np.eye(rtot)[:, :, None]
    
    # invCiSRi := Ci\(S*Ri)
    SRi = S @ Ri.reshape(TP, -1)
    SRi = SRi.reshape(rtot, TP, n)
    invCiSRi = slow_backslash(Ci, SRi)
    
    # Quadratic term: Qtermi = invCiSRi' * vec(S) for each neuron
    Qtermi = np.array([invCiSRi[:, :, i].T @ vec(S.T) for i in range(n)])
    Qterm = (lamb**2) @ Qtermi
    
    # log-det term using cholesky
    cholCi = slow_chol(Ci)
    logdetterm = 2 * np.sum([np.sum(np.log(np.diag(cholCi[:, :, i]))) for i in range(n)])
    
    # Regularization term
    regterm = g * s.T @ s
    
    # Negative log likelihood
    negloglik = 0.5 * (-T * ni @ np.log(lamb) + logdetterm + zzi @ lamb - Qterm + regterm)
    
    return negloglik

def neg_log_lik_btdr_incomp_obs_uneqvar_lamb_only(lamb, S, Ai, Ri, zzi, r, ni, g):
    """Negative log likelihood for lambda only"""
    P = len(r)
    T = Ri.shape[0] // P
    TP = T * P
    n = len(ni)
    rtot = np.sum(r)
    
    # lambAiIS := lambi*kron(I,Ai)*S'
    # Compute for each row of S
    lambAiIS = np.zeros((TP, n, rtot))
    lambAi = Ai * lamb[None, None, :]  # [P x P x n]
    
    for p in range(rtot):
        # Get row p of S and reshape to [T x P]
        Sp_row = S[p, :].reshape(T, P)  # [T x P]
        # Repeat for all neurons: [T x P x n]
        Sp_row_rep = np.tile(Sp_row[:, :, None], (1, 1, n))
        # Multiply: [T x P x n] @ [P x P x n] -> [T x P x n]
        M = np.zeros((T, P, n))
        for i in range(n):
            M[:, :, i] = Sp_row_rep[:, :, i] @ lambAi[:, :, i]
        # Reshape: [T x P x n] -> [TP x n]
        lambAiIS[:, :, p] = M.reshape(TP, n)
    
    lambAiIS = np.transpose(lambAiIS, (0, 2, 1))  # [TP x rtot x n]
    
    # Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
    Ci = np.zeros((rtot, rtot, n))
    for i in range(n):
        Ci[:, :, i] = S @ lambAiIS[:, :, i] + np.eye(rtot)
    
    # invCiSRi := Ci\(S*Ri)
    SRi = S @ Ri.reshape(TP, -1)
    SRi = SRi.reshape(rtot, TP, n)
    invCiSRi = slow_backslash(Ci, SRi)
    
    # Quadratic term
    Qtermi = np.array([invCiSRi[:, :, i].T @ vec(S.T) for i in range(n)])
    Qterm = (lamb**2) @ Qtermi
    
    # log-det term using cholesky
    cholCi = slow_chol(Ci)
    logdetterm = 2 * np.sum([np.sum(np.log(np.diag(cholCi[:, :, i]))) for i in range(n)])
    
    # Regularization term (typically 0 for lambda-only)
    regterm = g * lamb.T @ lamb if g > 0 else 0
    
    # Negative log likelihood
    negloglik = 0.5 * (-T * ni @ np.log(lamb) + logdetterm + zzi @ lamb - Qterm + regterm)
    
    return negloglik

def ecm_regress_wrapper(r, init_regress_fun, em_regress_fun, Ybar):
    """
    ECM regression wrapper
    
    Args:
        r: ranks
        init_regress_fun: initialization function
        em_regress_fun: EM regression function
        Ybar: mean responses
    
    Returns:
        pars: parameters
    """
    T, n = Ybar.shape
    pars0 = init_regress_fun([r, min(n, T)])
    pars = em_regress_fun(r, np.concatenate([pars0[:n+np.sum(r)*T], vec(Ybar)]))
    return pars

def mmle_b(Ci, S, lamb, ni, xbari, Ybar, Xzetai, r):
    """
    MMLE estimation of b
    
    Args:
        Ci: covariance matrices
        S: basis functions
        lamb: precision parameters
        ni: trial counts
        xbari: mean regressors
        Ybar: mean responses
        Xzetai: cross terms
        r: ranks
    
    Returns:
        bhati: estimated coefficients
    """
    P = len(r)
    T, n = Ybar.shape
    bhati = np.zeros((T, n))
    rtot = np.sum(r)
    
    for ii in range(n):
        # construct XiS matrix
        XiS = np.zeros((T, rtot))
        for p in range(P):
            colind = np.arange(T * p, T * (p + 1))
            if p == 0:
                rowind = np.arange(r[0])
            else:
                rowind = np.arange(np.sum(r[:p]), np.sum(r[:p+1]))
            Sp = S[rowind, colind]
            XiS[:, rowind] = xbari[ii, p] * Sp.T
        
        # solve for bhati
        CIXS = np.linalg.solve(Ci[:, :, ii], XiS.T)
        A = np.eye(T) - lamb[ii] * ni[ii] * XiS @ CIXS
        ybarhat = lamb[ii] * CIXS.T @ S @ Xzetai[:, ii]
        bhati[:, ii] = np.linalg.solve(A, Ybar[:, ii] - ybarhat)
    
    return bhati

def make_bhat_data(histfile, Ai, Xzetai, r, b=None):
    """
    Make Bhat data from history file
    
    Args:
        histfile: history file path
        Ai: regressor covariances
        Xzetai: cross terms
        r: ranks
        b: specific iteration (optional)
    
    Returns:
        Bhat, r, What, Shat, lambhat
    """
    TP, n = Xzetai.shape
    
    # load history
    data = np.load(histfile + '.npz', allow_pickle=True)
    
    if b is not None:
        lambhat = data['parhist'][b][:n]
        shat = data['parhist'][b][n:]
    else:
        lambhat = data['parhist'][-1][:n]
        shat = data['parhist'][-1][n:]
    
    if r is None:
        r = data['rhist'][-1]
    
    P = Ai.shape[0]
    T = TP // P
    
    # reconstruct Shat
    Shat = []
    endind = 0
    for p in range(P):
        startind = endind
        endind = startind + T * r[p]
        Shat.append(shat[startind:endind].reshape(T, r[p]).T)
    
    Shatblock = sparse.block_diag(Shat).toarray()
    Wt = eb_post_W_uneqvar(Shatblock, lambhat, Ai, Xzetai, np.sum(r))
    
    Bhat = []
    What = []
    for p in range(P):
        colind = np.arange(np.sum(r[:p]), np.sum(r[:p+1]))
        What.append(Wt[colind, :].T)
        Bhat.append(What[p] @ Shat[p])
    
    return Bhat, r, What, Shat, lambhat

def eb_post_W_uneqvar(Shatblock, lambhat, Ai, Xzetai, rtot):
    """
    Empirical Bayes posterior for W with unequal variance
    
    Args:
        Shatblock: block diagonal basis functions
        lambhat: precision parameters
        Ai: regressor covariances
        Xzetai: cross terms
        rtot: total rank
    
    Returns:
        Wt: posterior weights
    """
    P, _, n = Ai.shape
    T = Shatblock.shape[1] // P
    
    I_T = sparse.eye(T)
    I_r = sparse.eye(rtot)
    
    SXIZi = Shatblock @ Xzetai
    Wt = np.zeros((rtot, n))
    Ci = np.zeros((rtot, rtot, n))
    
    for i in range(n):
        # Compute AiIS = kron(I_T, Ai(:,:,i)) * S'
        AiIS = kronmult([I_T, Ai[:, :, i]], Shatblock.T)
        
        # Compute Ci = lambi(i)*S*AiIS + I_r
        Ci[:, :, i] = lambhat[i] * Shatblock @ AiIS + I_r.toarray()
        
        # Solve for Wt: Ci \ (SXIZi * lambi)
        Wt[:, i] = np.linalg.solve(Ci[:, :, i], SXIZi[:, i]) * lambhat[i]
    
    return Wt

def btdr_aic_s_lamb_b_wrapper(pars, Ai, Xi, zetai, r, ni, g):
    """
    BTDR AIC wrapper
    
    Args:
        pars: parameters
        Ai: regressor covariances
        Xi: regressor matrices
        zetai: data
        r: ranks
        ni: trial counts
        g: regularization parameter
    
    Returns:
        AIC: Akaike Information Criterion
    """
    n = len(ni)
    T = zetai[0].shape[0]
    rtot = np.sum(r)
    newpars = pars[:n + rtot * T]
    b0 = pars[n + rtot * T:].reshape(T, n)
    
    _, zzi, Xzetai = ecm_suff_stat(zetai, Xi, b0)
    negloglik = neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(
        newpars[:n + rtot * T], Ai, Xzetai, zzi, r, ni, g)
    AIC = 2 * negloglik + 2 * len(pars)
    return AIC

def svd_reg_b_aic(Yn, allstim, pars, r):
    """
    SVD regression AIC
    
    Args:
        Yn: neural responses
        allstim: stimulus values
        pars: parameters
        r: ranks
    
    Returns:
        AIC: Akaike Information Criterion
    """
    n = len(Yn)
    T = Yn[0].shape[0]
    P = allstim[0].shape[1]
    Bhat = pars.reshape(n, -1)
    
    negloglik = 0
    lambhat = np.zeros(n)
    
    for ii in range(n):
        Bi = Bhat[ii, :]
        ri = vec(Yn[ii].T) - kronmult([sparse.eye(T), allstim[ii]], Bi)
        lambhat[ii] = allstim[ii].shape[0] * T / (ri.T @ ri)
        negloglik += (ri.T @ ri) * lambhat[ii] + Yn[ii].size * np.log(lambhat[ii])
    
    K = (n * P + T * P - np.sum(r)) * np.sum(r)  # number of free parameters
    AIC = negloglik + 2 * K
    return AIC

def neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(pars, Ai, Xzetai, zzi, r, ni, g):
    """Negative log likelihood for SVD AIC"""
    P = len(r)
    TP = Xzetai.shape[0]
    T = TP // P
    rtot = np.sum(r)
    n = len(ni)
    lambi = pars[:n]
    
    # Make block diagonal S matrix
    Shat = []
    endind = n
    for p in range(P):
        startind = endind
        endind = startind + T * r[p]
        Shat.append(pars[startind:endind].reshape(T, r[p]).T)
    S = sparse.block_diag(Shat).toarray()
    
    # lambAiIS := lambi*kron(I,Ai)*S'
    # MATLAB uses a loop, but we can use the simpler approach from S_only version
    # MATLAB: lambAi = bsxfun(@times,Ai,permute(repmat(lambi,1,P),[3 2 1]))
    lambAi = Ai * lambi[None, None, :]  # [P x P x n]
    lambAi = lambAi.reshape(P, P*n)  # [P x P*n]
    # MATLAB: S2 = reshape(S,[],P) - but S is rtot x TP, so this gives rtot*T x P
    # Actually, for this function, we use the loop approach from MATLAB
    # But let's use the simpler reshape approach which should be equivalent
    S2 = S.reshape(-1, P)  # Reshape S to have P columns: (rtot*TP/P) x P = rtot*T x P
    lambAiIS_temp = S2 @ lambAi  # [rtot*T x P] @ [P x P*n] = [rtot*T x P*n]
    lambAiIS_temp = lambAiIS_temp.reshape(rtot, TP, n)  # [rtot x TP x n]
    lambAiIS = np.transpose(lambAiIS_temp, (1, 0, 2))  # [TP x rtot x n]
    
    # Ci := lambi*S*kron(I,Ai)*S' + eye(rtot)
    Ci = np.zeros((rtot, rtot, n))
    for i in range(n):
        Ci[:, :, i] = S @ lambAiIS[:, :, i] + np.eye(rtot)
    
    # Quadratic term: S*Xzetai
    SXzetai = S @ Xzetai  # [rtot x n]
    invCiSXzetai = slow_backslash(Ci, SXzetai.reshape(rtot, 1, n))
    Qtermi = np.array([SXzetai[:, i].T @ invCiSXzetai[:, 0, i] for i in range(n)])
    Qterm = (lambi**2) @ Qtermi
    
    # log-det term using cholesky
    cholCi = slow_chol(Ci)
    logdetterm = 2 * np.sum([np.sum(np.log(np.diag(cholCi[:, :, i]))) for i in range(n)])
    
    # Regularization term
    regterm = g * pars[n:].T @ pars[n:] if g > 0 else 0
    
    # Negative log likelihood
    negloglik = 0.5 * (-T * ni @ np.log(lambi) + logdetterm + zzi @ lambi - Qterm + regterm)
    
    return negloglik

def q_tdr(par_new, par_old, Ai, Ri, zzi, r, ni, alpha):
    """
    Q function for TDR
    
    Args:
        par_new: new parameters
        par_old: old parameters
        Ai: regressor covariances
        Ri: population covariances
        zzi: squared norms
        r: ranks
        ni: trial counts
        alpha: regularization parameter
    
    Returns:
        Q: Q function value
        dQ: gradient (optional)
    """
    P = len(r)
    T = Ri.shape[0] // P
    TP = T * P
    rtot = np.sum(r)
    n = len(ni)
    
    lambinew = par_new[:n]
    snew = par_new[n:]
    lambiold = par_old[:n]
    sold = par_old[n:]
    
    # make block diagonal S matrix
    Snew = []
    Sold = []
    for p in range(P):
        start_idx = np.sum(r[:p]) * T
        end_idx = np.sum(r[:p+1]) * T
        Snew.append(snew[start_idx:end_idx].reshape(r[p], T))
        Sold.append(sold[start_idx:end_idx].reshape(r[p], T))
    
    Snew = sparse.block_diag(Snew).toarray()
    Sold = sparse.block_diag(Sold).toarray()
    
    # make new feature covariance
    lambnewAi = Ai * lambinew[None, None, :]
    lambnewAi = lambnewAi.reshape(P, P*n)
    S2 = Snew.reshape(-1, P)
    lambAiISnew = np.transpose(S2 @ lambnewAi.reshape(-1, n), (1, 0, 2))
    lambAiISnew = lambAiISnew.reshape(rtot, TP, n)
    lambiSAiSnew = Snew @ lambAiISnew.reshape(TP, rtot*n)
    lambiSAiSnew = lambiSAiSnew.reshape(rtot, rtot, n)
    Cinew = lambiSAiSnew + np.eye(rtot)[:, :, None]
    
    # make old feature covariance
    lamboldAi = Ai * lambiold[None, None, :]
    lamboldAi = lamboldAi.reshape(P, P*n)
    lambAiISold = np.transpose(S2 @ lamboldAi.reshape(-1, n), (1, 0, 2))
    lambAiISold = lambAiISold.reshape(rtot, TP, n)
    lambiSAiSold = Sold @ lambAiISold.reshape(TP, rtot*n)
    lambiSAiSold = lambiSAiSold.reshape(rtot, rtot, n)
    Ciold = lambiSAiSold + np.eye(rtot)[:, :, None]
    
    # compute Q function
    SoldRi = Sold @ Ri.reshape(TP, -1)
    SoldRi = SoldRi.reshape(rtot, TP, n)
    BiSoldRi = np.array([np.linalg.solve(Ciold[:, :, i], SoldRi[:, :, i]) for i in range(n)]).T
    BiSoldRi = BiSoldRi.transpose(1, 0, 2)
    
    BiCi = np.array([np.linalg.solve(Ciold[:, :, i], Cinew[:, :, i]) for i in range(n)]).T
    BiCi = BiCi.transpose(1, 0, 2)
    
    RSBCB = np.array([BiSoldRi[:, :, i].T @ BiCi[:, :, i] for i in range(n)]).T
    RSBCB = RSBCB.transpose(1, 0, 2)
    
    Quadi = RSBCB.reshape(n, rtot*TP) @ vec(Sold.T)
    
    Quad1 = zzi @ lambinew
    Quad2 = 2 * (lambinew * lambiold) @ BiSoldRi.reshape(n, rtot*TP) @ vec(Snew)
    Quad3 = (lambiold**2) @ Quadi
    Quad4 = np.sum([np.trace(BiCi[:, :, i]) for i in range(n)])
    
    Q = T * ni @ np.log(lambinew) - Quad1 + Quad2 - Quad3 - Quad4 - alpha * (snew - sold) @ (snew - sold)
    
    return Q

def keep_active_s(S, r):
    """
    Keep active S components
    
    Args:
        S: S matrix
        r: ranks
    
    Returns:
        s: active components
    """
    P = len(r)
    TP = S.shape[1]
    T = TP // P
    
    s = []
    for p in range(P):
        colind = np.arange(T * p, T * (p + 1))
        if p == 0:
            rowind = np.arange(r[0])
        else:
            rowind = np.arange(np.sum(r[:p]), np.sum(r[:p+1]))
        Sp = S[rowind, colind]
        s.extend(vec(Sp.T))
    
    return np.array(s)

def ecm_tdr(stopmode, stopcrit, pars0, Ai, Xi, r, ni, zetai, xbari, Ybar, Xzetai0):
    """
    ECM TDR estimation - Full implementation matching MATLAB ECMEtdr
    
    Args:
        stopmode: stopping mode ('steps' or 'converge')
        stopcrit: stopping criterion
        pars0: initial parameters [lamb; s; b]
        Ai: regressor covariances [P x P x n]
        Xi: regressor matrices (list of n arrays)
        r: ranks (list of P integers)
        ni: trial counts [n]
        zetai: data (list of n arrays, each T x ntrials)
        xbari: mean regressors [n x P]
        Ybar: mean responses [T x n]
        Xzetai0: cross terms [TP x n]
    
    Returns:
        parhat, Q, parerr, nll
    """
    maxsteps = 100
    
    # Unpack parameters
    n = len(ni)
    T = zetai[0].shape[0]
    P = len(r)
    TP = T * P
    rtot = np.sum(r)
    lambiold = pars0[:n].copy()
    olds = pars0[n:n+rtot*T].copy()
    bold = pars0[n+rtot*T:].reshape(T, n).copy()
    
    # For first step, set new and old pars equal
    lambinew = lambiold.copy()
    news = olds.copy()
    bnew = bold.copy()
    Ri, zzi, Xzetai = ecm_suff_stat(zetai, Xi, bold)
    
    if stopmode == 'steps':
        stopcrit = stopcrit + 1
    
    # Initialize Q and nll
    pars0q = np.concatenate([lambiold, olds])
    Q = [q_tdr(pars0q, pars0q, Ai, Ri, zzi, r, ni, 0)]
    nll = [neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(pars0, Ai, Xzetai, zzi, r, ni, 0)]
    
    k = 2
    stopvar = True
    parerr = []
    
    while stopvar and k <= maxsteps:
        bold = bnew.copy()
        bold0 = bnew.copy()
        lambiold0 = lambinew.copy()
        olds0 = news.copy()
        
        # Reset sufficient stats using current estimate of b
        Ri, zzi, Xzetai = ecm_suff_stat(zetai, Xi, bold)
        
        # ------------------ M-step for lambda------------------
        # Build Snew and Sold from news
        Snew_list = []
        for p in range(P):
            start_idx = np.sum(r[:p]) * T
            end_idx = np.sum(r[:p+1]) * T
            Snew_list.append(news[start_idx:end_idx].reshape(r[p], T))
        Snew = sparse.block_diag(Snew_list).toarray()
        Sold = Snew.copy()
        
        # Old feature covariance: Ci := lambi*S*kron(Ai,I)*S' + eye(rtot)
        S2 = Sold.reshape(-1, P)  # Reshape to have P columns
        Ai_reshaped = Ai.reshape(P, P*n)  # [P x P*n]
        AiISold_temp = S2 @ Ai_reshaped  # [rtot*T x P] @ [P x P*n] = [rtot*T x P*n]
        AiISold_temp = AiISold_temp.reshape(rtot, TP, n)
        AiISold = np.transpose(AiISold_temp, (1, 0, 2))  # [TP x rtot x n]
        SAiISold = Sold @ AiISold.reshape(TP, rtot*n)
        SAiISold = SAiISold.reshape(rtot, rtot, n)
        lambiSAiSold = SAiISold * lambiold[None, None, :]  # [rtot x rtot x n]
        Ciold = lambiSAiSold + np.eye(rtot)[:, :, None]
        
        # BiSRi := Ciold\(S*Ri)
        SoldRi = Sold @ Ri.reshape(TP, -1)
        SoldRi = SoldRi.reshape(rtot, TP, n)
        BiSoldRi = slow_backslash(Ciold, SoldRi)
        
        # Compute g1i, g2i, g3i for lambda update
        # g1i = 2*lambiold.*(reshape(permute(BiSoldRi,[3,1,2]),n,rtot*TP)*vec(Snew))
        BiSoldRi_perm = np.transpose(BiSoldRi, (2, 0, 1))  # [n x rtot x TP]
        BiSoldRi_reshaped = BiSoldRi_perm.reshape(n, rtot*TP)
        g1i = 2 * lambiold * (BiSoldRi_reshaped @ vec(Snew.T))
        
        # Compute SnewAiSnew
        AiISnew_temp = S2 @ Ai_reshaped
        AiISnew_temp = AiISnew_temp.reshape(rtot, TP, n)
        AiISnew = np.transpose(AiISnew_temp, (1, 0, 2))  # [TP x rtot x n]
        SnewAiSnew = Snew @ AiISnew.reshape(TP, rtot*n)
        SnewAiSnew = SnewAiSnew.reshape(rtot, rtot, n)
        BiSAS = slow_backslash(Ciold, SnewAiSnew)
        
        # g2i = trace(BiSAS) for each neuron
        g2i = np.array([np.trace(BiSAS[:, :, ii]) for ii in range(n)])
        
        # g3i computation
        # MATLAB: RSBSASB = mmx_mkl_single('mult',permute(BiSoldRi,[2,1,3]),permute(BiSAS,[2,1,3]))
        # BiSoldRi is [rtot x TP x n], permute([2,1,3]) -> [TP x rtot x n]
        # BiSAS is [rtot x rtot x n], permute([2,1,3]) -> [rtot x rtot x n] (no change)
        BiSoldRi_perm2 = np.transpose(BiSoldRi, (1, 0, 2))  # [TP x rtot x n]
        BiSAS_perm = np.transpose(BiSAS, (1, 0, 2))  # [rtot x rtot x n]
        RSBSASB = slow_mult(BiSoldRi_perm2, BiSAS_perm)  # [TP x rtot x n] @ [rtot x rtot x n] -> [TP x rtot x n]
        # MATLAB: g3i = reshape(permute(RSBSASB,[3,2,1]),n,rtot*TP)*sparse(vec(Sold))
        RSBSASB_perm = np.transpose(RSBSASB, (2, 1, 0))  # [n x rtot x TP]
        RSBSASB_reshaped = RSBSASB_perm.reshape(n, rtot*TP)
        g3i = RSBSASB_reshaped @ vec(Sold.T)  # [n x rtot*TP] @ [rtot*TP x 1] -> [n]
        g3i = g3i * lambiold**2
        
        # Update lambda
        lambinew = T * ni / (zzi - g1i + g2i + g3i)
        lambiold = lambinew.copy()
        
        # ------------------ M-step for S------------------
        # Remake old feature covariance with new lambda
        lambiSAiSold = SAiISold * lambiold[None, None, :]
        Ciold = lambiSAiSold + np.eye(rtot)[:, :, None]
        
        # Compute M0 and m0
        lambnewilamboldBiSoldRi = BiSoldRi * (lambinew * lambiold)[None, None, :]
        M0 = np.sum(lambnewilamboldBiSoldRi, axis=2)  # Sum over neurons
        m0 = keep_active_s(M0, r)
        
        # Compute SRSB_old
        BiSoldRi_perm3 = np.transpose(BiSoldRi, (1, 0, 2))  # [TP x rtot x n]
        SRSB_old = Sold @ BiSoldRi_perm3.reshape(TP, -1)
        SRSB_old = SRSB_old.reshape(rtot, rtot, n)
        lamb2oldSRSB_old = SRSB_old * (lambiold**2)[None, None, :]
        Gi_input = lamb2oldSRSB_old + np.eye(rtot)[:, :, None]
        Gi = slow_backslash(Ciold, Gi_input)
        
        # Build Gammap matrices
        # MATLAB: SSnew = mat2cell(reshape(news,T,rtot)',r,T)
        # reshape(news,T,rtot) gives T x rtot, transpose gives rtot x T
        # Then split into cells of size r[p] x T each
        news_reshaped_temp = news.reshape(T, rtot).T  # [rtot x T]
        SSnew_list = []
        for p in range(P):
            if p == 0:
                row_start = 0
            else:
                row_start = np.sum(r[:p])
            row_end = np.sum(r[:p+1])
            SSnew_list.append(news_reshaped_temp[row_start:row_end, :])  # [r[p] x T]
        SSnew = np.vstack(SSnew_list)  # [rtot x T] - same as news_reshaped_temp
        
        Gammap = []
        for p in range(P):
            gammapq = []
            for q in range(P):
                # MATLAB: lambda_ipq = bsxfun(@times,Ai(p,q,:),permute(lambinew,[3,2,1]))
                lambda_ipq = Ai[p, q, :] * lambinew  # [n]
                if q == 0:
                    Gind = np.arange(r[0])
                else:
                    Gind = np.arange(np.sum(r[:q]), np.sum(r[:q+1]))
                # MATLAB: aGi = bsxfun(@times,Gi(:,Gind,:),lambda_ipq)
                aGi = Gi[:, Gind, :] * lambda_ipq[None, None, :]  # [rtot x r[q] x n]
                # MATLAB: gammapq{q} = sum(aGi,3)
                gammapq.append(np.sum(aGi, axis=2))  # Sum over neurons: [rtot x r[q]]
            # MATLAB: Gammap{p} = cat(2,gammapq{:})
            Gammap.append(np.hstack(gammapq))  # [rtot x rtot]
        
        # Build GG matrix
        # MATLAB: G = cat(1,Gammap{:}) - but then builds GG selectively
        GG = np.zeros((rtot, rtot))
        for p in range(P):
            G = Gammap[p]
            if p == 0:
                # MATLAB: GG(1:r(1),:) = G(1:r(1),:); GG(:,1:r(1)) = G(1:r(1),:)'
                GG[:r[0], :] = G[:r[0], :]
                GG[:, :r[0]] = G[:r[0], :].T
            else:
                startind = np.sum(r[:p])
                endind = np.sum(r[:p+1])
                # MATLAB: GG(startind:endind,startind:rtot) = G(startind:endind,startind:rtot)
                GG[startind:endind, startind:] = G[startind:endind, startind:]
                # MATLAB: GG(startind:rtot,startind:endind) = G(startind:endind,startind:rtot)'
                GG[startind:, startind:endind] = G[startind:endind, startind:].T
        
        # Solve for news
        # MATLAB: news = GG\reshape(m0,T,rtot)'
        m0_reshaped = m0.reshape(T, rtot).T  # [rtot x T]
        news_reshaped = np.linalg.solve(GG, m0_reshaped)  # [rtot x T]
        # MATLAB: news = vec(news')
        news = vec(news_reshaped.T)  # Vectorize: [rtot*T]
        
        # ------------------------------------------------------
        # ---------- MMLE of condition-independent term----------
        # Rebuild S from news (the updated S)
        S_list = []
        for p in range(P):
            start_idx = np.sum(r[:p]) * T
            end_idx = np.sum(r[:p+1]) * T
            S_list.append(news[start_idx:end_idx].reshape(r[p], T))
        S = sparse.block_diag(S_list).toarray()
        
        # Recompute Ci with new S and old lambda
        # MATLAB uses Sold here, but Sold was set to Snew earlier, and now we have the new S
        # Actually, looking at MATLAB: it rebuilds S from news, then uses Sold in the computation
        # This seems like it might be using the old S for the covariance computation
        # But let's match MATLAB exactly - it uses Sold which was the S from the start of the iteration
        S2 = S.reshape(-1, P)  # Use new S for S2
        AiISold_temp = S2 @ Ai_reshaped
        AiISold_temp = AiISold_temp.reshape(rtot, TP, n)
        AiISold = np.transpose(AiISold_temp, (1, 0, 2))
        # MATLAB uses Sold here - which is the S from the start of M-step
        SAiISold = Sold @ AiISold.reshape(TP, rtot*n)
        SAiISold = SAiISold.reshape(rtot, rtot, n)
        lambiSAiSold = SAiISold * lambiold[None, None, :]
        Ci = lambiSAiSold + np.eye(rtot)[:, :, None]
        # Use new S in MMLE_b
        bnew = mmle_b(Ci, S, lambiold, ni, xbari, Ybar, Xzetai0, r)
        
        # Convergence checks
        newpars = np.concatenate([lambinew, news, vec(bnew)])
        oldpars = np.concatenate([lambiold0, olds0, vec(bold0)])
        parerr.append(np.max((newpars - oldpars)**2 / (oldpars**2 + 1e-10)))
        
        # Evaluate Q and nll
        Q.append(q_tdr(newpars[:n+rtot*T], oldpars[:n+rtot*T], Ai, Ri, zzi, r, ni, 0))
        Ri_new, zzi_new, Xzetai_new = ecm_suff_stat(zetai, Xi, bnew)
        nll.append(neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(
            newpars[:n+rtot*T], Ai, Xzetai_new, zzi_new, r, ni, 0))
        
        # Check stopping criteria
        if stopmode == 'steps':
            if k >= stopcrit:
                stopvar = False
        elif stopmode == 'converge':
            if (len(parerr) > 0 and parerr[-1] < stopcrit) or k >= maxsteps:
                stopvar = False
        
        k += 1
    
    parhat = np.concatenate([lambinew, news, vec(bnew)])
    return parhat, Q, parerr, nll

def mmle_coord_ascent_wrapper(em_pars_fun, mmle_pars_fun, r, n, T):
    """
    MMLE coordinate ascent wrapper
    
    Args:
        em_pars_fun: EM parameters function
        mmle_pars_fun: MMLE parameters function
        r: ranks
        n: number of neurons
        T: number of time points
    
    Returns:
        pars: parameters
    """
    rtot = np.sum(r)
    em_pars = em_pars_fun(r)
    lamb0 = em_pars[:n]
    s0 = em_pars[n:n+rtot*T]
    bhat0 = em_pars[n+rtot*T:].reshape(T, n)
    
    lambhat, shat, _, _, bhat = mmle_pars_fun(lamb0, s0, bhat0, r)
    pars = np.concatenate([lambhat, shat, vec(bhat)])
    
    return pars

def est_rank_greedily(obj_fun, est_fun, r0, maxrank, stepthresh, histfile):
    """
    Greedy rank estimation
    
    Args:
        obj_fun: objective function
        est_fun: estimation function
        r0: initial ranks
        maxrank: maximum rank
        stepthresh: step threshold
        histfile: history file
    
    Returns:
        rest, rhist, FunHist, parhist
    """
    # check dimensions
    if r0.ndim == 1:
        rest = r0.copy()
        P = len(r0)
    else:
        rest = r0.T.copy()
        P = r0.shape[0]
    
    rhist = [rest.copy()]
    if stepthresh is None:
        stepthresh = 0
    
    # initialize
    parhat0 = est_fun(rest)
    Obj0 = obj_fun(parhat0, rest)
    FunHist = [Obj0]
    iters = 0
    parhist = []
    
    # iterate until stopping criteria met
    while np.all(rest <= maxrank):
        print(f'Current estimate: r = {rest}, AIC = {Obj0:.3f}')
        
        # check step forward in each dimension
        Obj = np.zeros(P)
        indmove = np.where(rest < maxrank)[0]
        
        for p in range(P):
            if rest[p] < maxrank:
                rtest = rest.copy()
                rtest[p] += 1
                parhat_p = est_fun(rtest)
                Obj[p] = obj_fun(parhat_p, rtest)
        
        # decide whether to step
        dObj = Obj - Obj0
        if np.all(dObj[indmove] > stepthresh):
            print('Stop estimation: Insufficient improvement in log likelihood.')
            break
        
        # increment in direction of greatest decrease
        mindiff = np.min(dObj[indmove])
        indmin = np.where(dObj == mindiff)[0]
        if len(indmin) > 1:
            indmin = np.random.choice(indmin)
        else:
            indmin = indmin[0]
        
        rest[indmin] += 1
        Obj0 = Obj[indmin]
        FunHist.append(Obj0)
        rhist.append(rest.copy())
        
        iters += 1
        parhist.append(parhat_p)
        
        # save history
        if histfile:
            np.savez(histfile, rhist=rhist, FunHist=FunHist, parhist=parhist)
    
    # save results if no change
    if iters == 0:
        parhist = [parhat0]
        FunHist = [Obj0]
        rhist = [r0]
        if histfile:
            np.savez(histfile, rhist=rhist, FunHist=FunHist, parhist=parhist)
    
    return rest, rhist, FunHist, parhist
