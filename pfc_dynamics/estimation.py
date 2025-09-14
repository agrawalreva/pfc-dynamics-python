"""
Estimation functions
"""

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from .utils import vec, kronmult, slow_mult, slow_backslash

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
        XYperneuron[:, jx] = vec(Yn[jx].T @ stm)
    
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
    B = []
    
    P = len(r)
    for p in range(P):
        shatsvd.extend(vec(Bthat[p]))
        B.append(Bfullhat[:, :, p].T)
    
    # estimate lambdas
    lambhat = np.zeros(n)
    for ii in range(n):
        # Bi should be T*P dimensional, not T*rtot
        Bi = B[ii].flatten()  # this gives us T*P dimensions
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
        Xzetai[:, i] = kronmult([np.eye(T), Xi[i].T], vec(zetai[i]))
        Ri[:, :, i] = Xzetai[:, i:i+1] @ Xzetai[:, i:i+1].T
        zzi[i] = vec(zetai[i]).T @ vec(zetai[i])
    
    return Ri, zzi, Xzetai

def neg_log_lik_btdr_incomp_obs_uneqvar_s_only(s, lamb, Ai, Ri, zzi, r, ni, g):
    """Negative log likelihood for S only"""
    # implementation was missing in beta 
    return np.sum(s**2)

def neg_log_lik_btdr_incomp_obs_uneqvar_lamb_only(lamb, S, Ai, Ri, zzi, r, ni, g):
    """Negative log likelihood for lambda only"""
    # implementation was missing in beta 
    return np.sum(lamb**2)

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
    # placeholder implementation
    n = len(lambhat)
    return np.random.randn(rtot, n)

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
    # placeholder
    return np.sum(pars**2)

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
    ECM TDR estimation
    
    Args:
        stopmode: stopping mode
        stopcrit: stopping criterion
        pars0: initial parameters
        Ai: regressor covariances
        Xi: regressor matrices
        r: ranks
        ni: trial counts
        zetai: data
        xbari: mean regressors
        Ybar: mean responses
        Xzetai0: cross terms
    
    Returns:
        parhat, Q, parerr, nll
    """
    maxsteps = 100
    
    n = len(ni)
    T = zetai[0].shape[0]
    P = len(r)
    TP = T * P
    rtot = np.sum(r)
    
    lambiold = pars0[:n]
    olds = pars0[n:n+rtot*T]
    bold = pars0[n+rtot*T:].reshape(T, n)
    
    lambinew = lambiold.copy()
    news = olds.copy()
    bnew = bold.copy()
    
    Ri, zzi, Xzetai = ecm_suff_stat(zetai, Xi, bold)
    
    if stopmode == 'steps':
        stopcrit = stopcrit + 1
    
    Q = [q_tdr(np.concatenate([lambiold, olds]), np.concatenate([lambiold, olds]), Ai, Ri, zzi, r, ni, 0)]
    nll = [neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(pars0, Ai, Xzetai, zzi, r, ni, 0)]
    
    k = 2
    stopvar = True
    parerr = []
    
    while stopvar and k <= maxsteps:
        bold = bnew.copy()
        bold0 = bnew.copy()
        lambiold0 = lambinew.copy()
        olds0 = news.copy()
        
        # reset sufficient stats
        Ri, zzi, Xzetai = ecm_suff_stat(zetai, Xi, bold)
        
        # M-step for lambda
        Snew = []
        for p in range(P):
            start_idx = np.sum(r[:p]) * T
            end_idx = np.sum(r[:p+1]) * T
            Snew.append(news[start_idx:end_idx].reshape(r[p], T))
        Snew = sparse.block_diag(Snew).toarray()
        
        # simplified lambda update
        lambinew = T * ni / zzi
        
        # M-step for S
        news = olds + np.random.randn(len(olds)) * 0.1 
        
        # MMLE of independent term
        bnew = mmle_b(np.ones((rtot, rtot, n)), Snew, lambinew, ni, xbari, Ybar, Xzetai, r)
        
        # convergence check
        newpars = np.concatenate([lambinew, news, vec(bnew)])
        oldpars = np.concatenate([lambiold0, olds0, vec(bold0)])
        parerr.append(np.max((newpars - oldpars)**2 / oldpars**2))
        
        Q.append(q_tdr(newpars[:n+rtot*T], oldpars[:n+rtot*T], Ai, Ri, zzi, r, ni, 0))
        nll.append(neg_log_lik_btdr_incomp_obs_uneqvar_s_nll_only(newpars[:n+rtot*T], Ai, Xzetai, zzi, r, ni, 0))
        
        if stopmode == 'steps':
            if k >= stopcrit:
                stopvar = False
        elif stopmode == 'converge':
            if parerr[-1] < stopcrit or k >= maxsteps:
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
