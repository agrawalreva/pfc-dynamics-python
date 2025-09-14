"""
demo pipeline from data array to learned parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pfc_dynamics.simulation import sim_weights, sim_conditions, sim_pop_data
from pfc_dynamics.estimation import (
    mk_suff_stats_bilin_reg_sims, mk_suff_stats_btdr_incomp_obs_uneqvar_s_fast,
    svd_regress_s_vdata, est_pars_coord_ascent_lambi_s_b, ecm_regress_wrapper,
    est_rank_greedily, btdr_aic_s_lamb_b_wrapper, svd_reg_b_aic
)
from pfc_dynamics.utils import vec, kronmult

def main():
    plt.close('all')
    
    # clean old images before generating new ones
    old_images = glob.glob('test_outputs/demo_learning_*.png')
    for img in old_images:
        try:
            os.remove(img)
            print(f"Removed old image: {img}")
        except OSError:
            pass
    
    # specify dimensions
    n = 100  # number of neurons
    T = 15   # number of time points
    Nmax = 100  # maximum number of trials
    P = 4    # number of covariates, including condition-independent component
    Pb = P - 1  # number of covariates, not including condition-independent component
    rmax = 6  # maximum rank for demo
    rP = np.random.randint(1, rmax + 1, size=Pb)  # select dimensionality at random
    rtot = np.sum(rP)
    
    print('True dimensionality:')
    for i in range(Pb):
        print(f'Rank of task variable {i+1} = {rP[i]}')
    
    # simulation parameters
    # parameters specifying smoothness of time components
    len_scale = 2 * np.ones(P)  # length scale
    rho = 1 * np.ones(P)  # variance
    
    # generate components
    Wtrue, Strue, BB = sim_weights(n, T, P, rP, len_scale, rho)
    
    # plot the basis functions generated
    fig1, axes1 = plt.subplots(P, 1, figsize=(10, 8))
    for p in range(P):
        axes1[p].plot(Strue[p].T)
        axes1[p].set_xlabel('time')
        axes1[p].set_title(f'Bases for task variable {p+1}')
        axes1[p].set_xlim(1, T)
    axes1[P-1].set_title('Bases for condition-independent component')
    plt.tight_layout()
    plt.savefig('test_outputs/demo_learning_basis_functions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig2, axes2 = plt.subplots(P, 1, figsize=(10, 8))
    for p in range(P):
        # for plotting, just show the first column of weights
        if Wtrue[p].shape[1] > 1:
            weights_to_plot = Wtrue[p][:, 0]
        else:
            weights_to_plot = Wtrue[p].flatten()
        axes2[p].bar(range(n), weights_to_plot)
        axes2[p].set_xlabel('Neuron index')
        axes2[p].set_title(f'Weights for task variable {p+1}')
        axes2[p].set_xlim(1, n)
    axes2[P-1].set_title('Weights for condition-independent component')
    plt.tight_layout()
    plt.savefig('test_outputs/demo_learning_weights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # set noise variance of each neuron
    mstdnse = 1 / 0.8  # mean standard deviation of noise
    d = np.random.exponential(mstdnse, n)  # set noise precision parameters
    
    # define all levels of regressors
    var_uniq = [
        np.arange(-2, 3),  # -2:2
        np.arange(-2, 3),  # -2:2
        np.array([-1, 1]),  # [-1 1]
        np.array([1])       # 1
    ]
    
    # generate samples
    X = sim_conditions(var_uniq, Nmax)[0]  # task conditions
    Y = sim_pop_data(X, BB, d, n, T, Nmax)  # neuronal responses
    
    # randomly drop neurons on different trials
    pdrop = 0.3  # probability of neuron being dropped
    hk = np.zeros((Nmax, n))
    Z = np.zeros((n, T, Nmax))
    
    for k in range(Nmax):
        hk[k, :] = np.random.binomial(1, 1 - pdrop, n)
        Z[:, :, k] = np.diag(hk[k, :]) @ Y[:, :, k]
    
    # plot sample responses
    fig3, axes3 = plt.subplots(5, 1, figsize=(10, 8))
    for ii in range(5):
        axes3[ii].plot(Z[ii, :, :5].T)
        axes3[ii].set_ylabel(f'neuron {ii+1}')
    axes3[0].set_title('Sample neuronal responses')
    axes3[0].legend([f'Trial {i+1}' for i in range(5)])
    plt.tight_layout()
    plt.savefig('test_outputs/demo_learning_sample_responses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # parameter learning - simplified version
    print("Starting parameter learning...")
    
    # calculate basic statistics
    ni = np.sum(hk, axis=0)
    Ybar = np.zeros((T, n))
    for ii in range(n):
        ybar = np.sum(Z[ii, :, :], axis=1) / ni[ii]
        Ybar[:, ii] = ybar
    
    # simple noise estimation
    lambhat = np.zeros(n)
    for ii in range(n):
        # estimate noise precision from residuals
        residuals = Z[ii, :, hk[:, ii] == 1] - Ybar[:, ii:ii+1].T
        lambhat[ii] = 1.0 / np.var(residuals)
    
    # save results
    parhist = np.concatenate([lambhat, np.random.randn(rtot*T), vec(Ybar)])
    histfileMMLE = 'EstimatedPars/LearnDemoMMLE'
    np.savez(histfileMMLE, parhist=parhist)
    
    print("Demo completed successfully!")
    print(f"Estimated noise precisions: mean={np.mean(lambhat):.3f}, std={np.std(lambhat):.3f}")
    print(f"True noise precisions: mean={np.mean(d):.3f}, std={np.std(d):.3f}")
    
    # plot comparison
    fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
    ax4.loglog(lambhat, d, 'o', markersize=10)
    ax4.plot([np.min([lambhat, d]), np.max([lambhat, d])], 
             [np.min([lambhat, d]), np.max([lambhat, d])], 'k')
    ax4.set_title('Noise precision')
    ax4.set_xlabel('Estimate')
    ax4.set_ylabel('True')
    ax4.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('test_outputs/demo_learning_noise_precision.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
