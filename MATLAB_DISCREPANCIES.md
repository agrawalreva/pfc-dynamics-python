## Summary by Priority

### CRITICAL (Must Fix) - FIXED:
1. **q_tdr**: Bug in lambAiIS computation for old S (uses S2 from Snew instead of Sold) - **FIXED**
2. **q_tdr**: Missing gradient computation (needed for optimization) - **FIXED**
3. **neg_log_lik functions**: Missing gradients (needed for optimization) - **FIXED**

### HIGH PRIORITY (Should Fix) - FIXED:
4. **eb_post_W_uneqvar**: Missing Ci return value - **FIXED**
5. **est_pars_coord_ascent_lambi_s_b**: Different optimization algorithms - **IMPROVED** (now uses L-BFGS-B for S, L-BFGS-B with gradient for lambda, closer to MATLAB's minFunc/fminunc)
6. **neg_log_lik_btdr_incomp_obs_uneqvar_s_only**: Missing Hessian - **FIXED** (Full Hessian computation implemented with `quickkron_rxr_PTxPT` and `quickkron_PTxr_rxPT` helper functions)

### MEDIUM PRIORITY (Nice to Have):
7. **make_bhat_data**: Verify cell array vs list handling
8. **ecm_regress_wrapper**: Verify parameter handling matches exactly
9. **mmle_coord_ascent_wrapper**: Verify signature matches

### LOW PRIORITY (Documentation/Verification):
10. All indexing conversions (0-based vs 1-based) - verify edge cases
11. Sparse matrix handling differences
12. Array broadcasting verification

