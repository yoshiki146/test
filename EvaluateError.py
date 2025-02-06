#### Chopra and Ziemba (1993) 
#### The Effect of Means, Variance, and Covariance on Optimal Portfolio Choice
#### Journal of Portfolio Management

import numpy as np

def mean_err(ret_mean, error_intensity):
    n_assets = len(ret_mean)
    err_z = np.random.normal(size=n_assets)
    ret_mean_err = ret_mean * (1 + error_intensity * err_z)
    return ret_mean_err 

def var_err(vcov, error_intensity, **kwargs):
    n_assets = vcov.shape[0]
    err_z = np.random.normal(size=n_assets)
    vcov_var_err = vcov.copy()
    np.fill_diagonal(vcov_var_err.values, np.diag(vcov)*(1+error_intensity*err_z))
    return vcov_var_err

def cov_err(vcov, error_intensity, **kwargs):
    error_type = kwargs.get("error_type", "all")
    block = kwargs.get("block", None)
    n_assets = vcov.shape[0]
    err_z = np.random.normal(size=n_assets*(n_assets-1) // 2)
    vcov_cov_err = vcov.copy()
    upper_tri_indices = np.triu_indices(n_assets, k=1)
    if error_type == "all":
        cov_err = vcov.values[upper_tri_indices] * (1+error_intensity*err_z) # introduce error into all cov element
    elif error_type == "intra":
        cov_err = vcov.values[upper_tri_indices] * (1+error_intensity*err_z*(block.values[upper_tri_indices]==1)) # introduce error into intra-class element
    else:
        cov_err = vcov.values[upper_tri_indices] * (1+error_intensity*err_z*(block.values[upper_tri_indices]==0)) # introduce error into inter-class element
    vcov_cov_err.values[upper_tri_indices] = cov_err
    vcov_cov_err.values[(upper_tri_indices[1], upper_tri_indices[0])] = cov_err
    return vcov_cov_err

def err_positive_definite(fn, vcov, err_k, **kwargs):
    def generate_vcov_err():
        return fn(vcov, err_k, **kwargs)
    vcov_err = generate_vcov_err()
    evals = np.linalg.eigvals(vcov_err)
    while np.any(evals <= 0): # loop until vcov_err is positive definite
        vcov_err = generate_vcov_err()
        evals = np.linalg.eigvals(vcov_err)
    return vcov_err

def cash_equivalent_loss(wgt, wgt_err, eret, vcov, risk_tolerance):
    def cash_equivalent_value(w):
        return w.T @ eret - (1/risk_tolerance) * (w.T @ vcov @ w)
    ce_0 = cash_equivalent_value(wgt)
    ce_x = cash_equivalent_value(wgt_err)
    return (ce_0 - ce_x)*100 / abs(ce_0)