import pandas as pd
import numpy as np
from hmmlearn import hmm
import statsmodels.api as sm

def hmm_states(dat, n_states, zscore=True, col=None, random=42):
    np.random.seed(seed=random)
    def hmm_seed(seed):
        hmm_model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", 
                                    n_iter=10000, random_state=seed, tol=1e-3)
        hmm_model.fit(dat_values)
        hmm_status = hmm_model.monitor_.converged
        return hmm_model, hmm_status
    
    if col is not None:
        dat = dat.loc[:, col]
    if zscore:
        dat = dat.apply(lambda x: (x-x.mean())/x.std(), axis=0) if isinstance(dat, pd.DataFrame) else (dat-dat.mean())/dat.std()
    n_asset = 1 if isinstance(dat, pd.Series) else dat.shape[1] 
    dat_values = dat.values.reshape(-1, n_asset)
    hmm_model, hmm_status = hmm_seed(np.random.randint(100))
    while hmm_status is not True:
        hmm_model, hmm_status = hmm_seed(np.random.randint(100))
    return pd.DataFrame(hmm_model.predict_proba(dat_values), index=dat.index)


def _factor_model_decomp(asset_return, factor_return):
    f_loads_const = {}

    asset_cov = asset_return.cov()
    factor_return = factor_return.reindex(asset_return.index)
    factor_return_const = sm.add_constant(factor_return)
    factor_cov = factor_return.cov() if isinstance(factor_return, pd.DataFrame) else factor_return.var() # no covariance for single factor
    
    for i_asset in asset_return.columns:
        ret_i = asset_return.loc[:, i_asset]
        model_i = sm.OLS(ret_i, factor_return_const)
        res_i = model_i.fit()
        f_loads_const[i_asset] = res_i.params
    df_f_loads_const = pd.concat(f_loads_const, axis=1)
    df_f_loads = df_f_loads_const.loc[factor_return.columns,:] if isinstance(factor_return, pd.DataFrame) else df_f_loads_const.loc[factor_return.name,:].to_frame().T
    const = df_f_loads_const.loc["const",:]
    if isinstance(factor_cov, pd.DataFrame):
        cov_replicate = pd.DataFrame(df_f_loads.values.T @ factor_cov.values @ df_f_loads.values, 
                                     index=asset_return.columns, columns=asset_return.columns)
    else: # single factor
        cov_replicate = pd.DataFrame(df_f_loads.values.T @ df_f_loads.values * factor_cov, 
                                     index=asset_return.columns, columns=asset_return.columns)
    resid = asset_cov - cov_replicate
    return df_f_loads, resid, const

def _rs_params(const_ols, trans_prob, factor_load_ols_s, factor_mu_s, factor_cov_s, cov_resid, n_ols_factor):
    """
    Eq. 11 of Costa & Kwon (2019)
    """
    g1 = trans_prob[0]; g2 = trans_prob[1] # scalar of transition probability to next period regime
    d = cov_resid.values # n_asset x n_asset matrix of residual vcov

    if n_ols_factor==1: # hmm factor is one
        v1 = factor_load_ols_s[0].values.reshape(1,-1); v2 = factor_load_ols_s[1].values.reshape(1,-1) # 1 x n_asset vector of factor loadings
        f1 = factor_mu_s[0]; f2 = factor_mu_s[1] # scalar of expected factor return
        ff1 = factor_cov_s[0]; ff2 = factor_cov_s[1] # scalar of factor variance
        vcov_rs = g1 * (v1.T @ v1 * ff1) + g2 * (v2.T @ v2 * ff2) + g1*(1-g2) * (v1.T @ v1 * f1**2) \
                    + g2*(1-g1) * (v2.T @ v2 * f2**2) - g1*g2 * (v1.T @ v2 * f1*f2) \
                    - g1*g2 * (v2.T @ v1 * f2*f1) + cov_resid
        eret_rs = const_ols + g1*(v1.T*f1).flatten() + g2*(v2.T*f2).flatten()
    else:
        v1 = factor_load_ols_s[0].values; v2 = factor_load_ols_s[1].values # n_factor x n_asset matrix of factor loadings
        f1 = factor_mu_s[0].values.reshape(-1,1); f2 = factor_mu_s[1].values.reshape(-1,1) # n_factor x 1 vector of factor return
        ff1 = factor_cov_s[0].values; ff2 = factor_cov_s[1].values # n_factor x n_factor matrix of factor vcov    
        vcov_rs = g1 * (v1.T @ ff1 @ v1) + g2 * (v2.T @ ff2 @ v2) + g1*(1-g2) * (v1.T @ f1 @ f1.T @ v1) \
                    + g2*(1-g1) * (v2.T @ f2 @ f2.T @ v2) - g1*g2 * (v1.T @ f1 @ f2.T @ v2) \
                    - g1*g2 * (v2.T @ f2 @ f1.T @ v1) + cov_resid
#         eret_rs = const_ols.values.reshape(-1,1) + g1 * (v1.T @ f1) + g2 * (v2.T @ f2)
        eret_rs = const_ols + (g1*(v1.T @ f1)).flatten() + (g2*(v2.T @ f2)).flatten()
    return eret_rs, vcov_rs

def calc_rs_params(asset_return, factor_return, hmm_return=None, hmm_window="expanding", zscore=True, random=42):
    """
    calculate regime-switching vcov under two-regime hmm
    """
    hmm_return = factor_return if hmm_return is None else hmm_return
    factor_return = factor_return.reindex(asset_return.index)
    
    assert hmm_window in {"rolling", "expanding"}, f"hmm_window must be either 'rolling' or 'freq'"
    if hmm_window=="rolling":
        hmm_return = hmm_return.reindex(asset_return.index)
    else:
        hmm_return = hmm_return.loc[:asset_return.index.max(),:] if isinstance(hmm_return, pd.DataFrame) else hmm_return.loc[:asset_return.index.max()]
    
    if zscore:
        hmm_return = hmm_return.apply(lambda x: (x-x.mean())/x.std(), axis=0) if isinstance(hmm_return, pd.DataFrame) else (hmm_return-hmm_return.mean())/hmm_return.std()
    
    hmm_return_np = hmm_return.values.reshape(-1, hmm_return.values.ndim)
    def hmm_seed(seed):
        hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="full", 
                                    n_iter=1000, random_state=seed)
        hmm_model.fit(hmm_return_np)
        hmm_status = hmm_model.monitor_.converged
        return hmm_model, hmm_status
    np.random.seed(seed=random)
    hmm_model, hmm_status = hmm_seed(np.random.randint(100))
    while hmm_status is not True:
        hmm_model, hmm_status = hmm_seed(np.random.randint(100))
    estim_state = pd.Series(hmm_model.predict(hmm_return_np), index=hmm_return.index).reindex(asset_return.index)
    trans_prob = hmm_model.transmat_[estim_state[-1]]
    
    if len(estim_state.unique())==1 or estim_state.value_counts().min()==1: # only one regime is estimated (is this possible?)
        rs_vcov = asset_return.cov()
        rs_eret = asset_return.mean()
    else:
        factor_load_ols_s = {}; factor_mu_s = {}; factor_cov_s = {} # placeholder
        for state in [0,1]:
            state_dt = estim_state[estim_state==state].index # slice dates for each estim_state
            asset_return_s = asset_return.reindex(state_dt).dropna()
            factor_load_ols_s[state] = _factor_model_decomp(asset_return_s, factor_return)[0]
            factor_mu_s[state] = factor_return.reindex(state_dt).mean()
            factor_cov_s[state] = factor_return.reindex(state_dt).cov() if isinstance(factor_return, pd.DataFrame) else factor_return.reindex(state_dt).var()

        factor_load_ols, cov_resid, const_ols = _factor_model_decomp(asset_return, factor_return)
        rs_eret, rs_vcov = _rs_params(const_ols, trans_prob, factor_load_ols_s, factor_mu_s, factor_cov_s, cov_resid, factor_return.values.ndim)
    return rs_eret, rs_vcov