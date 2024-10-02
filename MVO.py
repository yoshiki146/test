import cvxpy as cp
import pandas as pd
import numpy as np

def _freq_to_scale(freq):
    assert freq in {"BM", "W-Fri", "B"}, f"Invalid freq: {freq}. Must be one of ['BM', 'W-Fri', 'B']"
    if freq == "BM":
        scale = 12
    elif freq == "W-Fri":
        scale = 52
    elif freq == "B":
        scale = 252
    return scale


def calc_mvo(eret, vcov, risk_tolerance, **kwargs):
    asset_weight_lower_bound = kwargs.get("asset_weight_lower_bound", 0)
    asset_weight_upper_bound = kwargs.get("asset_weight_upper_bound", 1)
    currency_group = kwargs.get("currency_group", np.repeat(0,len(eret))) # net long constraint for currency (for fx hedge)
    weight_sum = kwargs.get("weight_sum", 1)
    sum_one_asset = kwargs.get("sum_one_asset", np.repeat(1,len(eret))) # exclude certain assets from sum one const (e.g. FX)
    try:
        x = cp.Variable(len(eret))
        constraints = [sum_one_asset@x == weight_sum, 
                       x >= asset_weight_lower_bound,
                       x <= asset_weight_upper_bound, 
                       currency_group@x >= 0]
        obj = cp.Minimize((1/risk_tolerance) * cp.quad_form(x, vcov.values) - eret.values.T @ x)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)
        optimal_weight = pd.Series(x.value, index=eret.index, name="weight")
        return optimal_weight
    except Exception as e:
        print(f"An error occurred for mvo: {e}")
        return pd.Series(np.nan, index=eret.index, name="weight")
    
def calc_mvo_tv(eret, vcov, target_vol=0.1, freq=None, scale=None, **kwargs):
    asset_weight_lower_bound = kwargs.get("asset_weight_lower_bound", 0)
    asset_weight_upper_bound = kwargs.get("asset_weight_upper_bound", 1)
    currency_group = kwargs.get("currency_group", np.repeat(0,len(eret))) # net long constraint for currency (for fx hedge)
    weight_sum = kwargs.get("weight_sum", 1)
    sum_one_asset = kwargs.get("sum_one_asset", np.repeat(1,len(eret))) # exclude certain assets from sum one const (e.g. FX)
    
    scale = _freq_to_scale(freq) if scale==None else scale
   
    try:
        x = cp.Variable(len(eret))
        constraints = [sum_one_asset@x == weight_sum, 
                       x >= asset_weight_lower_bound,
                       x <= asset_weight_upper_bound, 
                       currency_group@x >= 0,
                       cp.quad_form(x, vcov.values)*scale <= target_vol**2]
        obj =  cp.Minimize(-eret.values.T@x)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)
        optimal_weight = pd.Series(x.value, index=eret.index, name="weight")
        return optimal_weight
    except Exception as e:
        print(f"An error occurred for mvo_tv: {e}")
        return pd.Series(np.nan, index=eret.index, name="weight")
    
def calc_minvol(vcov, **kwargs):
    asset_weight_lower_bound = kwargs.get("asset_weight_lower_bound", 0)
    asset_weight_upper_bound = kwargs.get("asset_weight_upper_bound", 1)
    currency_group = kwargs.get("currency_group", np.repeat(0,vcov.shape[0])) # net long constraint for currency (for fx hedge)
    weight_sum = kwargs.get("weight_sum", 1)
    sum_one_asset = kwargs.get("sum_one_asset", np.repeat(1,vcov.shape[0])) # exclude certain assets from sum one const (e.g. FX)
    try:
        x = cp.Variable(vcov.shape[0])
        constraints = [sum_one_asset@x == weight_sum, 
                       x >= asset_weight_lower_bound,
                       x <= asset_weight_upper_bound, 
                       currency_group@x >= 0]
        obj = cp.Minimize(cp.quad_form(x, vcov.values))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS)
        optimal_weight = pd.Series(x.value, index=vcov.index, name="weight")
        return optimal_weight
    except Exception as e:
        print(f"An error occurred for minimum variance: {e}")
        return pd.Series(np.nan, index=vcov.index, name="weight")
    
    
def calc_erc(vcov):
    """
    Calculate risk parity portofolio givem covariance matrix. 
    Can be extended for risk budget. See Spinu (2013) for the algorithm.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2297383
    https://quant.stackexchange.com/questions/71305/how-to-understand-this-convex-optimization-method-to-find-risk-budget-portfolio
    """
    # No FX hedge included in the programme
    n_assets = vcov.shape[0]
    b = np.ones(n_assets) / n_assets  # Risk budget

    x = cp.Variable(n_assets)
    constraints = [x >= 0]
    objective = 0.5 * cp.quad_form(x, vcov) - cp.sum(cp.multiply(b, cp.log(x)))
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        prob.solve(solver=cp.SCS)
        w = x / cp.sum(x) # Normalize weights
        optimal_weight = pd.Series(w.value, index=vcov.index, name="weight")
    except Exception as e:
        print(f"An error occurred for erc: {e}")
        optimal_weight = pd.Series(np.nan, index=vcov.index, name="weight")
    return optimal_weight

