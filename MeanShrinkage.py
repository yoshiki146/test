import MVO
import numpy as np

def bayes_stein_shrinkage(ret, **kwargs):
    """
    Bayes-Stein Shrinkage estimator of expected return with the shrinkage target to Minimum-Variance
    Given return series, returns shrinkage intensity and expected return
    """
    asset_weight_lower_bound = kwargs.get("asset_weight_lower_bound", 0)
    asset_weight_upper_bound = kwargs.get("asset_weight_upper_bound", 1)
    currency_group = kwargs.get("currency_group", np.repeat(0,ret.shape[1])) # net long constraint for currency (for fx hedge)
    weight_sum = kwargs.get("weight_sum", 1)
    sum_one_asset = kwargs.get("sum_one_asset", np.repeat(1,ret.shape[1])) # exclude certain assets from sum one const (e.g. FX)
    
    n_obs = ret.shape[0]
    n_assets = ret.shape[1]
    vcov = ret.cov()
    
    minvol_weight = MVO.calc_minvol(vcov, 
                                    asset_weight_lower_bound=asset_weight_lower_bound,
                                    asset_weight_upper_bound=asset_weight_upper_bound, 
                                    currency_group=currency_group,                                    
                                    weight_sum=weight_sum, sum_one_asset=sum_one_asset)
    minvol_return = minvol_weight @ ret.mean()
    
    bs_lambda =  (n_assets+2)*(n_obs-1)\
                    / ((ret.mean() - minvol_return * np.ones(n_assets))\
                        @ np.linalg.inv(vcov) \
                            @ (ret.mean() - minvol_return * np.ones(n_assets))\
                                * (n_obs - n_assets - 2))
    bs_intensity = bs_lambda / (n_obs + bs_lambda)
    bs_expected_return = bs_intensity*minvol_return + (1-bs_intensity)*ret.mean()
    return bs_intensity, bs_expected_return


