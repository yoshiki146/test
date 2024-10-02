import MVO
import numpy as np

def bayes_stein_shrinkage(ret):
    """
    Bayes-Stein Shrinkage estimator of expected return with the shrinkage target to Minimum-Variance
    Given return series, returns shrinkage intensity and expected return
    """
    n_obs = ret.shape[0]
    n_assets = ret.shape[1]
    vcov = ret.cov()
    
    minvol_weight = MVO.calc_minvol(vcov)
    minvol_return = minvol_weight @ ret.mean()
    
    bs_lambda =  (n_assets+2)*(n_obs-1)\
                    / ((ret.mean() - minvol_return * np.ones(n_assets))\
                        @ np.linalg.inv(vcov) \
                            @ (ret.mean() - minvol_return * np.ones(n_assets))\
                                * (n_obs - n_assets - 2))
    bs_intensity = bs_lambda / (n_obs + bs_lambda)
    bs_expected_return = bs_intensity*minvol_return + (1-bs_intensity)*ret.mean()
    return bs_intensity, bs_expected_return


