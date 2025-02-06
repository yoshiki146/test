import numpy as np
import pandas as pd

def _freq_to_scale(freq):
    assert freq in {"BM", "W-Fri", "B"}, f"Invalid freq: {freq}. Must be one of ['BM', 'W-Fri', 'B']"
    if freq == "BM":
        scale = 12
    elif freq == "W-Fri":
        scale = 52
    elif freq == "B":
        scale = 252
    return scale

styler_config = {
    "ret(ann)": "{:,.2%}".format,
    "vol": "{:,.2%}".format,
    "ret/vol": "{:,.3f}".format,
    "drawdown": "{:,.2%}".format
}

def summary_stats(ret, riskfree=None, freq=None, scale=None, styler=True):
    scale = _freq_to_scale(freq) if scale==None else scale
    n_obs = ret.shape[0]
    
    cumret = (1+ret).cumprod() - 1
    annret = (1+cumret.iloc[-1,:])**(scale/n_obs) - 1 
    vol = ret.std() * np.sqrt(scale)
    dd = ((1+cumret).div(1+cumret.cummax()) - 1).min() * -1
    cr = annret.div(dd)
    
    if riskfree is None:
        sr = annret.div(vol)   
    else:
        assert isinstance(riskfree, pd.Series), "riskfree must be pd.Series indexed by date"
        # assert ret.index.freq==riskfree.index.freq, "freq does not match for ret and riskfree"
        rf_cum = (1+riskfree.reindex(ret.index)).prod() - 1
        annret_excess = (1+cumret.iloc[-1,:].sub(rf_cum)) **(scale/n_obs) - 1 
        sr = annret_excess.div(vol)
    
    res = pd.DataFrame({"ret(ann)":annret, "vol":vol, "drawdown":dd,
                        "SR":sr, "calmer_ratio": cr}).dropna()
    return res.style.format(styler_config) if styler else res