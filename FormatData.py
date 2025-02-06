import pandas as pd

def convert_return(dat, asset_class_map=None, ticker_level=0, asset_blk=False, agg=None):
    ret = dat.pct_change().dropna()
    if asset_class_map is not None:
        asset_cls = [asset_class_map[dat.columns.get_level_values(ticker_level)[i]\
                                     [:dat.columns.get_level_values(ticker_level)[i].find("1")].replace(" ", "")]\
                     for i in range(dat.shape[1])]
        ret.columns = pd.MultiIndex.from_arrays([dat.columns.get_level_values(i) for i in range(len(dat.columns.names))] + [asset_cls])
    if agg is not None:
        ret = ret.groupby(pd.Grouper(freq=agg)).sum()
    if asset_blk:
        blk = pd.DataFrame(0, index=asset_cls, columns=asset_cls)
        for cls_i in set(asset_cls):
                blk.loc[cls_i,cls_i] = 1
        blk = pd.DataFrame(blk.values, index=ret.columns, columns=ret.columns)
        return ret.dropna(), blk
    else:
        return ret.dropna()