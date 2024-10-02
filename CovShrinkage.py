import numpy as np
import numpy.matlib as mt
import pandas as pd
import math
from itertools import product

# import warnings
# warnings.filterwarnings("error")


# Ledoit and Wolf (2004b, JMA) A well- conditioned estimator for large-dimensional covariance
# Linear Shrinkage toward Identity Matrix
# Reference: https://github.com/pald22/

def LW_common_diag(ret, **kwargs):
    """
    Shrinkage toward common diagonal matrix (LW 2004b)
    Σ = ρ*νI + (1-ρ)S
    ν: mean of diag(S)
    Return: Shrinkage intensity, Shrunk cov
    """
    
    N,p = ret.shape 
    Y = ret.sub(ret.mean(axis=0), axis=1)
    n = N - 1

    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     

    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag)/len(diag)
    target = meanvar*np.eye(p)
    
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n     # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))  
    
    pihat = sum(piMat.sum())
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    rho_diag = 0
    rho_off = 0
    
    # compute shrinkage intensity
    rhohat=rho_diag+rho_off
    kappahat=(pihat-rhohat)/gammahat
    shrinkage=max(0,min(1,kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values,
                            index=ret.columns, columns=ret.columns)
    
    return shrinkage, sigmahat


def LW_common_vcov(ret, ignore_rho=False, **kwargs):
    """
    Shrinkage toward a target matrix, which has common variance on the diagonal elements 
         and common covariance on the off-diag elements. 
    Σ = ρ*(νI + ξ(1-I)) + (1-ρ)S
    ν: mean of variance (diag of S)
    ξ: mean of covariance (off-diag of S)
    """
    N,p = ret.shape
    Y = ret.sub(ret.mean(axis=0), axis=1)
    n = N - 1
    
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    
    diag = np.diag(sample.to_numpy())
    meanvar = sum(diag) / len(diag)
    meancov = (np.sum(sample.to_numpy()) - np.sum(np.eye(p)*sample.to_numpy()))/(p*(p-1)) # common covariance for off-diagonal elements
    target = pd.DataFrame(meanvar*np.eye(p) + meancov*(1-np.eye(p)))
    
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n # sample covariance matrix of squared returns
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    rho_diag = (sample2.sum().sum()-np.trace(sample.to_numpy())**2)/p;
    sum1 = Y.sum(axis=1)
    sum2 = Y2.sum(axis=1)
    temp = (np.multiply(sum1.to_numpy(),sum1.to_numpy())-sum2)
    rho_off1 = np.sum(np.multiply(temp,temp))/(p*n)
    rho_off2 = (sample.sum().sum()-np.trace(sample.to_numpy()))**2/p
    rho_off = (rho_off1-rho_off2)/(p-1)
    
    rhohat = 0 if ignore_rho else rho_diag + rho_off
    kappahat = (pihat-rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values,
                            index=ret.columns, columns=ret.columns)
    
    return shrinkage, sigmahat


def LW_common_corr(ret, ignore_rho=False, **kwargs):
    """
    Ledoit and Wolf (2004a)
    Shrinkage toward a target matrix, which has same variance as sample cov 
        and variances estimated using avg corr from sample cov. 
    Σ = ρ*(S*I + r*(1-I)*σ@σ.t) + (1-ρ)S
    r: mean of correlation from S
    σ: sqrt of variances (Px1 vector)
    """
    N,p = ret.shape
    Y = ret.sub(ret.mean(axis=0), axis=1)
    n = N - 1
    
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
        
    samplevar = np.diag(sample.to_numpy())
    sqrtvar = pd.DataFrame(np.sqrt(samplevar))
    rBar = (np.sum(np.sum(sample.to_numpy()/np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())))-p)/(p*(p-1)) # mean correlation
    target = pd.DataFrame(rBar*np.matmul(sqrtvar.to_numpy(),sqrtvar.T.to_numpy())) # estimate vcov matrix with common corr
    target[np.logical_and(np.eye(p),np.eye(p))] = sample[np.logical_and(np.eye(p),np.eye(p))] # overwrite diagnal element by sample cov
    
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n  
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())

    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    rho_diag =  np.sum(np.diag(piMat))   
    # off-diagonal part of the parameter that we call rho 
    term1 = pd.DataFrame(np.matmul((Y**3).T.to_numpy(),Y.to_numpy())/n)
    term2 = pd.DataFrame(np.transpose(mt.repmat(samplevar,p,1))*sample)
    thetaMat = term1-term2
    thetaMat[np.logical_and(np.eye(p),np.eye(p))] = pd.DataFrame(np.zeros((p,p)))[np.logical_and(np.eye(p),np.eye(p))]
    rho_off = rBar*(np.matmul((1/sqrtvar).to_numpy(),sqrtvar.T.to_numpy())*thetaMat).sum().sum()
    
    # compute shrinkage intensity
    rhohat = 0 if ignore_rho else rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    # compute shrinkage estimator
    sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values, 
                            index=ret.columns, columns=ret.columns)

    # compute shrinkage estimator
#     sigmahat = shrinkage*target+(1-shrinkage)*sample
    
    return shrinkage, sigmahat


def LW_sample_diag(ret, ignore_rho=False, **kwargs):
    """
    Shrinkage toward a diagonal matrix of sample var.
    Σ = ρ*diag(S)*I + (1-ρ)S
    """
    N,p = ret.shape
    Y = ret.sub(ret.mean(axis=0), axis=1)
    n = N - 1
    #Cov df: sample covariance matrix
    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
    target = pd.DataFrame(np.diag(np.diag(sample.to_numpy())))
    
    Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
    sample2= pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n 
    piMat=pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
    pihat = sum(piMat.sum())
    gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2
    
    rho_diag =  np.sum(np.diag(piMat))
    rho_off = 0
    rhohat = 0 if ignore_rho else rho_diag + rho_off
    kappahat = (pihat - rhohat) / gammahat
    shrinkage = max(0 , min(1 , kappahat/n))
    
    sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values,
                            index=ret.columns, columns=ret.columns)
    
    return shrinkage, sigmahat

def _vcov_assetclass_target(vcov, group_level):
    target_mat = pd.DataFrame(index=vcov.index, columns=vcov.columns)
    # asset_class_level = vcov.columns.names.index("AssetClass")
    for ac1, ac2 in product(vcov.columns.get_level_values(group_level).unique(), repeat=2):
        if group_level==1:
            loc_ac1 = pd.IndexSlice[:,ac1]
            loc_ac2 = pd.IndexSlice[:,ac2]
        elif group_level==2:
            loc_ac1 = pd.IndexSlice[:,:,ac1]
            loc_ac2 = pd.IndexSlice[:,:,ac2]
        elif group_level==3:
            loc_ac1 = pd.IndexSlice[:,:,:,ac1]
            loc_ac2 = pd.IndexSlice[:,:,:,ac2]
        else:
            raise ValueError("Define slicing method corresponding to group_level")
        submat_df = vcov.loc[loc_ac1, loc_ac2]
        submat = submat_df.to_numpy()
        if ac1==ac2:
            n_asset_sub = submat.shape[0]
            submat_tr = np.trace(submat)
            if n_asset_sub == 1:
                submat_target = submat_tr
            else:
                submat_off = submat.sum() - submat_tr
                submat_target = (submat_tr/n_asset_sub)*np.eye(len(submat)) \
                                    + (submat_off/(n_asset_sub*(n_asset_sub-1)))*(np.ones_like(submat)-np.eye(len(submat))) 
        else:
            submat_target = submat.mean() * np.ones_like(submat)
        target_mat.loc[loc_ac1, loc_ac2] = submat_target
    return target_mat

def denard_gcvc(ret, ignore_rho=False, multi_class=True, group_level=None, **kwargs):
    """
    Generalised Constant Variance Covariance by deNard(2022)
    Shrinkage toward a target matrix, which has intra-asset-class common variance and covariance,
        and inter-cluster covariance (three parameters)
    We temporarily ignore rho for calculating shrinkage intensity as it is negligible as per deNard. 
    """
    if multi_class:
        if group_level is None:
            raise ValueError("group_level is required when multi_class=True")
        N,p = ret.shape
        Y = ret.sub(ret.mean(axis=0), axis=1)
        n = N - 1

        sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
        target = _vcov_assetclass_target(Y.cov(), group_level).to_numpy()

        Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
        sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n # sample covariance matrix of squared returns
        piMat = pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
        pihat = sum(piMat.sum())

        gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2

        rho_diag = (sample2.sum().sum()-np.trace(sample.to_numpy())**2)/p;
        sum1 = Y.sum(axis=1)
        sum2 = Y2.sum(axis=1)
        temp = (np.multiply(sum1.to_numpy(),sum1.to_numpy())-sum2)
        rho_off1 = np.sum(np.multiply(temp,temp))/(p*n)
        rho_off2 = (sample.sum().sum()-np.trace(sample.to_numpy()))**2/p
        rho_off = (rho_off1-rho_off2)/(p-1)

        rhohat = 0 if ignore_rho else rho_diag + rho_off
        kappahat = (pihat-rhohat) / gammahat
        shrinkage = max(0 , min(1 , kappahat/n))

        sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values,
                                index=ret.columns, columns=ret.columns)
    else:
        shrinkage, sigmahat = LW_common_vcov(ret=ret, ignore_rho=ignore_rho)
    return shrinkage, sigmahat



# def _vcov_assetclass_target(vcov):
#     target_mat = pd.DataFrame(index=vcov.index, columns=vcov.columns)
#     # asset_class_level = vcov.columns.names.index("AssetClass")
#     for ac1, ac2 in product(vcov.columns.get_level_values(2).unique(), repeat=2):
#         loc_ac1 = pd.IndexSlice[:,:,ac1]
#         loc_ac2 = pd.IndexSlice[:,:,ac2]
#         submat_df = vcov.loc[loc_ac1, loc_ac2]
#         submat = submat_df.to_numpy()
#         if ac1==ac2:
#             n_asset_sub = submat.shape[0]
#             submat_tr = np.trace(submat)
#             if n_asset_sub == 1:
#                 submat_target = submat_tr
#             else:
#                 submat_off = submat.sum() - submat_tr
#                 submat_target = (submat_tr/n_asset_sub)*np.eye(len(submat)) \
#                                     + (submat_off/(n_asset_sub*(n_asset_sub-1)))*(np.ones_like(submat)-np.eye(len(submat))) 
#         else:
#             submat_target = submat.mean() * np.ones_like(submat)
#         target_mat.loc[loc_ac1, loc_ac2] = submat_target
#     return target_mat

# def denard_gcvc(ret, ignore_rho=False, multi_class=True, **kwargs):
#     """
#     Generalised Constant Variance Covariance by deNard(2022)
#     Shrinkage toward a target matrix, which has intra-asset-class common variance and covariance,
#         and inter-cluster covariance (three parameters)
#     We temporarily ignore rho for calculating shrinkage intensity as it is negligible as per deNard. 
#     """
#     if multi_class:
#         N,p = ret.shape
#         Y = ret.sub(ret.mean(axis=0), axis=1)
#         n = N - 1

#         sample = pd.DataFrame(np.matmul(Y.T.to_numpy(),Y.to_numpy()))/n     
#         target = _vcov_assetclass_target(Y.cov()).to_numpy()

#         Y2 = pd.DataFrame(np.multiply(Y.to_numpy(),Y.to_numpy()))
#         sample2 = pd.DataFrame(np.matmul(Y2.T.to_numpy(),Y2.to_numpy()))/n # sample covariance matrix of squared returns
#         piMat = pd.DataFrame(sample2.to_numpy()-np.multiply(sample.to_numpy(),sample.to_numpy()))
#         pihat = sum(piMat.sum())

#         gammahat = np.linalg.norm(sample.to_numpy()-target,ord = 'fro')**2

#         rho_diag = (sample2.sum().sum()-np.trace(sample.to_numpy())**2)/p;
#         sum1 = Y.sum(axis=1)
#         sum2 = Y2.sum(axis=1)
#         temp = (np.multiply(sum1.to_numpy(),sum1.to_numpy())-sum2)
#         rho_off1 = np.sum(np.multiply(temp,temp))/(p*n)
#         rho_off2 = (sample.sum().sum()-np.trace(sample.to_numpy()))**2/p
#         rho_off = (rho_off1-rho_off2)/(p-1)

#         rhohat = 0 if ignore_rho else rho_diag + rho_off
#         kappahat = (pihat-rhohat) / gammahat
#         shrinkage = max(0 , min(1 , kappahat/n))

#         sigmahat = pd.DataFrame((shrinkage*target+(1-shrinkage)*sample).values,
#                                 index=ret.columns, columns=ret.columns)
#     else:
#         shrinkage, sigmahat = LW_common_vcov(ret=ret, ignore_rho=ignore_rho)
#     return shrinkage, sigmahat