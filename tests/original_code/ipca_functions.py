import numpy as np
import scipy.linalg as linalg
from parameters import *
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

import dkkm_functions as dkkm


# ridge regression function
def regress(X, y, alpha = 0):
    T, P = X.shape
    if T < P:
        U, d, VT = linalg.svd(X @ X.T)
        V = VT.T
        W = X.T @ V @ np.diag(1/np.sqrt(d))
        # sequential multiplication avoids creating PxP matrix
        XTy = X.T @ y
        WTXTy = W.T @ XTy
        pi = W @ np.diag(1 / (d + alpha)) @ WTXTy
    else:
        V, d, VT = linalg.svd(X.T @ X)
        pi = V @ np.diag(1 / (d + alpha)) @ VT @ X.T @ y
    return pi


# normalization function ensures Gamma'*Gamma = I and ff' is diagonal, and factors have positive mean (pp. 507)
def normalization(Gamma, f):
    chol = np.linalg.cholesky(Gamma.T @ Gamma)
    cholinv = np.linalg.inv(chol)
    U, _, _ = np.linalg.svd(chol @ f @ f.T @ chol.T)
    Gamma = Gamma @ cholinv @ U
    f = U.T @ chol @ f

    # sign convention: f has positive mean
    sign_conv = np.sign(np.mean(f, axis=1)).reshape((-1,1))
    sign_conv[sign_conv==0] = 1
    f = f*sign_conv
    Gamma = Gamma*sign_conv.T
    return Gamma, f


# compute one iteration of alternating least squares
def ipca_iter(Gamma0, f0, panel, start, rets_stack, firm_counts, Z_stack):
    
    chars = panel.columns[:-1]
    L = len(chars)

    f1 = np.zeros(f0.shape)
    K = f0.shape[0]

    for t in range(360):
        data = panel.loc[start + t,:][chars]
        rets = panel.loc[start + t,:]['xret']
        
        f1[:, t] = regress(data @ Gamma0, rets)  # formula (6)
    
    #numer, denom = 0, 0
    #for t in range(360):
    #    data = panel.loc[start + t,:][chars]
    #    rets = panel.loc[start + t,:]['xret']
    #    kron_product = np.kron(data, f1[:, t].T)
    #    numer += kron_product.T @  rets
    #    denom += kron_product.T @ kron_product
    #Gamma1 = linalg.lstsq(np.array(denom), np.array(numer))[0].reshape(Gamma0.shape) # formula (7)
    
    
    f_stack = np.repeat(np.repeat(f1.T, firm_counts, axis = 0), L, axis = 1)

    X = Z_stack*f_stack # X vector of characteristics interacted with factors
    A = X.T@X
    y = (X.T)@rets_stack
    Gamma1 = linalg.solve(A, y, assume_a = 'pos').reshape((K, L)).T

    return Gamma1, f1

# main iteration loop for alternating least squares - alternates between updating latent factors and linear coefficients on latent factors
def fit_ipca(panel, start, K, tol, Gamma0 = None, f0 = None): 

    chars = panel.columns[:-1]
    #import pdb; pdb.set_trace()
    L = len(chars)
    # SJP edit -- to undo, delete this code block, move commented block starting "maxiter = 1000" up and uncomment it
    # >>
    '''
    Q = np.zeros((360,L))
    f1 = np.zeros((L, 360)) # is f_hat when K==L, otherwise overwritten - need to set rowlength to L even though overwritten version will be K by 360
    ridge = Ridge(fit_intercept=False, alpha=360*0) # set up for loop usage
    for t in range(360):
        # for K<L: used in SVD below
        
        Q[t,:] = linalg.lstsq(a = panel.loc[t+start,:][chars], b = panel.loc[t+start,:]['xret'])[0]

        # for K==L
        f1[:,t] = Q[t,:] #ridge.fit(
            #X=(panel.loc[start + t,:][chars]).to_numpy(), 
            #y=(panel.loc[start + t,:]['xret']).to_numpy()
            #).coef_ # this should be the same as Q[t,:], so no need to compute twice
    if K == L: # FMR if K=L - normalize Gamma_beta = I
        # f_hat is produced from loop above
        Gamma1 = np.eye(L)
        Gamma_beta_hat, f_hat = Gamma1, f1
    else: # SVD to find
        [u,s,vh] = linalg.svd(Q, full_matrices=False)
        f1 = s[:K].reshape((-1,1))*u[:,:K].T
        Gamma1 = vh[:,:K]
    # <<
    '''
    # # SJP comment and moved down from immediately below "L = len(chars)
    # >>
    maxiter = 1000

    # initial guess pp.507
    if (np.sum(Gamma0) == None) & (np.sum(f0) == None):
        X = np.zeros((360, L))
        for t in range(360):
            Nt = panel.loc[t+start,:]['xret'].shape[0]
            X[t,:] = panel.loc[t+start,:][chars].T@panel.loc[t+start,:]['xret']/Nt 

        U, D, VT = linalg.svd(X.T, full_matrices = False) 
        Gamma0 = U[:, 0:K] #np.linalg.eig(X.T@X).eigenvectors[:,:K]
        f0 = np.zeros((K, 360))

    Z = np.array(panel.loc[start : start + 359,:][chars])
    rets_stack = panel.loc[start : start + 359,:]['xret']
    firm_counts = np.array(panel.loc[start : start + 359,:].groupby('month').size())
    Z_stack = np.tile(Z, [1, K])

    Gammaprev, fprev = Gamma0, f0
    for iter in range(maxiter): 

        if K == L: # FMR if K=L - normalize Gamma_beta = I

            f_hat = np.zeros(f0.shape)
            for t in range(360):
                data = panel.loc[start + t,:][chars]
                rets = panel.loc[start + t,:]['xret']

                ridge = Ridge(fit_intercept=False, alpha=360*0)
                f_hat[:, t] = ridge.fit(X=data.to_numpy(), y=rets.to_numpy()).coef_

            Gamma_beta_hat = np.eye(L)

            f1, Gamma1 = f_hat, Gamma_beta_hat
            break

        
        # update Gamma_beta_hat, f_hat
        Gamma1, f1 = ipca_iter(Gamma0, f0, panel, start, rets_stack, firm_counts, Z_stack) 

        error_gamma = np.max(abs(Gamma0 - Gamma1))
        mom = 0 #0.5*((error_gamma < 0.5) & (error_gamma > 0.1)) + 0.9*(error_gamma < 0.1) # momentum schedule
        Gamma1, f1 = Gamma1 + mom*(Gamma0 - Gammaprev), f1 + mom*(f0 - fprev)

        # error tolerance 
        Gammaprev, fprev = Gamma0, f0
        Gamma0, f0 = Gamma1, f1 

        if (error_gamma < tol) & (start % 10 == 0):
            print(f'{K} Factors converged, month = {start}')
            break
        
        if (error_gamma < tol):
            break

        if (error_gamma > tol) & (iter == maxiter-1):
            print(f'{K} Factors DID NOT converge, month = {start}, error: {error_gamma}')
    # <<
        

    if K != L:
        Gamma_beta_hat, f_hat = normalization(Gamma1, f1)

    # compute IPCA portfolio of firm returns
    X = panel.loc[start + 360,:][chars] @ Gamma_beta_hat
    factor_port = linalg.pinvh(X.T @ X) @ X.T
    
    # compute portfolio of IPCA factors
    X = f_hat.T
    y = np.ones(len(X))
    ridge = Ridge(fit_intercept=False, alpha=360*0)
    pi = ridge.fit(X=X, y=y).coef_

    return factor_port, pi, Gamma1, f1 


# function to prepare panels and roll forward IPCA estimation
def fit_ipca_360(panel, K, N, start, end, rff , W, chars):
    #chars = ["size", "bm", "agr", "roe", "mom"]

    # generate and rank-standardize panels
    if rff == 1: # block for using rff characteristics
        panel_ranked_dkkm = panel.groupby('month').apply(lambda x: dkkm.rff(x[chars], W=W)).droplevel(0)
        panel_ranked_dkkm['ones'] = 1
        panel_ranked_dkkm['xret'] = panel['xret']

        panel = panel_ranked_dkkm
        tol = 1e-3
    else:
        panel_ranked = panel[chars].groupby('month').apply(lambda g: dkkm.rank_standardize(g)).reset_index(level=0, drop=True) 
        panel_ranked['ones'] = 1
        panel_ranked['xret'] = panel['xret']

        panel = panel_ranked
        tol = 1e-4

    firms = np.array(panel.loc[start+ 360,:].index)
    ipca_weights_on_stocks = np.zeros((K, N, end+1 - start - 360))
    ipca_factor_weights = np.zeros((K, end+1 - start - 360))
    ipca_weights_on_stocks[:, firms, 0], ipca_factor_weights[:, 0], Gamma0, f0 = fit_ipca(panel, start, K, tol)

    for t in range(start+1, end+1 - 360):
        firms = np.array(panel.loc[t+ 360,:].index)
        ipca_weights_on_stocks[:, firms, t-start], ipca_factor_weights[:, t-start], Gamma0, f0 = fit_ipca(panel, t, K, tol, Gamma0, f0)
        

    return ipca_weights_on_stocks, ipca_factor_weights

# function to compute factor loadings as a linear function of characteristics (panel) given of a single beta factor (factor)
def loadings(panel, factor, month, W = None, K = 1):
    chars = ["size", "bm", "agr", "roe", "mom"]

    if W is None:
        panel_ranked = panel[chars].groupby('month').apply(lambda g: dkkm.rank_standardize(g)).reset_index(level=0, drop=True) 
        panel_ranked['ones'] = 1
        panel_ranked['xret'] = panel['xret']
        panel = panel_ranked
    else: 
        panel_ranked_dkkm = panel.groupby('month').apply(lambda x: dkkm.rff(x[chars], W=W)).droplevel(0)
        panel_ranked_dkkm['ones'] = 1
        panel_ranked_dkkm['xret'] = panel['xret']
        panel = panel_ranked_dkkm
    
    chars = panel.columns[:-1]
    L = len(chars)
    Z = np.array(panel.loc[month : month + 359,:][chars])
    rets_stack = panel.loc[month : month + 359,:]['xret']
    firm_counts = np.array(panel.loc[month : month + 359,:].groupby('month').size())
    Z_stack = np.tile(Z, [1, K])

    f_stack = np.repeat(np.repeat(np.array(factor.loc[month : month + 359]).reshape((-1, 1)), firm_counts, axis = 0), L, axis = 1)

    X = Z_stack*f_stack # X vector of characteristics interacted with factors
    A = X.T@X
    y = (X.T)@rets_stack
    Gamma1 = linalg.solve(A, y, assume_a = 'pos').reshape((1, L)).T
    return panel.loc[month+360,:][chars] @Gamma1



"""

def loss_function(panel, chars, start, L, K, params):

    Gamma = params[:L * K].reshape(L, K)
    f = params[L * K:].reshape(K, 360)
    loss = 0
    for t in range(360):
        rt = panel.loc[t+start,:]['xret']
        Zt = panel.loc[t+start,:][chars]
        ft = f[:,t]
        loss += ((rt - Zt@Gamma@ft)**2).sum()
    return loss

def fit_ipca(panel, start, K, Gamma0 = None, f0 = None): 

    chars = panel.columns[:-1]

    L = len(chars)
    if (np.sum(Gamma0) == None) & (np.sum(f0) == None):
        # initial guess pp.507
        X = np.zeros((360, L))
        for t in range(360):
            Nt = panel.loc[t+start,:]['xret'].shape[0]
            X[t,:] = panel.loc[t+start,:][chars].T@panel.loc[t+start,:]['xret']/Nt 

        U, D, VT = linalg.svd(X.T, full_matrices = False) 
        Gamma0 = U[:, 0:K] #np.linalg.eig(X.T@X).eigenvectors[:,:K]
        
        f0 = np.zeros((K, 360))
        for t in range(360):
            data = panel.loc[start + t,:][chars]
            rets = panel.loc[start + t,:]['xret']
            
            f0[:, t] = regress(data @ Gamma0, rets)  # formula (6)
            
    params_init = np.concatenate([Gamma0.ravel(), f0.ravel()])

    print('starting')
    result = minimize(lambda params: loss_function(panel, chars, start, L, K, params), params_init, options={'disp': True} )

    import pdb; pdb.set_trace()
    return result.x
"""
