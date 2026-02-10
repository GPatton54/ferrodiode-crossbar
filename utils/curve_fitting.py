import numpy as np
from scipy.optimize import curve_fit
from utils.iv_csv import read_IV_csv
from typing import Dict

def base_exp_model(V,G,alpha):
    return G * np.exp(alpha * np.abs(V))

def fit_IV_curve(V: np.ndarray, I: np.ndarray, initial_guess: tuple[float, float]=None):

    if initial_guess is not None:
        # allows manual initial guess input
        # for experimental_data.py, where there is branch splitting/different expressions  
        G0, alpha0 = initial_guess
    else: 
        #come up with initial guesses
        G0 = np.min(np.abs(I))
        alpha0 = np.log(np.max(np.abs(I)) / G0) / np.max(np.abs(V))

    params0 = [G0, alpha0] 
    
    #do the fitting
    params_optimized, params_covariance = curve_fit(base_exp_model, V, I, p0=params0, maxfev=10000)
    G_fit, alpha_fit = params_optimized
    
    #error metrics
    I_pred = base_exp_model(V, G_fit, alpha_fit) # prediction
    loss = np.mean((I - I_pred)**2) #MSE loss
    ss_res = np.sum((I - I_pred)**2) #residual sum of squares (deviation from data)
    ss_tot = np.sum((I - np.mean(I))**2) #total sum of squares 
    R2 = 1 - ss_res / ss_tot #r^2
    
    return G_fit, alpha_fit, loss, R2

def fit_all_curves(filename: str) -> Dict[str, dict[str,float]]:
    curves = read_IV_csv(filename)
    V = curves["voltage"]
    fitresults = {}
    
    for key, I in curves.items():

        if key == "voltage":
            continue

        G, alpha, loss, R2 = fit_IV_curve(V, I)
        fitresults[key] = {
            "G": G,
            "alpha": alpha,
            "loss": loss,
            "R2": R2
        }
    
    return fitresults