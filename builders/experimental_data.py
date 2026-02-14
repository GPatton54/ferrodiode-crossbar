import numpy as np
from typing import Dict
from utils.iv_csv import read_IV_csv
from utils.curve_fitting import fit_IV_curve
from models.voltage_shift import FeD, FeDParams

def fit_one_branch(V: np.ndarray, I: np.ndarray) -> Dict[str, float]:

    V = np.asarray(V)
    I = np.asarray(I)

    V_abs = np.abs(V - V[0]) #standardized voltages abs value distance from cusp
    G0 = I[0] #I min at cusp
    alpha0 = np.log(I[-1] / G0) / V_abs[-1]

    G_fit, alpha_fit, loss, R2 = fit_IV_curve(V_abs, I, [G0, alpha0])

    LRS_dict = {"G": G_fit, "alpha": alpha_fit, "loss": loss, "R2": R2}

    return LRS_dict

def create_LRS_from_data(filename: str) -> Dict[str, Dict[str, float]]:

    curves = read_IV_csv(filename)
    V = curves["voltage"]

    I = next(I for col_name,I in curves.items() if col_name != "voltage")

    sort_indices = np.argsort(V)
    V = V[sort_indices]
    I = I[sort_indices]

    I_mag = np.abs(I)

    cusp_index = np.argmin(I_mag)

    V_upper, I_upper = V[cusp_index:], I_mag[cusp_index:]
    V_lower, I_lower = V[:cusp_index+1], I_mag[:cusp_index+1]

    upper_branch_exp_curve = fit_one_branch(V_upper, I_upper)
    lower_branch_exp_curve = fit_one_branch(V_lower[::-1], I_lower[::-1])

    LRS_params = {
        "upper": upper_branch_exp_curve,
        "lower": lower_branch_exp_curve
    }

    return LRS_params

def experimental_data_FeD(filename: str, experimental_FeD_params: FeDParams) -> FeD:
    """
    Create a fed object using the LRS curve parameters we got
    by fitting experimental data. Set our FeD object parameters
    to the fitted params. Voltage shifting (HRS) is handled automatically
    """

    LRS_params = create_LRS_from_data(filename)

    #LRS upper branch (forward bias)
    experimental_FeD_params.Gf_LRS_A = LRS_params["upper"]["G"]
    experimental_FeD_params.af_LRS_per_V = LRS_params["upper"]["alpha"]

    #LRS lower branch (reverse bias)
    experimental_FeD_params.Gr_LRS_A = LRS_params["lower"]["G"]
    experimental_FeD_params.ar_LRS_per_V = LRS_params["lower"]["alpha"]

    """
    for HRS, set the exact same G,alpha params.
    then voltage shifting of the curve is done automatically
    by voltage_shift.py when current(self,state="HRS",V) is called
    """

    #HRS upper branch
    experimental_FeD_params.Gf_HRS_A = experimental_FeD_params.Gf_LRS_A
    experimental_FeD_params.af_HRS_per_V = experimental_FeD_params.af_LRS_per_V

    #HRS lower branch
    experimental_FeD_params.Gr_HRS_A = experimental_FeD_params.Gr_LRS_A
    experimental_FeD_params.ar_HRS_per_V = experimental_FeD_params.ar_LRS_per_V

    fed_object = FeD(experimental_FeD_params)

    return fed_object