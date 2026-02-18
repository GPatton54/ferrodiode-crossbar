import numpy as np
from typing import Tuple
from models.voltage_shift import FeD, FeDParams
import matplotlib.pyplot as plt

"""
User inputs for basic params:
- Vread, Vwrite
- Gon,Goff (dI/dV at Vread)
- Ion = LRS I at Vread
- Ioff = HRS I at VRead
"""

def get_exp_params( #helper func to get G and alpha from the basic inputs
    I_target: float,
    V_target: float,
    dIdV: float
) -> Tuple[float,float]:
    """
    Given:
    - Itarget = I at Vread
    - Vtarget = Vread
    - G_on = dI/dV
    Solve analytically for G, alpha.
    """

    if I_target <= 0 or V_target < 0 or dIdV <= 0:
        raise ValueError("I, V, derivative must be positive for fitting")
    
    alpha = dIdV / I_target
    G = I_target / np.exp(alpha * V_target)

    return G, alpha

def basic_params_FeD(
        *,
        V_read: float,
        V_write: float,
        G_on: float,
        G_off: float,
        I_on: float,
        I_off: float,

        asym_G_LRS: float,
        asym_a_LRS: float,
        asym_G_HRS: float,
        asym_a_HRS: float,

        smoothing_voltage: float = 0.015,
        fe_kwargs: dict | None=None,
) -> FeD:
    
    """
    Use user-inputted basic paramss to produce FeD object
    asym_G/a_LRS/HRS are asymmetry factors to make sure upper/lower branches are asymmetric
    """

    if (not(0.0<asym_G_LRS<1.0) or not(0.0<asym_a_LRS<1.0) or not(0.0<asym_G_HRS<1.0) or not(0.0<asym_a_HRS<1.0)):
        raise ValueError("All asymmetries must be on (0,1).")

    # LRS upper branch
    Gf_LRS, af_LRS = get_exp_params(I_on,V_read,G_on)
    # LRS lower branch
    Gr_LRS, ar_LRS = asym_G_LRS*Gf_LRS, asym_a_LRS*af_LRS

    # HRS upper branch
    Gf_HRS, af_HRS = get_exp_params(I_off,V_read,G_off)
    # HRS lower branch
    Gr_HRS, ar_HRS = asym_G_HRS*Gf_HRS, asym_a_HRS*af_HRS

    #OPTIONAL fundamental FE params override
    fe_kwargs = fe_kwargs or {}

    fed_params = FeDParams(
        Gf_LRS_A = Gf_LRS,
        af_LRS_per_V = af_LRS,
        Gr_LRS_A = Gr_LRS,
        ar_LRS_per_V = ar_LRS,
        Gf_HRS_A = Gf_HRS,
        af_HRS_per_V = af_HRS,
        Gr_HRS_A = Gr_HRS,
        ar_HRS_per_V = ar_HRS,
        V_smooth_V = smoothing_voltage,
        **fe_kwargs
    )

    fed_object = FeD(fed_params, enable_Vshift=False)
    # dont Vshift basic params
    # user inputs in this case lock the voltage (x) position of curves
    # additional voltage shifts breaks the enforced physics

    return fed_object