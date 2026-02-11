import numpy as np
from typing import Dict, Any
from models.voltage_shift import FeD, FeDParams

from utils.fed_literature import (
    kim_2024,
    han_chen_2025,
    liu_2022,
    hu_2025,
)

fed_literature_options = {
    "kim_2024": kim_2024,
    "han_chen_2025": han_chen_2025,
    "liu_2022": liu_2022,
    "hu_2025": hu_2025,
}

def pick(x, variant: int): # pick the selected variant value ONLY IF x has variants 
    if isinstance(x, (list, tuple)):
        return x[variant]
    else:
        return x

def fed_from_literature(name: str, *, variant: int = 0) -> FeD:
    """
    User inputs name of preset FeD they want
    Some parameters have multiple values (device variants)
    variant: int is a lever to select among these params
    """

    if name not in fed_literature_options:
        raise ValueError(f"Unknown preset device '{name}' selected.")
    
    device = fed_literature_options[name]

    # extract params

    V_read = pick(device["V_read"], variant)

    LRS_GOhm = pick(device["LRS"], variant)
    HRS_GOhm = pick(device["HRS"], variant)

    #voltage and current ranges
    V_range = device["V_range"]
    I_range = device["I_range"]

    # max voltage
    if isinstance(V_range, (list, tuple)):
        V_max = max(V_range)
    else:
        V_max = V_range

    # max/min current
    I_max = max(I_range)
    I_min = min(I_range)

    #Deriving relevant quantities
    G_on = 1.0 / (LRS_GOhm * 1e9)
    G_off = 1.0 / (HRS_GOhm * 1e9)

    I_on = G_on * V_read
    I_off = G_off * V_read

    #exp slope
    alpha = np.log(I_max / I_min) / V_max

    #G conductance related exp parameter
    Gf_LRS = I_on / np.exp(alpha * V_read)
    Gf_HRS = I_off / np.exp(alpha * V_read)

    Gr_LRS = Gf_LRS
    Gr_HRS = Gf_HRS
    ar = alpha

    params = FeDParams(
        Gf_LRS_A = Gf_LRS,
        af_LRS_per_V = alpha,
        Gr_LRS_A = Gr_LRS,
        ar_LRS_per_V = ar,
        Gf_HRS_A = Gf_HRS,
        af_HRS_per_V = alpha,
        Gr_HRS_A = Gr_HRS,
        ar_HRS_per_V = ar,
    )

    return FeD(params, enable_voltage_shift=False)
    #dont Vshift because physics is already enforced by ON/OFF info