import numpy as np
import pandas as pd
from typing import Dict

def read_IV_csv(filename: str, voltage_col: str = "V") -> Dict[str, np.ndarray]:
    IV_curve_df = pd.read_csv(filename)

    #look for correct column name for the voltage column
    if voltage_col not in IV_curve_df.columns:
        raise ValueError(f"CSV must have voltage col '{voltage_col}'")
    
    #convert volts col to a 1d nparray
    V = IV_curve_df[voltage_col].to_numpy()
    
    #dict for current curves, with LRS HRS and V
    #put in voltage 1dnparray to start
    curves: Dict[str, np.ndarray] = {"voltage": V}

    #for each col, put in dict
    for col in IV_curve_df.columns:
        if (col != voltage_col):
            curves[col] = IV_curve_df[col].to_numpy()

    return curves