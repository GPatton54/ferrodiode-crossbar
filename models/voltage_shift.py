from dataclasses import dataclass
from typing import Union
from typing import Tuple
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

from utils.iv_csv import *
from utils.curve_fitting import *
from utils.fed_literature import (
    kim_2024,
    han_chen_2025,
    liu_2022,
    hu_2025,
)

"""
Simple compact Ferrodiode (FeD) model that mimics the FeD's qualitative I–V 
shape (in log-scale and linear-scale): a strong, diode-like exponential increase 
for +V, a weaker increase for -V, and a deep minimum near V≈0 on a log(I) axis.

Key idea:
- Use a polarization-dependent horizontal voltage shift: ΔV = E_dp * t
- E_dp = E_C * tanh( (β * P_r) / (ε_fe * E_C) )
- Build a piecewise exponential I–V (forward/reverse) with a small current floor.

NOTE:
- This is a compact phenomenological model for plotting and crossbar prototyping.
- Units are handled explicitly for ΔV; the I–V backbone is empirical.

"""

EPS0 = 8.854187817e-12  # F/m


@dataclass
class FeDParams:
    # --- Ferroelectric / stack parameters ---
    t_nm: float = 5.0 # FE thickness
    beta: float = 0.30 # beta = [(C_ox / C_fe) + 1]^{-1} (we could also compute this explicitly using cap eqs)
    Pr_uC_cm2: float = 100.0 # Remnant polarization [uC/cm^2]
    Ec_MV_cm: float = 3.00 # Coercive field [MV/cm]
    eps_r_fe: float = 19.0 # Dielectric constant

    sw_thres: float = 2e-9 #Voltage-time integral threshold to switch polarization. V*uS

    # --- Empirical I–V parameters: LRS --- (approximated from previous work)
    # Forward (+V): I ~ Gf_LRS * exp(af_LRS * |V|)
    Gf_LRS_A: float = 1e-15
    af_LRS_per_V: float = 18.0
    # Reverse (-V): I ~ Gr_LRS * exp(ar_LRS * |V|)
    Gr_LRS_A: float = 3e-16
    ar_LRS_per_V: float = 10.0

    # --- Empirical I–V parameters: HRS --- (approximated from previous work)
    # Forward (+V): I ~ Gf_HRS * exp(af_HRS * |V|)
    Gf_HRS_A: float = 3e-17
    af_HRS_per_V: float = 16.0
    # Reverse (-V): I ~ Gr_HRS * exp(ar_HRS * |V|)
    Gr_HRS_A: float = 1e-17
    ar_HRS_per_V: float = 9.0

    # Small floor current to avoid log(0) and to create a deep minimum near 0 V
    I_floor_A: float = 1e-22

    # Optional: additional "knee softening" (controls how sharp the cusp is)
    V_smooth_V: float = 0.015

class FeD:
    """
    Ferrodiode compact model with polarization-dependent voltage shift.

    - LRS: unshifted I–V with its own (Gf, af, Gr, ar).
    - HRS: shifted by +ΔV with its own (Gf, af, Gr, ar).
    """

    # -------------------------------
    # MASTER CONSTRUCTOR
    # -------------------------------

    def __init__(
        self,
        initialize_type: str,
        *,
        file_csv: str = None,
        preset_model: str = None,
        enable_Vshift: bool = True,
        **custom_basic_params_kwargs,
        ):

        self.initialize_type = initialize_type
        self.enable_Vshift = enable_Vshift
        self.state = "LRS"
        self.p = None #FeDparams placeholder, each option will fill this with FeDparams

        if initialize_type == "experimental":
            self._init_experimental_data(file_csv)

        elif initialize_type == "custom":
            self._init_basic_params(**custom_basic_params_kwargs)

        elif initialize_type == "preset":
            self._init_preset_fed(preset_model, variant=0)

        else:
            raise ValueError(f"Invalid initialization type '{initialize_type}'")

    # -------------------------------
    # INITIALIZATION OPTIONS
    # -------------------------------


    # OPTION 1: EXPERIMENTAL DATA

    def _init_experimental_data(self, csv_filename: str):
        """
        Create a fed object using the LRS curve parameters we got
        by fitting experimental data. Set our FeD object parameters
        to the fitted params. Voltage shifting (HRS) is handled automatically
        """

        LRS_params = self._create_LRS_from_data(csv_filename)

        params = FeDParams(
            #LRS upper branch (forward bias)
            Gf_LRS_A = LRS_params["upper"]["G"],
            af_LRS_per_V = LRS_params["upper"]["alpha"],
            #LRS lower branch (reverse bias)
            Gr_LRS_A = LRS_params["lower"]["G"],
            ar_LRS_per_V = LRS_params["lower"]["alpha"],
        )

        """
        for HRS, set the exact same G,alpha params.
        then voltage shifting of the curve is done automatically
        by voltage_shift.py when current(self,state="HRS",V) is called
        """

        #HRS upper branch
        params.Gf_HRS_A = params.Gf_LRS_A
        params.af_HRS_per_V = params.af_LRS_per_V
        
        #HRS lower branch
        params.Gr_HRS_A = params.Gr_LRS_A
        params.ar_HRS_per_V = params.ar_LRS_per_V

        self.p = params

    # OPTION 2: CUSTOM BASIC PARAMS
    
    def _init_basic_params(
        self,
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
        fe_kwargs: dict | None = None,
    ):
        
        Gf_LRS, af_LRS = self.get_exp_params(I_on, V_read, G_on)
        Gr_LRS, ar_LRS = asym_G_LRS * Gf_LRS, asym_a_LRS * af_LRS

        Gf_HRS, af_HRS = self.get_exp_params(I_off, V_read, G_off)
        Gr_HRS, ar_HRS = asym_G_HRS * Gf_HRS, asym_a_HRS * af_HRS

        fe_kwargs = fe_kwargs or {}

        self.p = FeDParams(
            Gf_LRS_A=Gf_LRS,
            af_LRS_per_V=af_LRS,
            Gr_LRS_A=Gr_LRS,
            ar_LRS_per_V=ar_LRS,
            Gf_HRS_A=Gf_HRS,
            af_HRS_per_V=af_HRS,
            Gr_HRS_A=Gr_HRS,
            ar_HRS_per_V=ar_HRS,
            V_smooth_V=smoothing_voltage,
            **fe_kwargs
        )

        self.enable_Vshift = False

    # OPTION 3: PRESET FEDs

    def _init_preset_fed(self, model_name: str, variant: int):

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
            
        if model_name not in fed_literature_options:
            raise ValueError(f"Unknown preset device '{model_name}' selected.")
        
        device = fed_literature_options[model_name]

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

        self.p = params
        self.enable_Vshift = False
        

    # -------------------------------
    # Experimental Data Fitting Functions
    # -------------------------------
    
    @staticmethod
    def _fit_one_branch_(V: np.ndarray, I: np.ndarray) -> Dict[str, float]:

        V = np.asarray(V)
        I = np.asarray(I)

        V_abs = np.abs(V - V[0]) #standardized voltages abs value distance from cusp
        G0 = I[0] #I min at cusp
        alpha0 = np.log(I[-1] / G0) / V_abs[-1]

        G_fit, alpha_fit, loss, R2 = fit_IV_curve(V_abs, I, [G0, alpha0])

        LRS_dict = {"G": G_fit, "alpha": alpha_fit, "loss": loss, "R2": R2}

        return LRS_dict

    @staticmethod
    def _create_LRS_from_data(filename: str) -> Dict[str, Dict[str, float]]:

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

        upper_branch_exp_curve = FeD._fit_one_branch_(V_upper, I_upper)
        lower_branch_exp_curve = FeD._fit_one_branch_(V_lower[::-1], I_lower[::-1])

        LRS_params = {
            "upper": upper_branch_exp_curve,
            "lower": lower_branch_exp_curve
        }

        return LRS_params
    
    # -------------------------------
    # Custom Basic Params Helper
    # -------------------------------

    @staticmethod
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

    # ---------------------------
    # READ/WRITE
    # ---------------------------
        
    def read(self, V_read, read_disturb=False):
        I = self.current(self.state, V_read)
        
        if read_disturb:
            noise = np.random.normal(scale=0.01 * abs(I))
        else:
            noise = 0.0
            
        return I + noise
    
    def write(self, V_pulse, pulse_width_uS):
        """
        Apply a voltage pulse and update polarization state if switching condition met.
        """
        Vc = self._Ec_MV_cm_to_V_m(self.p.Ec_MV_cm) * self._t_nm_to_m(self.p.t_nm)
        sw = self.p.sw_thres

        #coercive voltage met
        if abs(V_pulse) <= Vc:
            return False  # no switching

        pulse_integral = (abs(V_pulse) - Vc) * pulse_width_uS

        #pulse integral too low
        if pulse_integral < sw:
            return False
        
        #switching occurs
        if V_pulse > 0:
            self.state = "LRS" #SET
        else:
            self.state = "HRS" #RESET

        return True

    # ---------------------------
    # Unit conversions
    # ---------------------------
    @staticmethod
    def _Pr_uC_cm2_to_C_m2(Pr_uC_cm2: float) -> float:
        # 1 µC/cm^2 = 0.01 C/m^2
        return Pr_uC_cm2 * 0.01

    @staticmethod
    def _Ec_MV_cm_to_V_m(Ec_MV_cm: float) -> float:
        # 1 MV/cm = 1e8 V/m
        return Ec_MV_cm * 1e8

    @staticmethod
    def _t_nm_to_m(t_nm: float) -> float:
        # 1 nm = 10^-9 m
        return t_nm * 1e-9

    # ---------------------------
    # Ferroelectric shift model
    # ---------------------------
    def eps_fe_F_per_m(self) -> float: #absolute permittivity = dielectric constant * eps naught
        return self.p.eps_r_fe * EPS0

    def E_dp_V_per_m(self) -> float: # Computes depolarization field E_dp
        """
        E_dp = E_C * tanh( (β * P_r) / (ε_fe * E_C) )
        """
        Pr = self._Pr_uC_cm2_to_C_m2(self.p.Pr_uC_cm2) #Pr remnant polarization
        Ec = self._Ec_MV_cm_to_V_m(self.p.Ec_MV_cm) #Ec coercive field
        eps = self.eps_fe_F_per_m() #permittivity

        arg = (self.p.beta * Pr) / (eps * Ec)
        return Ec * np.tanh(arg) #tanh formula for E_dp

    def deltaV_V(self) -> float: # Computes voltage shift
        """
        ΔV = E_dp * t
        """
        return self.E_dp_V_per_m() * self._t_nm_to_m(self.p.t_nm)

    # ---------------------------
    # Backbone I–V (state-selectable)
    # ---------------------------

    def _state_current_A(
        self,
        V: np.ndarray,
        *,
        Gf_A: float,
        af_per_V: float,
        Gr_A: float,
        ar_per_V: float,
    ) -> np.ndarray:
        """
        Generic piecewise exponential backbone.

        Uses a smooth |V| for the exponential argument to create a rounded cusp.
        """
        Vs = self.p.V_smooth_V # constant factor that performs cusp smoothing
        abs_V = np.sqrt(V * V + Vs * Vs)

        I = np.empty_like(V) #junk array of same shape V, overwrite all elements
        forward = (V >= 0.0) #bool array, true if forward

        I[forward] = self.p.I_floor_A + Gf_A * np.exp(af_per_V * abs_V[forward]) #forward bias
        I[~forward] = self.p.I_floor_A + Gr_A * np.exp(ar_per_V * abs_V[~forward]) #reverse bias

        return I

    # ---------------------------
    # Public I–V interfaces
    # ---------------------------
    def states(self) -> list[str]:
        """
        Available polarization states
        """
        return ["LRS","HRS"]
    
    def current(self, state: str, V: Union[float, np.ndarray]):
        """
        Current interface w/ state-selection
        """
        V = np.asarray(V, dtype=float)

        if state == "LRS":
            return self._state_current_A(
                V,
                Gf_A = self.p.Gf_LRS_A,
                af_per_V = self.p.af_LRS_per_V,
                Gr_A = self.p.Gr_LRS_A,
                ar_per_V = self.p.ar_LRS_per_V,
                )
        
        if state == "HRS":
            if self.enable_Vshift:    
                Veff = V - self.deltaV_V()
            else:
                Veff = V
            
            return self._state_current_A(
                Veff,
                Gf_A=self.p.Gf_HRS_A,
                af_per_V=self.p.af_HRS_per_V,
                Gr_A=self.p.Gr_HRS_A,
                ar_per_V=self.p.ar_HRS_per_V,
            )