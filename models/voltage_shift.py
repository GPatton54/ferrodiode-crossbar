from dataclasses import dataclass
from typing import Union
import numpy as np

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

    def __init__(self, params: FeDParams, *, enable_Vshift: bool = True):
        if not (0.0 < params.beta < 1.0): #beta = [(C_ox / C_fe) + 1]^{-1}
            raise ValueError(f"beta must be in (0,1). Got beta={params.beta}")
        self.p = params
        self.enable_Vshift = enable_Vshift

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