import matplotlib.pyplot as plt
import numpy as np
from models.voltage_shift import FeD
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D


def plot_IV(
        fed: FeD, 
        V: np.ndarray=None,
        title: str = "FeD IV Curves",
        log_scale: bool = True,
): 
    
    #default voltage array if none provided
    if V is None:
        V = np.linspace(-2.0,2.0,1000)

    # current arrays, HRS and LRS
    I_LRS = fed.current("LRS",V)
    I_HRS = fed.current("HRS",V)
    
    #figure setup
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    
    #log scale mag
    axes[0].plot(V, np.abs(I_LRS), color="orange", label="LRS")
    axes[0].plot(V, np.abs(I_HRS), color="blue", label="HRS")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("|Current| (A)")
    axes[0].set_title("Log scale IV")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    #linear scale signed
    axes[1].plot(V, I_LRS, color="orange", label="LRS")
    axes[1].plot(V, I_HRS, color="blue", label="HRS")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Linear scale IV")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    return fig

def plot_IV_data(V: np.ndarray, I: np.ndarray, figname: str = "IV Data Curve", log_scale: bool = True):
    fig = plt.figure(figsize=(6,4))

    if log_scale:
        plt.semilogy(V, np.abs(I), 'o', label=figname)
        plt.ylabel("|Current| (A)")
    else:
        plt.plot(V, I, 'o', label=figname)
        plt.ylabel("Current (A)")

    return fig

def plot_IV_sbs(
    V: np.ndarray,
    I_exp: np.ndarray,
    I_model: np.ndarray,
    title: str = "Experimental vs Model IV Curves",
    log_scale: bool = True,
    labels: tuple[str,str] = ("Experimental","Model")
):
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    
    #logscale mag
    if log_scale:
        axes[0].semilogy(V, np.abs(I_exp), 'o', label=labels[0])
        axes[0].semilogy(V, np.abs(I_model), '-', label=labels[1])
        axes[0].set_ylabel("|Current| (A)")
        axes[0].set_xlabel("Voltage (V)")
        axes[0].set_title("Log scale IV")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
    else:
        axes[0].plot(V, I_exp, 'o', label=labels[0])
        axes[0].plot(V, I_model, '-', label=labels[1])
        axes[0].set_ylabel("Current (A)")
        axes[0].set_xlabel("Voltage (V)")
        axes[0].set_title("Linear scale IV")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
    #linear scale, signed
    axes[1].plot(V, I_exp, 'o', label=labels[0])
    axes[1].plot(V, I_model, '-', label=labels[1])
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Linear scale IV")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    fig.suptitle(title)


    return fig

def plot_IV_fit_overlay(V: np.ndarray, I_experimental: np.ndarray, G_fit: float, alpha_fit: float, loss: float, R2: float, log_scale: bool = True):
    
    I_fit = G_fit * np.exp(alpha_fit * np.abs(V))
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)
    
    # log-scale magnitude
    if log_scale:
        axes[0].semilogy(V, np.abs(I_experimental), 'o', label="Experimental")
        axes[0].semilogy(V, np.abs(I_fit), '-', label="Fitted")
        axes[0].set_ylabel("|Current| (A)")
    else:
        axes[0].plot(V, I_experimental, 'o', label="Experimental")
        axes[0].plot(V, I_fit, '-', label="Fitted")
        axes[0].set_ylabel("Current (A)")
    
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_title("log/linear fit overlay")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    
    # linear-scale signed
    axes[1].plot(V, I_experimental, 'o', label="Experimental")
    axes[1].plot(V, I_fit, '-', label="Fitted")
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("linear scale")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # put metrics on figure
    fig.suptitle(f"Loss = {loss:.2e}, R^2 = {R2:.4f}")
    
    return fig

def signed_rectified_current(V, mag_I, prefer_positive: bool, leak=1e-6):
    V = np.asarray(V, dtype=float)
    sgn = np.sign(V)
    sgn[sgn == 0.0] = 1.0

    if prefer_positive:
        on = V >= 0.0
    else:
        on = V <= 0.0

    I = np.empty_like(V, dtype=float)
    I[on] = mag_I[on] * sgn[on]
    I[~on] = leak * mag_I[~on] * sgn[~on]
    return I

def draw_diode_vertical(ax, center=(0.78, 0.70), size=0.10, color="k", direction="up"):
    cx, cy = center
    s = size
    if direction == "up":
        tri = np.array([[cx - s*0.45, cy - s*0.60],
                        [cx + s*0.45, cy - s*0.60],
                        [cx,          cy + s*0.15]])
        bar_y = cy + s*0.25
        bar = [(cx - s*0.55, bar_y), (cx + s*0.55, bar_y)]
    else:  # down
        tri = np.array([[cx - s*0.45, cy + s*0.60],
                        [cx + s*0.45, cy + s*0.60],
                        [cx,          cy - s*0.15]])
        bar_y = cy - s*0.25
        bar = [(cx - s*0.55, bar_y), (cx + s*0.55, bar_y)]

    tri_patch = Polygon(tri, closed=True, fill=False, edgecolor=color, linewidth=2, transform=ax.transAxes)
    ax.add_patch(tri_patch)
    ax.add_line(Line2D([bar[0][0], bar[1][0]], [bar[0][1], bar[1][1]], transform=ax.transAxes, color=color, linewidth=2))

def draw_polarization_arrow(ax, start=(0.90, 0.82), end=(0.90, 0.64), color="k", label=None):
    arr = FancyArrowPatch(start, end, transform=ax.transAxes,
                          arrowstyle="-|>", mutation_scale=18,
                          linewidth=2, color=color)
    ax.add_patch(arr)
    if label is not None:
        ax.text(end[0] + 0.02, (start[1] + end[1]) / 2.0, label,
                transform=ax.transAxes, va="center", ha="left",
                fontsize=11, color=color)

def plot_IV_signed_with_arrows(fed, V=None):
    if V is None:
        V = np.linspace(-1.5, 3.0, 1200)

    # Use current() interface
    I_LRS_mag = fed.current("LRS", V)
    I_HRS_mag = fed.current("HRS", V)
    dV = fed.deltaV_V()  # optional if you want shifted annotation

    # Linear signed curves
    I_LRS_lin = signed_rectified_current(V, I_LRS_mag, prefer_positive=True, leak=3e-6)
    I_HRS_lin = signed_rectified_current(V, I_HRS_mag, prefer_positive=False, leak=3e-6)

    # Create figure
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Log plot
    ax_log.plot(V, I_LRS_mag, linewidth=2.5, label="LRS (Unshifted)", color='orange')
    ax_log.plot(V, I_HRS_mag, linewidth=2.5, label="HRS (Shifted by ΔV)", color='blue')
    ax_log.set_xlabel("Voltage (V)")
    ax_log.set_ylabel("Abs Current (A)")
    ax_log.set_yscale("log")
    ax_log.grid(True, which="both", alpha=0.25)
    ax_log.legend()

    # Linear plot
    ax_lin.plot(V, I_LRS_lin, linewidth=2.5, label="Polarization Down / LRS", color='orange')
    ax_lin.plot(V, I_HRS_lin, linewidth=2.5, label="Polarization Up / HRS", color='blue')
    ax_lin.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax_lin.grid(True, which="both", alpha=0.25)
    ax_lin.legend()

    # Decorations
    draw_diode_vertical(ax_lin, center=(0.5, 0.72), size=0.10, color="orange", direction="down")
    draw_polarization_arrow(ax_lin, start=(0.60, 0.82), end=(0.60, 0.64), color="orange", label="Polarization Down")
    draw_diode_vertical(ax_lin, center=(0.5, 0.28), size=0.10, color="blue", direction="up")
    draw_polarization_arrow(ax_lin, start=(0.60, 0.18), end=(0.60, 0.36), color="blue", label="Polarization Up")

    ax_lin.set_ylim(-np.percentile(np.abs(np.concatenate([I_LRS_lin, I_HRS_lin])), 99.5),
                    np.percentile(np.abs(np.concatenate([I_LRS_lin, I_HRS_lin])), 99.5))

    ax_log.set_title("Log Scale: LRS vs HRS")
    ax_lin.set_title("Linear Scale: Signed I–V")

    return fig
