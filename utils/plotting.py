import matplotlib.pyplot as plt
import numpy as np
from models.voltage_shift import FeD

from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D

#old helpers

# -----------------------------
# Directional signed I–V helper
# -----------------------------
def signed_rectified_current(V, mag_I, prefer_positive: bool, Vshift=0.0, leak=1e-6):
    """
    Convert a magnitude curve |I(V)| into a signed, rectified curve.

    prefer_positive=True  => strong current for +V, suppressed for -V
    prefer_positive=False => strong current for -V, suppressed for +V

    leak: fraction of current that leaks in the "off" polarity
    """
    #MODIFIED FROM ORIGINAL: now includes the voltage shift
    V = np.asarray(V, dtype=float)
    V_eff = V - Vshift

    sgn = np.sign(V)
    sgn[sgn == 0.0] = 1.0

    if prefer_positive:
        on = V >= 0.0
    else:
        on = V <= 0.0

    I = np.empty_like(V, dtype=float)
    # ON polarity: full magnitude with sign
    I[on] = mag_I[on] * sgn[on]
    # OFF polarity: strongly suppressed leakage
    I[~on] = leak * mag_I[~on] * sgn[~on]
    return I

# -----------------------------
# Helper: vertical diode symbol + polarization arrow (axes coords)
# -----------------------------

def draw_diode_vertical(ax, center=(0.78, 0.70), size=0.10, color="k", direction="up"):
    """
    Draw a minimalist VERTICAL diode symbol (triangle + bar) in AXES coordinates.
    direction: "up" or "down"
    """
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

    ax.add_line(Line2D([bar[0][0], bar[1][0]], [bar[0][1], bar[1][1]],
                       transform=ax.transAxes, color=color, linewidth=2))

def draw_polarization_arrow(ax, start=(0.90, 0.82), end=(0.90, 0.64), color="k", label=None):
    arr = FancyArrowPatch(start, end, transform=ax.transAxes,
                          arrowstyle="-|>", mutation_scale=18,
                          linewidth=2, color=color)
    ax.add_patch(arr)
    if label is not None:
        ax.text(end[0] + 0.02, (start[1] + end[1]) / 2.0, label,
                transform=ax.transAxes, va="center", ha="left",
                fontsize=11, color=color)
        

# ------------------------
# main plotting functionality
# ------------------------

def plot_fed(
        fed, 
        V=None,
        *,
        show_signed=True,
        log_scale=True,
        title=None
):
    """
    Plot FeD I-V curves
    Produces log scale, optional signed linear scale plots
    """

    if V is None:
        V = np.linspace(-3.0,3.0, 1500)
    
    V = np.asarray(V, dtype=float)

    I_LRS = fed.current("LRS", V)
    I_HRS = fed.current("HRS", V)

    if show_signed:
        fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    else:
        fig, ax_log = plt.subplots(1, 1, figsize=(6.5, 5), constrained_layout=True)

    # log scale magnitude plot

    ax_log.plot(V, np.abs(I_LRS), linewidth=2.5, label="LRS", color="orange")
    ax_log.plot(V, np.abs(I_HRS), linewidth=2.5, label="HRS", color="blue")

    ax_log.set_xlabel("Voltage (V)", fontsize=14)
    ax_log.set_ylabel("|Current| (A)", fontsize=14)

    if log_scale:
        ax_log.set_yscale("log")

    ax_log.grid(True, which="both", alpha=0.25)
    ax_log.legend(frameon=True)
    ax_log.set_title(title or "FeD IV (Log Scale)")

    # signed linear scale plot

    if show_signed:

        # voltage shift to pass into sig_rect_cur 
        Vshift = fed.deltaV_V()

        I_LRS_lin = signed_rectified_current(V, np.abs(I_LRS), prefer_positive=True)
        I_HRS_lin = signed_rectified_current(V, np.abs(I_HRS), prefer_positive=False, Vshift=Vshift)

        ax_lin.plot(V, I_LRS_lin, linewidth=2.5, label="LRS", color="orange")
        ax_lin.plot(V, I_HRS_lin, linewidth=2.5, label="HRS", color="blue")

        ax_lin.axhline(0.0, linewidth=1.0, alpha=0.5)
        ax_lin.set_xlabel("Voltage (V)", fontsize=14)
        ax_lin.set_ylabel("Current (A)", fontsize=14)
        ax_lin.grid(True, which="both", alpha=0.25)
        ax_lin.legend(frameon=True)

        absI = np.abs(np.concatenate([I_LRS_lin, I_HRS_lin]))
        ymax = np.percentile(absI, 99.5)
        ax_lin.set_ylim(-ymax, ymax)

        draw_diode_vertical(ax_lin, center=(0.50, 0.72), size=0.10, color="orange", direction="down")
        draw_polarization_arrow(ax_lin, start=(0.60, 0.82), end=(0.60, 0.64), color="orange", label="Polarization\nDown")

        draw_diode_vertical(ax_lin, center=(0.50, 0.28), size=0.10, color="blue", direction="up")
        draw_polarization_arrow(ax_lin, start=(0.60, 0.18), end=(0.60, 0.36), color="blue", label="Polarization\nUp")

        ax_lin.set_title("Signed Linear IV")

        plt.close(fig)
        return fig
    
def plot_compare(
    fed,
    V_data,
    I_data,
    V_model=None,
    *,
    log_scale=True,
    title=None,
):
    """
    compare experimental or literature IV data with generated IV model, side by side
    Left: data, right: model
    """

    V_data = np.asarray(V_data, dtype=float)
    I_data = np.asarray(I_data, dtype=float)

    if V_model is None:
        V_model = np.linspace(V_data.min(), V_data.max(), 1500)

    I_LRS = fed.current("LRS", V_model)
    I_HRS = fed.current("HRS", V_model)

    fig, (ax_d, ax_m) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # data
    ax_d.plot(V_data, np.abs(I_data), "k.", markersize=4, label="Data")
    ax_d.set_xlabel("Voltage (V)", fontsize=13)
    ax_d.set_ylabel("|Current| (A)", fontsize=13)
    if log_scale:
        ax_d.set_yscale("log")
    ax_d.grid(True, which="both", alpha=0.25)
    ax_d.set_title("Measured IV")
    ax_d.legend(frameon=True)

    # ---- Model ----
    ax_m.plot(V_model, np.abs(I_LRS), color="orange", linewidth=2.5, label="LRS")
    ax_m.plot(V_model, np.abs(I_HRS), color="blue", linewidth=2.5, label="HRS")
    ax_m.set_xlabel("Voltage (V)", fontsize=13)
    if log_scale:
        ax_m.set_yscale("log")
    ax_m.grid(True, which="both", alpha=0.25)
    ax_m.set_title("Model IV")
    ax_m.legend(frameon=True)

    if title:
        fig.suptitle(title, fontsize=15)

    return fig

