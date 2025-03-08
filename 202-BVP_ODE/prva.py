# Standard imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pomozne_fun import *
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from tqdm import tqdm

from Helpers.plot_metadata import *

# General file settings
mpl.style.use("./porocilo.mplstyle")
pd.set_option("display.max_columns", 50)
# ==================================================================================


def resitev_trajektorija(beta, yf, F0_init, alpha0_init, tolerance=1e-3, max_iter=5000):
    """Reši problem robnih pogojev z iterativnim prilagajanjem ugibanja."""
    F0_guess, alpha0_guess = F0_init, alpha0_init
    for i in range(max_iter):
        sol = solve_ivp(odes, [0, 1], [F0_guess, alpha0_guess, 0, 0], args=(beta,))
        x_end, y_end = sol.y[2, -1], sol.y[3, -1]
        error = calc_diff(x_end, y_end, yf)  # Izračunamo razdaljo do (0, yf)
        # print(f"F0 = {F0_guess}, alpha0 = {alpha0_guess} error = {error}")
        if error <= tolerance:
            print(i)
            return sol, F0_guess, alpha0_guess  # Konvergenca dosežena

        # Iterativno prilagajanje začetnih pogojev
        F0_guess += (0 - x_end) * 0.01
        alpha0_guess += (yf - y_end) * 0.01
    print(error)
    return None, None, None


def draw_traj(sol, beta, yf, ax):
    force = sol.y[0]  # F(s)

    # Interpolacija za gladko krivuljo
    spline_x = CubicSpline(sol.t, sol.y[2])
    spline_y = CubicSpline(sol.t, sol.y[3])
    s_smooth = np.linspace(0, 1, 500)
    x_smooth = spline_x(s_smooth)
    y_smooth = spline_y(s_smooth)

    # Interpolacija sile
    spline_force = CubicSpline(sol.t, force)
    force_smooth = spline_force(s_smooth)

    points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=force_smooth.min(), vmax=force_smooth.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    lc.set_array(force_smooth)
    lc.set_linewidth(2)

    ax.add_collection(lc)
    ax.set_xlim(x_smooth.min() * 1.1, x_smooth.max() * 1.1)
    ax.set_ylim(y_smooth.min() - 0.05, y_smooth.max() + 0.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return lc


""" REŠITVE ZA HOMOGENO VRV, VELIK BETA """
# betas = [10, 60, 120]
# F0_init = 2
# alpha0_init = -0.6
# yf = -0.8
# beta = 20
# savename = "1_traj"

""" REŠITVE ZA HOMOGENO VRV, MALI YF """
betas = [10, 60, 120]
F0_init = 2
alpha0_init = -0.6
yf = -0.3
beta = 20
savename = "1_traj_maliyf"

""" REŠITVE ZA NEHOMOGENO VRV """
# betas = [200, 500, 1000]
# yf = -0.8
# F0_init = 2
# alpha0_init = -0.6
# savename = "1_traj_nehom"

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
for i, beta in enumerate(betas):
    sol, F0_guess, alpha0_guess = resitev_trajektorija(
        beta, yf, F0_init, alpha0_init, tolerance=0.001
    )

    if sol:
        print(f"Konvergirano F0: {F0_guess}, konvergirano alpha0: {alpha0_guess}")
        lc = draw_traj(sol, beta, yf, axs[i])
        axs[i].set_title(
            rf"$\beta$={beta}, F0 = {F0_guess:.3f}, $\alpha_0$ = {alpha0_guess:.3f}"
        )

        fig.colorbar(lc, ax=axs[i], label="Sila (F)")

    else:
        print("Konvergenca ni bila dosežena.")

fig.tight_layout()
savefigure(savename)
plt.show()

# ==================================================================================
# ANALIZA TRAJEKTORIJE


def analiza_trajektorija(beta, yf, F0_init, alpha0_init, tolerance=1e-2, max_iter=5000):
    """Reši problem robnih pogojev z iterativnim prilagajanjem ugibanja."""
    F0_guess, alpha0_guess = F0_init, alpha0_init
    for i in range(max_iter):
        sol = solve_ivp(odes, [0, 1], [F0_guess, alpha0_guess, 0, 0], args=(beta,))
        x_end, y_end = sol.y[2, -1], sol.y[3, -1]
        error = calc_diff(x_end, y_end, yf)  # Izračunamo razdaljo do (0, yf)
        # print(f"F0 = {F0_guess}, alpha0 = {alpha0_guess} error = {error}")
        if error < tolerance:
            x_max = max(sol.y[2])
            y_max = min(sol.y[3])

            print(i)

            # Izračun dolžine trajektorije
            dx_ds = np.cos(sol.y[1])
            dy_ds = np.sin(sol.y[1])
            ds = np.sqrt(dx_ds**2 + dy_ds**2)
            dolzina = np.trapezoid(ds, sol.t)

            # Izračun povprečnega x in y
            x_povprecje = np.trapezoid(sol.y[2], sol.t) / dolzina
            y_povprecje = np.trapezoid(sol.y[3], sol.t) / dolzina
            # Params contains: x_max, y_end-yf, 1 - L, <x>, <y>
            return sol, [
                F0_guess,
                alpha0_guess,
                x_max,
                y_max - yf,
                1 - dolzina,
                x_povprecje,
                y_povprecje,
            ]

        # Iterativno prilagajanje začetnih pogojev
        F0_guess += (0 - x_end) * 0.01
        alpha0_guess += (yf - y_end) * 0.01

    return None, [0, 0, 0, 0, 0, 0, 0]


betas = np.linspace(10, 1500, 50)
yf = -0.3
F0_init = 2
alpha0_init = -0.6

rezultati = [
    analiza_trajektorija(beta, yf, F0_init, alpha0_init)[1] for beta in tqdm(betas)
]
indeksi_za_odstranitev = []
for i, element in enumerate(rezultati):
    if element == [0, 0, 0, 0, 0, 0, 0]:
        indeksi_za_odstranitev.append(i)

seznam_filtrirano = [
    element for i, element in enumerate(rezultati) if i not in indeksi_za_odstranitev
]
drugi_seznam_filtrirano = np.delete(betas, indeksi_za_odstranitev)
podatki = list(zip(*rezultati))


# Imena osi y za grafe
labels = ["F(0)", r"$\alpha(0)$", "x_max", "y_max - y_f", "1 - L", "<x>", "<y>"]

# Izris grafov
fig, axs = plt.subplots(nrows=7, figsize=(15, 18), sharex=True)
for ax, data, label in zip(axs, podatki, labels):
    ax.plot(betas, data, "-o")
    ax.set_ylabel(label)

plt.xlabel("beta")
plt.tight_layout()
savefigure("1_analiza_maliyf")
plt.show()

# ==================================================================================
# ANALIZA STABILNOSTI


def resi_problem(F0_guess, alpha0_guess, beta, yf, tolerance=1e-1):
    sol = solve_ivp(odes, [0, 1], [F0_guess, alpha0_guess, 0, 0], args=(beta,))
    x_end, y_end = sol.y[2, -1], sol.y[3, -1]
    odmik = calc_diff(x_end, y_end, yf)
    return odmik, odmik < tolerance


# Parametri
beta = 80
yf = -0.6
F0_values = np.logspace(-1, 2, 100)
alpha0_values = np.linspace(-np.pi / 2, 0, 100)

# Izračun odmikov
odmiki = np.zeros((len(alpha0_values), len(F0_values)))
konvergenca = np.zeros((len(alpha0_values), len(F0_values)))

for i, alpha0 in tqdm(enumerate(alpha0_values)):
    for j, F0 in enumerate(F0_values):
        odmiki[i, j], konvergenca[i, j] = resi_problem(F0, alpha0, beta, yf)

# Izris imshow
plt.imshow(
    odmiki,
    extent=[F0_values.min(), F0_values.max(), alpha0_values.max(), alpha0_values.min()],
    aspect="auto",
    cmap="viridis",
)
plt.xlabel("F0_guess")
plt.ylabel("alpha0_guess")
plt.title("Končni odmik od (0, yf)")
plt.colorbar(label="Odmik")
savefigure("1_stabilnost80")
plt.show()


plt.matshow(
    konvergenca,
    extent=[
        np.log10(F0_values.min()),
        np.log10(F0_values.max()),
        alpha0_values.max(),
        alpha0_values.min(),
    ],
    aspect="auto",
    cmap="gray",
)
plt.xlabel("F0_guess")
plt.ylabel("alpha0_guess")
plt.title("Konvergenca (True/False)")
plt.show()
