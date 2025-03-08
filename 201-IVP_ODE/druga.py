# Standard imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pomozne_eq import *

from Helpers.io_functions import load_results, save_results
from Helpers.plot_metadata import *

# General file settings
mpl.style.use("./porocilo.mplstyle")
pd.set_option("display.max_columns", 50)
# ==================================================================================


# ================================================
G = 1.0
M = 1.0
epsilon = 1e-5
# ================================================


def gravitational_force(x, y, x_sun, y_sun):
    dx = x - x_sun
    dy = y - y_sun
    r_squared = dx**2 + dy**2 + epsilon**2
    r = np.sqrt(r_squared)
    fx = -G * M * dx / r**3
    fy = -G * M * dy / r**3
    return fx, fy


def equations_of_motion(t, state, R, theta):
    x, y, vx, vy = state
    # print(f"Theta = {theta}")
    x_sun1 = R * np.cos(theta + t)
    y_sun1 = R * np.sin(theta + t)
    x_sun2 = -x_sun1
    y_sun2 = -y_sun1

    fx1, fy1 = gravitational_force(x, y, x_sun1, y_sun1)
    fx2, fy2 = gravitational_force(x, y, x_sun2, y_sun2)

    ax = fx1 + fx2
    ay = fy1 + fy2

    return [vx, vy, ax, ay]


# NARISANE TRAJEKTORIJE
def plot_trajectory(vy0, R, theta, ax=None, tk=30, t_points=1000):
    vx0 = 0.0
    planet_initial_conditions = [1.0, 0.0, vx0, vy0]
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    print("Solving")
    solution = solve_ivp(
        lambda t, state: equations_of_motion(t, state, R, theta),
        t_span,
        planet_initial_conditions,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-10,
        atol=1e-10,
    )
    print("solved")

    x_traj = solution.y[0]
    y_traj = solution.y[1]

    ax = ax or plt.gca()
    # Začetne pozicije
    ax.scatter(planet_initial_conditions[0], planet_initial_conditions[1], color="blue")
    ax.scatter(R * np.cos(theta), R * np.sin(theta), color="green")
    ax.scatter(-R * np.cos(theta), -R * np.sin(theta), color="green")

    ax.plot(x_traj, y_traj, label="Planet")
    ax.plot(
        R * np.cos(theta + t_eval),
        R * np.sin(theta + t_eval),
        color="green",
        label="Sonce 1",
    )
    ax.plot(
        -R * np.cos(theta + t_eval),
        -R * np.sin(theta + t_eval),
        color="red",
        label="Sonce 2",
    )

    format_plot(
        ax=ax,
        xlabel="x",
        ylabel="y",
        title=rf"$\theta$ = {theta}, R = {R}, $v_z$ = {vy0}",
    )
    ax.axis("equal")


vy0 = [1, 2, 1]  # Collision, escape, "normal"
R = [0.5, 0.2, 0.9]
theta = [0.2, 0, 0.1]

fig, axs = plt.subplots(nrows=1, ncols=3)
for i in range(len(R)):
    plot_trajectory(vy0[i], R[i], theta[i], ax=axs[i])
axs[1].legend()
fig.tight_layout()
savefigure("2_orbits")
plt.show()

# ===================================================================
# VEZANOST SISTEMA
vy0 = 1
tk = 30
t_points = 1000


def solve_single(R, theta, t_span, t_eval):
    vx0 = 0.0
    planet_initial_conditions = [1.0, 0.0, vx0, vy0]
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    solution = solve_ivp(
        lambda t, state: equations_of_motion(t, state, R, theta),
        t_span,
        planet_initial_conditions,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-10,
        atol=1e-10,
    )
    print(f"Finished ODE solution for R={R}, theta={theta}")

    states = solution.y.T
    energija_sez = [energija(state) for state in states]
    return energija_sez[-1], states[-1]


# Main function
def vezan_sistem(Rs, thetas, tk=50, t_points=100):
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Use Parallel to solve for all (R, theta) pairs in parallel
    results_ = Parallel(n_jobs=-1)(  # n_jobs=-1 uses all available CPU cores
        delayed(solve_single)(R, theta, t_span, t_eval) for R in Rs for theta in thetas
    )
    results = [result[0] for result in results_]
    states = [result[1] for result in results_]
    print(states)
    # Reshape results into matrices
    eng_sez = np.array(results).reshape(len(Rs), len(thetas))
    vez_sez = (eng_sez < 0).astype(int)

    return vez_sez, eng_sez, states


def run_sim(Rs, thetas, filename=None):
    filename = "Sistem_2_sonci.pkl"

    # Preveri, ali datoteka že obstaja
    vez_sez, eng_sez, states = load_results(filename)

    if vez_sez is None or eng_sez is None or states is None:
        print("Datoteka ne obstaja. Izvajanje funkcije vezan_sistem...")
        vez_sez, eng_sez, states = vezan_sistem(Rs, thetas)
        save_results(vez_sez, eng_sez, states, filename)
    else:
        print("Podatki so bili naloženi iz obstoječe datoteke.")

    return vez_sez, eng_sez, states


Rs = np.linspace(0, 2, 50)
thetas = np.linspace(0, np.pi, 50)

vez_sez, eng_sez, states = run_sim(Rs, thetas)

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].matshow(vez_sez, origin="lower")

eng_sez = np.log(eng_sez)
ax = axs[1]
im = axs[1].matshow(eng_sez, origin="lower", norm=LogNorm())
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
axs[1].matshow(eng_sez, origin="lower")


# Nastavitev labels
ticks = [0, 10, 20, 30, 40, 50]
x_labels = ["0", r"$\pi/5$", r"$2\pi/5$", r"$3\pi/5$", r"$4\pi/5$", r"$\pi$"]
y_labels = ["0", "0.4", "0.8", "1.2", "1.6", "2"]

axs[0].set_xticks(ticks, x_labels)
axs[0].set_yticks(ticks, y_labels)
axs[0].xaxis.set_ticks_position("bottom")
format_plot(ax=axs[0], xlabel=r"$\theta$", ylabel="R")

axs[1].set_xticks(ticks, x_labels)
axs[1].set_yticks(ticks, y_labels)
axs[1].xaxis.set_ticks_position("bottom")
format_plot(ax=axs[1], xlabel=r"$\theta$", ylabel="R")


savefigure("2_VezanostSistema")
plt.show()
