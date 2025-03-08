import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from pomozne_eq import *
from scipy.integrate import solve_ivp

from Helpers.plot_metadata import *

# General file settings
mpl.style.use("./porocilo.mplstyle")
pd.set_option("display.max_columns", 50)
# ==================================================================================


# ================================================
G = 1.0
M = 1.0
no_zero_div = 1e-5  # Softening parameter to avoid division by zero
R = 1.0
# ================================================


def equations_of_motion(t, state):
    x, y, vx, vy = state

    # Pozicija prve zvezde
    x_sun1 = 0.0
    y_sun1 = 0.0

    # Pozicija druge zvezde x(t) = -10 + 2t, y(t) = 1.5)
    x_sun2 = -10 + 2 * t
    y_sun2 = 1.5

    fx1, fy1 = gravitational_force(x, y, x_sun1, y_sun1)
    fx2, fy2 = gravitational_force(x, y, x_sun2, y_sun2)

    ax = fx1 + fx2
    ay = fy1 + fy2

    return [vx, vy, ax, ay]


def plot_trajectory(vy0, theta, ax=None, tk=50, t_points=1000):
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    # Zacetni pogoji planeta
    x0 = R * np.cos(theta)
    y0 = R * np.sin(theta)
    vx0 = 0
    planet_initial_conditions = [x0, y0, vx0, vy0]

    print(f"Solving for theta = {theta}...")
    solution = solve_ivp(
        equations_of_motion,
        t_span,
        planet_initial_conditions,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-10,
        atol=1e-10,
    )
    print("Solved.")

    x_traj = solution.y[0]
    y_traj = solution.y[1]

    ax = ax or plt.gca()
    ax.plot(x_traj, y_traj, label="Planet", color="blue")

    ax.scatter([0], [0], color="red", label="Fiksna zvezda")
    ax.plot(
        -10 + 2 * t_eval, [1.5] * len(t_eval), color="green", label="Mimobežna zvezda"
    )

    format_plot(
        ax=ax,
        xlabel="x",
        ylabel="y",
        title=rf"$\theta$ = {theta}, $v_z$ = {vy0}",
    )
    ax.axis("equal")


theta = [-0.4, -0.3, 0]  #
vy0 = [-1, -1, 1]
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
for i in range(len(theta)):
    plot_trajectory(vy0[i], theta[i], ax=axs[i])
axs[1].legend()
axs[2].set_xlim(-2, 2)
fig.tight_layout()
savefigure("3_orbits")
plt.show()


# VEZANOST SISTEMA
vy0 = 1
tk = 30
t_points = 1000


def energija_3Body(state, t):
    x, y, vx, vy = state
    r1 = np.sqrt(x**2 + y**2 + no_zero_div**2)
    r2 = np.sqrt((x - (-10 + 2 * t)) ** 2 + (y - 1.5) ** 2 + no_zero_div**2)
    T = 0.5 * (vx**2 + vy**2)
    U = -G * M / r1 - G * M / r2
    return T + U


def solve_single(vy0, theta, t_span, t_eval, R=1):
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    x0 = R * np.cos(theta)
    y0 = R * np.sin(theta)
    vx0 = 0
    planet_initial_conditions = [x0, y0, vx0, vy0]

    solution = solve_ivp(
        equations_of_motion,
        t_span,
        planet_initial_conditions,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-5,
        atol=1e-5,
    )
    print(f"Finished ODE solution for vy={vy0}, theta={theta}")

    final_state = solution.y[:, -1]
    final_time = t_eval[-1]
    final_energy = energija_3Body(final_state, final_time)

    return final_energy, final_state


def vezan_sistem(Rs, thetas, tk=50, t_points=100):
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    sist_matrix = np.zeros((len(Rs), len(thetas)))
    eng_sez = np.zeros((len(Rs), len(thetas)))

    results = Parallel(n_jobs=-1)(  # n_jobs=-1 uses all available CPU cores
        delayed(solve_single)(R, theta, t_span, t_eval)
        for i, R in enumerate(Rs)
        for j, theta in enumerate(thetas)
    )

    for idx, (final_energy, final_state) in enumerate(results):
        i = idx // len(thetas)
        j = idx % len(thetas)

        eng_sez[i, j] = final_energy

        # Compute final distances to stars
        final_distance_to_star1 = np.sqrt(final_state[0] ** 2 + final_state[1] ** 2)
        final_distance_to_star2 = np.sqrt(
            (final_state[0] - (-10 + 2 * t_eval[-1])) ** 2 + (final_state[1] - 1.5) ** 2
        )

        # Determine if the planet is bound or escaped
        if final_energy < 0:
            if final_distance_to_star1 < final_distance_to_star2:
                sist_matrix[i, j] = -1  # Bound to Star 1
            else:
                sist_matrix[i, j] = 1  # Bound to Star 2
        else:
            sist_matrix[i, j] = 0  # Escaped
        states = 0
    return sist_matrix, eng_sez, states


def run_sim(Rs, thetas, filename=None):
    filename = "Sistem_3_telesa.pkl"

    # Preveri, ali datoteka že obstaja
    vez_sez, eng_sez, states = load_results(filename)

    if vez_sez is None or eng_sez is None or states is None:
        print("Datoteka ne obstaja. Izvajanje funkcije vezan_sistem...")
        vez_sez, eng_sez, states = vezan_sistem(Rs, thetas)
        save_results(vez_sez, eng_sez, states, filename)
    else:
        print("Podatki so bili naloženi iz obstoječe datoteke.")

    return vez_sez, eng_sez, states


Rs = np.linspace(-1, 1, 100)
thetas = np.linspace(0, np.pi, 100)

sist_matrix, eng_sez, states = run_sim(Rs, thetas)
print("Sist matrix (Bound to Star 1: -1, Bound to Star 2: 1, Escaped: 0):")
print(sist_matrix)

cmap = ListedColormap(["purple", "yellow", "red"])  # Barve za -1, 0, 1
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = plt.Normalize(vmin=-1, vmax=1)

fig, ax = plt.subplots()
cax = ax.matshow(sist_matrix, cmap=cmap, norm=norm)
cbar = fig.colorbar(cax, ax=ax, ticks=[-1, 0, 1])
cbar.set_ticklabels(["Priključen stac. zvezdi", "Planet ušel", "Priključen mim. zvezdi"])

# Nastavitev labels
ticks = [0, 20, 40, 60, 80, 100]  # Prilagodi glede na podatke
x_labels = ["0", r"$\pi/5$", r"$2\pi/5$", r"$3\pi/5$", r"$4\pi/5$", r"$\pi$"]
y_labels = ["0", "0.4", "0.8", "1.2", "1.6", "2"]

plt.xticks(ticks, x_labels)
plt.yticks(ticks, y_labels)
# plt.xaxis.set_ticks_position("bottom")
format_plot(xlabel=r"$\theta$", ylabel="v")

savefigure("3_Vezava_sist")
plt.show()
