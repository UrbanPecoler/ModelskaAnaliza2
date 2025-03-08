# Standard imports
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pomozne_eq import *

from Helpers.plot_metadata import *

# General file settings
mpl.style.use("./porocilo.mplstyle")
pd.set_option("display.max_columns", 50)
# ==================================================================================

"""1. NALOGA"""


# Krožna orbita za različne integratorje
def krozna_orbita(vz, dt, tk, save=False):
    zac_pogoji = [1, 0, 0, vz]  # x(0) = 1, y(0) = 0, u(0) = 0, v(0) = vz
    methods = ["RK45", "Verlet", "Radau"]
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)
    for i, method in enumerate(methods):
        sol = solve_ode(zac_pogoji, dt, tk=60, method=method)
        axs[i].plot(sol[:, 0], sol[:, 1], label=f"Metoda -- {method}")
        axs[i].set(adjustable="box", aspect="equal")
        format_plot(ax=axs[i], xlabel="x", title=f"Metoda -- {method}")
    axs[0].set_ylabel("y")
    fig.tight_layout()
    if save:
        savefigure("1_krozna_integratorji")
    plt.show()


vz = 1
dt = 0.1
tk = 30
# krozna_orbita(vz, dt, tk, save=True)


# Različne orbite -- integrator Verlet
def orbita_v(v0s, save=False):
    method = "Verlet"
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=v0s.min(), vmax=v0s.max())
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(10, 8))
    for vz in v0s:
        zac_pogoji = [1, 0, 0, vz]
        tk = 10 if vz == v0s[0] else 50
        sol = solve_ode(zac_pogoji, dt, tk=tk, method=method)

        ax.plot(sol[:, 0], sol[:, 1], color=cmap(norm(vz)))

    ax.scatter(0, 0, color="red")
    ax.text(0.2, -0.1, "M")
    ax.set_xlim((-15, 2))
    ax.set_ylim((-3, 15))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$ v_z $")
    format_plot(ax=ax, xlabel="x", ylabel="y")
    ax.set_xlim(-6.5, 1.5)
    ax.set_ylim(-2, 6)
    if save:
        savefigure("1_razlicne_orbite")
    plt.show()


v0s = np.linspace(0.7, 2, 10)
# orbita_v(v0s, save=True)


# OHRANJANJE OBHODNEGA ČASA IN TOČKE PREHODA
def analyze_orbit_error(
    initial, dt=0.01, num_orbits=100, epsilon=0, methods=None, dt_values=None, RK45=False
):
    methods = ["Verlet", "RK45", "Radau"]
    results = {}

    a = np.linalg.norm(initial[:2])
    T_theoretical = 2 * np.pi * a**1.5
    t_max = num_orbits * T_theoretical

    if dt_values is not None:
        methods = ["RK45"] if RK45 else ["Verlet"]
        dt_list = dt_values
    else:
        methods = methods or ["Verlet", "RK45", "Radau"]
        dt_list = [dt]

    for dt in dt_list:
        for method in methods:
            solution = solve_ode(initial, dt, tk=t_max, method=method, epsilon=epsilon)
            t = np.arange(0, t_max, dt)

            x, y = solution[:, 0], solution[:, 1]
            # vx, vy = solution[:, 2], solution[:, 3]

            crossings = []
            times = []

            for i in range(1, len(y)):
                if y[i - 1] * y[i] < 0 and x[i] >= 0:
                    # Prehod čez y = 0 na pozitivni strani
                    # Interpolacija za boljšo oceno prehoda čez y=0
                    t_cross = t[i - 1] - y[i - 1] * (t[i] - t[i - 1]) / (y[i] - y[i - 1])
                    x_cross = x[i - 1] + (x[i] - x[i - 1]) * (t_cross - t[i - 1]) / (
                        t[i] - t[i - 1]
                    )
                    y_cross = 0
                    crossings.append(np.sqrt(x_cross**2 + y_cross**2))
                    times.append(t_cross)

            num_detected_orbits = len(times)
            if num_detected_orbits < num_orbits:
                print(
                    f"Warning: {method} completed only "
                    f"{num_detected_orbits}/{num_orbits} orbits."
                )

            r0_error = np.abs(np.array(crossings) - np.linalg.norm(initial[:2]))
            T_numerical = np.diff(times)
            T_error = np.abs(T_numerical - T_theoretical) / T_theoretical

        if dt_values is not None:
            results[str(dt)] = {
                "r0_error": r0_error[:num_detected_orbits],
                "T_numerical": T_numerical[:num_detected_orbits],
                "T_error": T_error[:num_detected_orbits],
            }
        else:
            results[method] = {
                "r0_error": r0_error[:num_detected_orbits],
                "T_numerical": T_numerical[:num_detected_orbits],
                "T_error": T_error[:num_detected_orbits],
            }
    return results


# --- Primerjava Različnih metod pri konst dt ---
initial_conditions = [1, 0, 0, 1]
dt = 0.01
num_orbits = 80

# # --- Primerjava Verlet pri različnih dt ---
dt_values = [0.05, 0.02, 0.01, 0.005]
results_verlet = analyze_orbit_error(
    initial_conditions, num_orbits=num_orbits, dt_values=dt_values
)
results_RK45 = analyze_orbit_error(
    initial_conditions, num_orbits=num_orbits, dt_values=dt_values, RK45=True
)

# Prikaz rezultatov -- Prehod koordinate
save = True
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
for method, data in results_verlet.items():
    num_orbits_completed = len(data["r0_error"])
    axs[0].plot(
        range(1, num_orbits_completed + 1), data["r0_error"], label=f"dt = {method}"
    )
for method, data in results_RK45.items():
    num_orbits_completed = len(data["r0_error"])
    axs[1].plot(
        range(1, num_orbits_completed + 1), data["r0_error"], label=f"dt = {method}"
    )
format_plot(
    ax=axs[0],
    xlabel="Orbita",
    ylabel="Napaka prehoda",
    yscale="log",
    title="Verlet",
    legend=True,
)
format_plot(
    ax=axs[1],
    xlabel="Orbita",
    ylabel="Napaka prehoda",
    yscale="log",
    title="RK45",
    legend=True,
)
if save:
    savefigure("1_NapakaPrehoda_x")
plt.show()

# Prikaz rezultatov -- čas prehoda
save = True
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
for method, data in results_verlet.items():
    num_orbits_completed = len(data["r0_error"])
    axs[0].plot(range(1, num_orbits_completed), data["T_error"], label=f"dt = {method}")
for method, data in results_RK45.items():
    num_orbits_completed = len(data["r0_error"])
    axs[1].plot(range(1, num_orbits_completed), data["T_error"], label=f"dt = {method}")
format_plot(
    ax=axs[0],
    xlabel="Orbita",
    ylabel="Napaka časa prehoda",
    yscale="log",
    title="Verlet",
    legend=True,
)
format_plot(
    ax=axs[1],
    xlabel="Orbita",
    ylabel="Napaka časa prehoda",
    yscale="log",
    title="RK45",
    legend=True,
)
if save:
    savefigure("1_NapakaČasaPrehoda")
plt.show()


# # POINCAREJEV PRESEK
# """ TODO !!! """

# # 3. KEPLERJEV ZAKON
# """ TODO !!! """


# # OHRANJANJE KOLIČIN
def plot_ohranitev_kolicin(vz, save=False):
    dt = 0.1
    tk = 30

    zac_pogoji = [1, 0, 0, vz]  # x(0) = 1, y(0) = 0, u(0) = 0, v(0) = vz
    t_span = np.arange(0, tk, dt)
    sol_RK45 = solve_ode(zac_pogoji, dt, tk=tk, method="RK45")
    sol_Radau = solve_ode(zac_pogoji, dt, tk=tk, method="Radau")
    sol_Verlet = solve_ode(zac_pogoji, dt, tk=tk, method="Verlet")

    RK45_energija_sez = [energija(state) for state in sol_RK45]
    RK45_vk_sez = [vrtilna_kol(state) for state in sol_RK45]
    RK45_rl_sez = [runge_lenz(state) for state in sol_RK45]

    Radau_energija_sez = [energija(state) for state in sol_Radau]
    Radau_vk_sez = [vrtilna_kol(state) for state in sol_Radau]
    Radau_rl_sez = [runge_lenz(state) for state in sol_Radau]

    Verlet_energija_sez = [energija(state) for state in sol_Verlet]
    Verlet_vk_sez = [vrtilna_kol(state) for state in sol_Verlet]
    Verlet_rl_sez = [runge_lenz(state) for state in sol_Verlet]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(25, 4))
    # Energija
    axs[0].plot(t_span, RK45_energija_sez, label="RK45")
    axs[0].plot(t_span, Radau_energija_sez, label="Radau")
    axs[0].plot(t_span, Verlet_energija_sez, label="Verlet")
    format_plot(
        ax=axs[0],
        xlabel="t",
        ylabel="Energija",
        title=rf"Energija - $v_z =$ {vz}",
        legend=True,
    )

    # Vrtilna kolicina
    axs[1].plot(t_span, RK45_vk_sez, label="RK45")
    axs[1].plot(t_span, Radau_vk_sez, label="Radau")
    axs[1].plot(t_span, Verlet_vk_sez, label="Verlet")
    format_plot(
        ax=axs[1],
        xlabel="t",
        ylabel="Energija",
        title=rf"Vrtilna količina - $v_z =$ {vz}",
    )

    # Runge-Lenz
    axs[2].plot(t_span, RK45_rl_sez, label="RK45")
    axs[2].plot(t_span, Radau_rl_sez, label="Radau")
    axs[2].plot(t_span, Verlet_rl_sez, label="Verlet")
    format_plot(
        ax=axs[2],
        xlabel="t",
        ylabel="Energija",
        title=rf"Velikost Runge-Lenz - $v_z =$ {vz}",
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, wspace=0.4)
    if save:
        savefigure(f"1_Ohranitev_v-{str(vz).replace(".", "")}")
    plt.show()


vz = np.sqrt(2)
# plot_ohranitev_kolicin(vz)


# NOV POTENCIAL
# NARISANE TRAJEKTORIJE
def plot_trajectory(vz, epsilon, tk, ax=None, t_points=1000):
    initial_conditions = [1, 0, 0, vz]
    t_span = (0, tk)
    t_eval = np.linspace(t_span[0], t_span[1], t_points)
    print("Solving")
    solution = solve_ivp(
        lambda t, state: odes(t, state, epsilon),
        t_span,
        initial_conditions,
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
    ax.scatter(initial_conditions[0], initial_conditions[1], color="blue")
    ax.scatter(0, 0, color="green")

    ax.plot(x_traj, y_traj, label="Planet")
    format_plot(
        ax=ax,
        xlabel="x",
        ylabel="y",
        title=rf"$v_z$ = {vz}, $\epsilon$ = {epsilon}",
        legend=True,
    )
    ax.axis("equal")


epsilon = [0.1, 0.3, 0.5]
vz = 1
tk = [30, 50, 30]
fig, axs = plt.subplots(nrows=1, ncols=3)
for i in range(len(epsilon)):
    plot_trajectory(vz, epsilon[i], tk[i], ax=axs[i])
fig.tight_layout()
savefigure(f"1_ModPotential_vz-{str(vz).replace(".", "")}")
plt.show()

# ČASOVNO ODVISNI POTENCIAL
""" TODO !!! """
