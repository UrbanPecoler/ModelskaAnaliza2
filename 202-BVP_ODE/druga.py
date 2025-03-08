import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pomozne_fun import *
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline, interp1d
from tqdm import tqdm

from Helpers.plot_metadata import *

mpl.style.use("./porocilo.mplstyle")
# =========================================================


def plot_potential(save=False):
    x = np.linspace(-1.5, 1.5, 20)
    y = np.linspace(-1.5, 1.5, 20)
    X, Y = np.meshgrid(x, y)

    Z = henon_potential(X, Y)
    dV_dx, dV_dy = henon_gradient(X, Y)

    fig, axs = plt.subplots(ncols=2, nrows=1)
    axs[0].contour(X, Y, Z, 40, colors="black")
    axs[0].contourf(X, Y, Z, 40, cmap="viridis")
    format_plot(
        ax=axs[0],
        xlabel="x",
        ylabel="y",
        title="Henonov potencial (ekvipotencialne črte)",
    )

    pcm = axs[1].contourf(X, Y, Z, 50, cmap="viridis")
    fig.colorbar(pcm, label="Henonov potencial", ax=axs[1])

    # Nariši vektorje
    axs[1].quiver(
        X, Y, -dV_dx, -dV_dy, color="black", angles="xy", scale_units="xy", scale=10
    )
    format_plot(
        ax=axs[1],
        xlabel="x",
        ylabel="y",
        title="Henonov potencial z gradientnimi vektorji",
    )

    fig.tight_layout()
    if save:
        savefigure("2_Potencial")
    plt.show()


# plot_potential()

# =========================================================
# ANALIZA ORBIT


def draw_orbit(ax, initial_y, initial_v, energy):
    initial_x = 0
    initial_u = np.sqrt(
        2 * (energy - henon_potential(initial_x, initial_y)) - initial_v**2
    )

    # Reši enačbe gibanja
    solution = solve_ivp(
        Henon_odes, (0, 100), [0, initial_y, initial_u, initial_v], dense_output=True
    )
    # Interpolacija
    t_new = np.linspace(solution.t[0], solution.t[-1], 1000)
    x_interp = interp1d(solution.t, solution.y[0], kind="cubic")(t_new)
    y_interp = interp1d(solution.t, solution.y[1], kind="cubic")(t_new)

    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(x, y)

    # Izračunaj vrednosti potenciala
    Z = henon_potential(X, Y)

    # Interpolacija
    f_interp = RectBivariateSpline(x, y, Z)
    x_new = np.linspace(-1.5, 1.5, 500)
    y_new = np.linspace(-1.5, 1.5, 500)
    X_new, Y_new = np.meshgrid(x_new, y_new)
    Z_new = f_interp(x_new, y_new)

    # Nariši orbito
    ax.contour(
        X_new, Y_new, Z_new, 40, colors="black", alpha=0.41
    )  # Povečamo število konturnih nivojev
    ax.contourf(
        X_new, Y_new, Z_new, 40, cmap="viridis"
    )  # Povečamo število konturnih nivojev
    ax.scatter(0, initial_y, color="black")
    ax.plot(x_interp, y_interp)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"y(0) = {initial_y}, E = 0.1, v(0) = {initial_v}")


# Začetni pogoji
initial_y = 0.5
energy = 0.1
initial_v = 0
zac_p = [(0.2, 0), (-0.2, -0.1), (0, 0.3), (0, -0.3), (-0.2, 0.3), (0.3, 0)]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15), sharex=True, sharey=True)
axs = axs.flatten()
for i, (initial_y, initial_v) in tqdm(enumerate(zac_p)):
    draw_orbit(axs[i])

axs[-1].set_xlabel("x")
axs[-2].set_xlabel("x")
axs[-3].set_xlabel("x")
axs[0].set_ylabel("y")
axs[3].set_ylabel("y")
fig.tight_layout()
savefigure("2_orbite")
plt.show()

energy_sez = [0.1, 0.2]
initial_v = 0
initial_y = 0.5

fig, axs = plt.subplots(nrows=1, ncols=2)
for i, e in enumerate(energy_sez):
    draw_orbit(axs[i], initial_y, initial_v, e)

savefigure("2_pobegla_orbita")
plt.show()

# =========================================================
# POINCAREJEV PRESEK


def poincare_section(energy, initial_ys, initial_vs):
    poincare_points = []

    for initial_y in initial_ys:
        for initial_v in initial_vs:
            initial_x = 0

            # Preverimo, ali je vrednost pod korenom pozitivna
            sqrt_argument = (
                2 * (energy - henon_potential(initial_x, initial_y)) - initial_v**2
            )
            if sqrt_argument < 0:
                continue  # Preskočimo neveljaven začetni pogoj

            initial_u = np.sqrt(sqrt_argument)

            def condition(t, z):
                return z[0]  # x = 0

            condition.terminal = False
            condition.direction = 1

            solution = solve_ivp(
                odes,
                (0, 400),
                [initial_x, initial_y, initial_u, initial_v],
                events=condition,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10,
            )
            for event_time in solution.t_events[0]:
                y_val = solution.sol(event_time)[1]
                v_val = solution.sol(event_time)[3]
                poincare_points.append((initial_y, initial_v, y_val, v_val))

    return poincare_points


# Parametri
energy_sez = [0.001, 0.01, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 1 / 6]
# energy_sez = [0.1]
initial_ys = np.linspace(0, 1, 30)
initial_vs = np.linspace(0, 1, 30)


if len(energy_sez) != 1:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, energy in enumerate(energy_sez):
        # Izračun Poincaréjevega preseka
        points = poincare_section(energy, initial_ys, initial_vs)

        colors = plt.cm.viridis(np.linspace(0, 1, len(initial_ys) * len(initial_vs)))
        color_map = {}
        color_index = 0
        for initial_y, initial_v, y, v in tqdm(points):
            key = (initial_y, initial_v)

            axs[i].scatter(y, v, s=5)
            axs[i].set_title(f"E = {energy}")

    axs[3].set_ylabel("y")
    axs[0].set_ylabel("y")
    axs[6].set_ylabel("y")
    axs[-1].set_xlabel("v")
    axs[-2].set_xlabel("v")
    axs[-3].set_xlabel("v")
    savefigure("2_poincare")
    plt.show()

else:
    points = poincare_section(energy_sez[0], initial_ys, initial_vs)

    colors = plt.cm.viridis(np.linspace(0, 1, len(initial_ys) * len(initial_vs)))
    color_map = {}
    color_index = 0
    for initial_y, initial_v, y, v in tqdm(points):
        key = (initial_y, initial_v)
        plt.scatter(y, v, s=5)
    plt.title(f"E = {energy_sez[0]}")
    plt.xlabel("v")
    plt.ylabel("y")
    savefigure("2_poincare_eden")
    plt.show()
