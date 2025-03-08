# Standard imports
import inspect
import os
import pickle
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

from Helpers.io_functions import get_output_root

# ==================================================================================

G = (1.0,)
M = (1.0,)
no_zero_div = 1e-5


# OSNOVNE ENAČBE/FUNKCIJE
def odes(t, sez, epsilon=0):
    x, y, u, v = sez
    x_dot = u
    y_dot = v
    u_dot = -(1 + epsilon) * x / (x**2 + y**2) ** ((3 + epsilon) / 2)
    v_dot = -(1 + epsilon) * y / (x**2 + y**2) ** ((3 + epsilon) / 2)
    return np.array([x_dot, y_dot, u_dot, v_dot])


def odes_a(sez, epsilon=0):
    x, y = sez
    u_dot = -(1 + epsilon) * x / (x**2 + y**2) ** ((3 + epsilon) / 2)
    v_dot = -(1 + epsilon) * y / (x**2 + y**2) ** ((3 + epsilon) / 2)
    return np.array([u_dot, v_dot])


def vrtilna_kol(sez):
    x, y, u, v = sez
    return x * v - y * u


def energija(sez):
    x, y, u, v = sez
    r = np.sqrt(x**2 + y**2)
    T = 0.5 * (u**2 + v**2)
    U = -1 / r if r != 0 else float("-inf")  # Preprečimo deljenje z 0
    return T + U


def runge_lenz(sez):
    L = vrtilna_kol(sez)
    x, y, u, v = sez
    r = np.array([x, y])
    v_vec = np.array([u, v])
    A = v_vec * L - r / np.linalg.norm(r)
    return np.sqrt(A[0] ** 2 + A[1] ** 2)


def gravitational_force(x, y, x_sun, y_sun):
    dx = x - x_sun
    dy = y - y_sun
    r_squared = dx**2 + dy**2 + no_zero_div**2
    r = np.sqrt(r_squared)
    fx = -G * M * dx / r**3
    fy = -G * M * dy / r**3
    return fx, fy


def energija_3Body(state, t):
    x, y, vx, vy = state
    r1 = np.sqrt(x**2 + y**2 + no_zero_div**2)
    r2 = np.sqrt((x - (-10 + 2 * t)) ** 2 + (y - 1.5) ** 2 + no_zero_div**2)
    T = 0.5 * (vx**2 + vy**2)
    U = -G * M / r1 - G * M / r2
    return T + U


# =============================================================
# SOLVING
def verlet(f, x0, v0, t, *args):
    n = len(t)
    x = np.zeros((n, len(x0)))
    v = np.zeros((n, len(v0)))

    x[0] = x0
    v[0] = v0

    sig = inspect.signature(f)
    num_params = len(sig.parameters)

    if num_params > 1:  # If f expects more than one argument
        force = lambda x: f(x, *args)
    else:
        force = lambda x: f(x)

    # Osnovni algoritem
    for i in range(n - 1):
        h = t[i + 1] - t[i]

        v_half = v[i] + (h / 2) * force(x[i])
        x[i + 1] = x[i] + h * v_half
        v[i + 1] = v_half + (h / 2) * force(x[i + 1])

    return np.column_stack((x, v))


def solve_ode(initial, dt, tk=10, method="Verlet", epsilon=0):
    t_span = np.arange(0, tk, dt)

    if method in ["RK45", "Radau"]:
        sol = solve_ivp(
            partial(odes, epsilon=epsilon),  # Popravljen klic
            t_span=[t_span.min(), t_span.max()],
            t_eval=t_span,
            method=method,
            y0=initial,
            dense_output=True,
        )
        return sol.y.T

    elif method == "Verlet":
        return verlet(odes_a, initial[:2], initial[2:], t_span, epsilon)

    else:
        print("Method should be: 'RK45', 'Radau' or 'Verlet'")
        return None


# =============================================================
# IO FUNCTIONS
def save_results(vez_sez, eng_sez, states, filename):
    output_root = get_output_root()
    data_dir = os.path.join(output_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, filename)

    with open(save_path, "wb") as f:
        pickle.dump({"vez_sez": vez_sez, "eng_sez": eng_sez, "states": states}, f)
    print(f"Results saved at: {save_path}")


def load_results(filename):
    output_root = get_output_root()
    data_dir = os.path.join(output_root, "Data")
    os.makedirs(data_dir, exist_ok=True)
    load_path = os.path.join(data_dir, filename)

    if os.path.exists(load_path):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            return data["vez_sez"], data["eng_sez"], data["states"]
    else:
        print(f"File {load_path} not found.")
        return None, None, None
