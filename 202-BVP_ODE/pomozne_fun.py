import numpy as np

# =========================================================
# PRVA NALOGA


def odes(s, y, beta):
    F, alpha, x, yy = y
    F_min = 1e-8  # Nastavite minimalno vrednost za F
    F = max(F, F_min)
    dFds = np.sin(alpha) - beta * x * np.cos(alpha)
    dalpha_ds = (np.cos(alpha) + beta * x * np.sin(alpha)) / F
    dxds = np.cos(alpha)
    dyds = np.sin(alpha)
    return [dFds, dalpha_ds, dxds, dyds]


def odes_gostota(s, y, beta):
    F, alpha, x, yy = y
    rho = 2 * s  # Gostota vrvi rho = 2 * s
    dFds = rho * (np.sin(alpha) - beta * x * np.cos(alpha))
    dalpha_ds = rho * (np.cos(alpha) + beta * x * np.sin(alpha)) / F
    dxds = np.cos(alpha)
    dyds = np.sin(alpha)
    return [dFds, dalpha_ds, dxds, dyds]


def calc_diff(x_end, y_end, yf):
    return np.sqrt((x_end - 0) ** 2 + (y_end - yf) ** 2)


# =========================================================
# DRUGA NALOGA


def henon_potential(x, y):
    return 0.5 * (x**2 + y**2) + x**2 * y - 1 / 3 * y**3


def henon_gradient(x, y):
    dV_dx = x + 2 * x * y
    dV_dy = y + x**2 - y**2
    return dV_dx, dV_dy


def henon_energy(x, y, u, v):
    return 0.5 * (u**2 + v**2) + henon_potential(x, y)


def Henon_odes(t, z):
    x, y, u, v = z
    dxdt = u
    dydt = v
    dudt = -x - 2 * x * y
    dvdt = -y - x**2 + y**2
    return [dxdt, dydt, dudt, dvdt]
