import numpy as np

def B_variance_appx(B1s, B1c, rmax):
    # Variance in B value as a function of phi
    var = (1/6) * (B1c*B1c + B1s*B1s) * rmax*rmax*rmax
    return var


def elongation_constraints(X1c, Y1c, X1s, Y1s, constraint = 4):
    # Places a constraint on Elongation
    p = X1c*X1c + Y1c*Y1c + X1s*X1s + Y1s*Y1s
    q = X1s*Y1c - Y1s*X1c
    elongation = p + np.sqrt(p*p - 4*q*q)
    elongation = elongation / (2 * np.abs(q))
    return np.average(elongation), np.maximum(0, (elongation - constraint))


def B1c_hat_even_calc(B1s, B1c, iota, nphi, nfp):
    # Metric for QI
    phi = np.linspace(0, 2*np.pi / nfp, nphi)
    B1c_hat_even = B1c * np.cos(iota) * np.cos(phi) + B1s * np.cos(iota) * np.sin(phi)
    return np.abs(B1c_hat_even)


def B1s_hat_even_calc(B1s, B1c, iota, nphi, nfp):
    # Metric for QI
    phi = np.linspace(0, 2*np.pi / nfp, nphi)
    B1s_hat_even = -B1c * np.sin(iota) * np.cos(phi) + B1s * np.sin(iota) * np.sin(phi)
    return np.abs(B1s_hat_even)


def iota_constraints(iota, max_iota = 0.8, min_iota = 0.4):
    # Places constraints on the maximum and minimum iota values
    return max(0, (np.abs(iota) - max_iota)) + max(0, (min_iota - np.abs(iota)))


def minimum_major_radius(R, constraint = 0.1):
    # Constrain on the major radius
    return np.maximum(0, (constraint - R))