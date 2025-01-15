import numpy as np


def differential_matrix(rayset, dPx, dPy, dHx, dHy, xp=np):

    x_cen_T = rayset[0, 0, :]
    x_px_T = rayset[1, 0, :]
    x_py_T = rayset[2, 0, :]
    x_pl_T = rayset[3, 0, :]
    x_pm_T = rayset[4, 0, :]

    l_cen_T = rayset[0, 1, :]
    l_px_T = rayset[1, 1, :]
    l_py_T = rayset[2, 1, :]
    l_pl_T = rayset[3, 1, :]
    l_pm_T = rayset[4, 1, :]

    y_cen_T = rayset[0, 2, :]
    y_px_T = rayset[1, 2, :]
    y_py_T = rayset[2, 2, :]
    y_pl_T = rayset[3, 2, :]
    y_pm_T = rayset[4, 2, :]

    m_cen_T = rayset[0, 3, :]
    m_px_T = rayset[1, 3, :]
    m_py_T = rayset[2, 3, :]
    m_pl_T = rayset[3, 3, :]
    m_pm_T = rayset[4, 3, :]

    # Computing each element of the matrix
    ABCD = xp.array([
        [(x_px_T - x_cen_T) / dPx, (x_py_T - x_cen_T) / dPy,
         (x_pl_T - x_cen_T) / dHx, (x_pm_T - x_cen_T) / dHy],
        [(y_px_T - y_cen_T) / dPx, (y_py_T - y_cen_T) / dPy,
         (y_pl_T - y_cen_T) / dHx, (y_pm_T - y_cen_T) / dHy],
        [(l_px_T - l_cen_T) / dPx, (l_py_T - l_cen_T) / dPy,
         (l_pl_T - l_cen_T) / dHx, (l_pm_T - l_cen_T) / dHy],
        [(m_px_T - m_cen_T) / dPx, (m_py_T - m_cen_T) / dPy,
         (m_pl_T - m_cen_T) / dHx, (m_pm_T - m_cen_T) / dHy]
    ], dtype=rayset.dtype)

    ABCD = ABCD.transpose(2, 0, 1)
    A = ABCD[:, 0:2, 0:2]
    B = ABCD[:, 0:2, 2:4]
    C = ABCD[:, 2:4, 0:2]
    D = ABCD[:, 2:4, 2:4]

    return A, B, C, D


def propagate_qpinv_abcd(Qinv, A, B, C, D, xp=np):
    num = C + D @ Qinv
    den = A + B @ Qinv
    return num @ xp.linalg.inv(den)


def diagonal_vec_mat_mul_quad(M, b, xp=np):
    Mxx = M[:, 0, 0]
    Myy = M[:, 1, 1]
    bx = b[:, 0]
    by = b[:, 1]

    return Mxx[:, np.newaxis] * (bx ** 2) + Myy[:, np.newaxis] * (by ** 2)


def diagonal_vec_mat_mul_linear(a, M, b, xp=np):
    Mxx = M[:, 0, 0]
    Myy = M[:, 1, 1]
    ax = a[:, 0]
    ay = a[:, 1]
    bx = b[:, 0]
    by = b[:, 1]

    Mxx_ax = ax * Mxx
    Myy_ay = ay * Myy

    return Mxx_ax[:, np.newaxis] * bx + Myy_ay[:, np.newaxis] * by


def calculate_Qinv(z_r, num_rays, xp=np):

    qinv = 1 / (-1j * z_r)
    try:
        dtype = qinv.dtype
    except AttributeError:
        dtype = complex
    Qinv = xp.zeros((num_rays, 2, 2), dtype=dtype)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


# Rayleigh Range
def R(z, z_r):
    return z * (1 + (z_r / z) ** 2)


# Waist
def w_z(w0, z, zR):
    return w0 * np.sqrt(1 + (z / zR) ** 2)


# Q matrix
def calculate_Qinv(z, z_r, w0, wl, num_rays, xp=np):
    if np.abs(z) < 1e-10:
        qinv = -1 / (1j * (np.pi * w0 ** 2) / (wl))
    else:
        qinv = 1 / R(z, z_r) - 1j * wl / (np.pi * w_z(w0, z, z_r) ** 2)

    try:
        dtype = qinv.dtype
    except AttributeError:
        dtype = complex

    Qinv = xp.zeros((num_rays, 2, 2), dtype=dtype)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


def calculate_Qpinv(A, B, C, D, Qinv, xp=np):

    NUM = C + (D @ Qinv)

    DEN = xp.linalg.inv(A + B @ Qinv)

    return NUM @ DEN

def gaussian_amplitude(Qinv, A, B, xp=np):
    den = A + B @ Qinv
    return 1 / xp.sqrt(xp.linalg.det(den))

def propagate_misaligned_gaussian(
    Q1_inv,
    Q2_inv,
    r2,
    r1m,
    theta1m,
    k,
    A,
    B,
    path_length,
    out,
    xp=np
):
    # Q1_inv : (n_gauss, 2, 2), complex
    # r2: (n_px, 2:[x ,y]), float => det coords relative to final, central pos
    # theta2m: (n_gauss, 2:[x, y]), float => slopes of arriving central ray
    # k: scalar float
    # A: (n_gauss, 2, 2), float
    # B: (n_gauss, 2, 2), float
    # path_length: (n_gauss,), float => path length of central ray
    # out: (n_px,), complex => output array

    # Need to find a better fix for this: I don't think B becomes the identity matrix if
    # it is singular, I think the integral is multiplied by a dirac delta function
    # and that gives us a new solution for the integral which needs to be implemented.
    try:
        B_inv = xp.linalg.inv(B)
    except xp.linalg.LinAlgError:
        B_inv = xp.eye(B.shape[0])

    r2m = (A @ r1m[:, :, None]).squeeze(-1) + (B @ theta1m[:, :, None]).squeeze(-1)

    Q1 = xp.linalg.inv(Q1_inv)

    amplitude = 1 / xp.sqrt(xp.linalg.det(A + B @ Q1_inv))[:, np.newaxis]
    constant_phase = xp.exp(-1j * k * path_length)[:, np.newaxis]

    misaligned_phase_r1 = (
        np.diag(diagonal_vec_mat_mul_linear(r1m, A @ B_inv, r1m))[:, np.newaxis] - 2
        * diagonal_vec_mat_mul_linear(r1m, B_inv, r2)
    )

    propagated_inverse_term = Q1 @ xp.linalg.inv(B @ (A @ Q1 + B))

    misaligned_phase_r2 = (
        np.diag(diagonal_vec_mat_mul_linear(r2m, propagated_inverse_term, r2m))[:, np.newaxis] - 2
        * diagonal_vec_mat_mul_linear(r2m, propagated_inverse_term, r2)
    )

    Q2_term = diagonal_vec_mat_mul_quad(Q2_inv, r2)

    out += (amplitude * xp.exp(1j * k / 2 * (
        Q2_term + misaligned_phase_r1 - misaligned_phase_r2
    ))).sum(axis=0)
