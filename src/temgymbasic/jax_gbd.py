import jax.numpy as jnp


def propagate_qpinv_abcd(Qinv, A, B, C, D):
    num = C + D @ Qinv
    den = A + B @ Qinv
    return num @ jnp.linalg.inv(den)


def diagonal_vec_mat_mul_quad(M, b):
    Mxx = M[:, 0, 0]
    Myy = M[:, 1, 1]
    bx = b[:, 0]
    by = b[:, 1]

    return Mxx[:, jnp.newaxis] * (bx ** 2) + Myy[:, jnp.newaxis] * (by ** 2)


def diagonal_vec_mat_mul_linear(a, M, b):
    Mxx = M[:, 0, 0]
    Myy = M[:, 1, 1]
    ax = a[:, 0]
    ay = a[:, 1]
    bx = b[:, 0]
    by = b[:, 1]

    Mxx_ax = ax * Mxx
    Myy_ay = ay * Myy

    return Mxx_ax[:, jnp.newaxis] * bx + Myy_ay[:, jnp.newaxis] * by


# Rayleigh Range
def R(z, z_r):
    return z * (1 + (z_r / z) ** 2)


# Waist
def w_z(w0, z, zR):
    return w0 * jnp.sqrt(1 + (z / zR) ** 2)


# Q matrix
def calculate_Qinv(z, z_r, w0, wl, num_rays):
    if jnp.abs(z) < 1e-10:
        qinv = -1 / (1j * jnp.pi * w0 ** 2) / (wl)
    else:
        qinv = 1 / R(z, z_r) - 1j * wl / jnp.pi * w_z(w0, z, z_r) ** 2

    try:
        dtype = qinv.dtype
    except AttributeError:
        dtype = complex

    Qinv = jnp.zeros((num_rays, 2, 2), dtype=dtype)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


def calculate_Qpinv(A, B, C, D, Qinv):

    NUM = C + (D @ Qinv)

    DEN = jnp.linalg.inv(A + B @ Qinv)

    return NUM @ DEN


def gaussian_amplitude(Qinv, A, B):
    den = A + B @ Qinv
    return 1 / jnp.sqrt(jnp.linalg.det(den))


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
    out
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
        B_inv = jnp.linalg.inv(B)
    except jnp.linalg.LinAlgError:
        B_inv = jnp.eye(B.shape[0])

    r2m = (A @ r1m[:, :, None]).squeeze(-1) + (B @ theta1m[:, :, None]).squeeze(-1)

    Q1 = jnp.linalg.inv(Q1_inv)

    amplitude = 1 / jnp.sqrt(jnp.linalg.det(A + B @ Q1_inv))[:, jnp.newaxis]
    # constant_phase = jnp.exp(-1j * k * path_length)[:, jnp.newaxis]

    misaligned_phase_r1 = (
        jnp.diag(diagonal_vec_mat_mul_linear(r1m, A @ B_inv, r1m))[:, jnp.newaxis] - 2
        * diagonal_vec_mat_mul_linear(r1m, B_inv, r2)
    )

    propagated_inverse_term = Q1 @ jnp.linalg.inv(B @ (A @ Q1 + B))

    misaligned_phase_r2 = (
        jnp.diag(diagonal_vec_mat_mul_linear(r2m, propagated_inverse_term, r2m))[:, jnp.newaxis] - 2
        * diagonal_vec_mat_mul_linear(r2m, propagated_inverse_term, r2)
    )

    Q2_term = diagonal_vec_mat_mul_quad(Q2_inv, r2)

    out += (amplitude * jnp.exp(1j * k / 2 * (
        Q2_term + misaligned_phase_r1 - misaligned_phase_r2
    ))).sum(axis=0)
