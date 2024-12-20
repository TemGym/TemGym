import numpy as np


def ref_sphere(X, Y, r, xs, ys, zs, xp=np):
    """
    Evaluate the reference sphere.

    Parameters:
    X, Y (float or ndarray): Coordinates at exit pupil.
    r (float): Radius of the sphere.
    xs, ys, zs (float): Coordinates of the sphere center.

    Returns:
    float or ndarray: The z-coordinate on the reference sphere.
    """
    return zs - xp.sqrt(r**2 - (X - xs)**2 - (Y - ys)**2)


def aber(r_aperture, r_object, psi, coeffs, R, M, xp=np):
    """
    Evaluate the path length difference of the aberrated function.

    Parameters:
    r_aperture (float or ndarray): Radial coordinate at the aperture.
    r_object (float or ndarray): Radial coordinate at the object.
    psi (float or ndarray): Angular coordinate at the aperture - equivalent to angle of ray at aperture - angle of ray at object. See Hawkes principles of electron 
    optics vol 1 2017, page 367, eq 24.34. We have made some extra modifications to include the radius of the reference sphere. 
    coeffs (ndarray): [Aberration coefficients.
    
    Returns:
    float or ndarray: Optical path length as given by the aberration function.
    """
    B, F, C, D, E = coeffs
    
    B = B
    F = F
    C = C
    D = D
    E = E

    Spherical = 1 / 4 * B * r_aperture ** 4
    Coma = (F * xp.cos(psi)) * r_aperture ** 3 * r_object
    Astig = 1 / 2 * (C * xp.cos(psi) ** 2) * r_aperture ** 2 * r_object ** 2
    Field = 1 / 2 * D * r_aperture ** 2 * r_object ** 2
    Dist = r_object ** 3 * r_aperture * (E * xp.cos(psi))

    W = Spherical + Coma + Astig + Field + Dist

    return W


def daber_dr_a(r_aperture, r_object, psi, coeffs, R, M, xp=np):
    """
    Evaluate the derivative of aberration function with respect to r_aperture.

    Parameters:
    r_aperture (float or ndarray): Radial coordinate at the aperture.
    r_object (float or ndarray): Radial coordinate at the object.
    psi (float or ndarray): Angular coordinate at the aperture - equivalent to angle of ray at aperture - angle of ray at object. See Hawkes principles of electron 
    optics vol 1 2017, page 367, eq 24.34. We have made some extra modifications to include the radius of the reference sphere. 
    coeffs (ndarray): [Aberration coefficients.
    xp (module): Numerical library, default is numpy.
    
    Returns:
    float or ndarray: Derivative of aberration functon with respect to r_aperture.
    """
    B, F, C, D, E = coeffs
    
    B = B * (M ** 4) / (R ** 4)
    F = F * (M ** 3) / (R ** 3)
    C = C * (M ** 2) / (R ** 2)
    D = D * (M ** 2) / (R ** 2)
    E = E * M / R

    
    dSpherical = B * r_aperture ** 3
    dComa = 3 * F * r_aperture ** 2 * r_object * xp.cos(psi)
    dAstig = C * r_aperture * r_object ** 2 * xp.cos(psi) ** 2
    dField = D * r_aperture * r_object ** 2
    dDist = E * r_object ** 3 * xp.cos(psi)

    dW_dr_a = dSpherical + dComa + dAstig + dField + dDist

    return dW_dr_a


def daber_dpsi(r_aperture, r_object, psi, coeffs, R, M, xp=np):
    """
    Evaluate the derivative of aberration function with respect to r_aperture.

    Parameters:
    r_aperture (float or ndarray): Radial coordinate at the aperture.
    r_object (float or ndarray): Radial coordinate at the object.
    psi (float or ndarray): Angular coordinate at the aperture - equivalent to angle of ray at aperture - angle of ray at object. See Hawkes principles of electron 
    optics vol 1 2017, page 367, eq 24.34. We have made some extra modifications to include the radius of the reference sphere. 
    coeffs (ndarray): [Aberration coefficients.
    xp (module): Numerical library, default is numpy.
    
    Returns:
    float or ndarray: Derivative of aberration functon with respect to psi
    """
    
    B, F, C, D, E = coeffs

    B = B * (M ** 4) / (R ** 4)
    F = F * (M ** 3) / (R ** 3)
    C = C * (M ** 2) / (R ** 2)
    D = D * (M ** 2) / (R ** 2)
    E = E * M / R

    dComa = - F * r_aperture ** 3 * r_object * xp.sin(psi)
    dAstig = - C * r_aperture ** 2 * r_object ** 2 * xp.cos(psi) * xp.sin(psi)
    dDist = - E * r_aperture * r_object ** 3 * xp.sin(psi)

    dW_dpsi = dComa + dAstig + dDist

    return dW_dpsi


def grad_r_a(X, Y, xp=np):
    """
    Calculate the gradient of r_a with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    xp (module): Numerical library, default is numpy.

    Returns:
    tuple: Gradients (dr_a_dx, dr_a_dy).
    """
    mask = (xp.abs(X) < 1e-12) & (xp.abs(Y) < 1e-12)
    dr_a_dx = xp.where(mask, 0, X / xp.sqrt(X**2 + Y**2))
    dr_a_dy = xp.where(mask, 0, Y / xp.sqrt(X**2 + Y**2))

    return dr_a_dx, dr_a_dy


def grad_psi(X, Y, xp=np):
    """
    Calculate the gradient of psi with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    xp (module): Numerical library, default is numpy.

    Returns:
    tuple: Gradients (dpsi_dx, dpsi_dy).
    """
    mask = (xp.abs(X) < 1e-12) & (xp.abs(Y) < 1e-12)
    dpsi_dx = xp.where(mask, 0, -Y / (X ** 2 + Y ** 2))
    dpsi_dy = xp.where(mask, 0, 1 / (X ** 2 + Y ** 2))

    return dpsi_dx, dpsi_dy


def opd(x_a, y_a, x_o, y_o, psi, coeffs, R, M, xp=np):
    """
    Evaluate the optical path difference (OPD).

    Parameters:
    r_aperture (float or ndarray): Radial coordinate at the aperture.
    r_object (float or ndarray): Radial coordinate at the object.
    psi (float or ndarray): Angular coordinate at the aperture - equivalent to angle of ray at aperture - angle of ray at object.
    coeffs (ndarray): Aberration coefficients.
    xp (module): Numerical library, default is numpy.

    Returns:
    float or ndarray: The optical path difference (OPD).
    """
    r_a = xp.sqrt(x_a**2 + y_a**2)
    r_o = xp.sqrt(x_o**2 + y_o**2)

    return aber(r_a, r_o, psi, coeffs, R, M, xp=xp)


def aber_x_aber_y(x_a, y_a, x_o, y_o, coeffs, R, M, xp=np):
    B, F, C, D, E = coeffs

    dx = -R*(B*x_a*(x_a**2 + y_a**2)  # Spherical 
             + 2*F*x_a*(x_a*x_o + y_a*y_o) + F*x_o*(x_a**2 + y_a**2)  # Coma
             + C*x_o*(x_a*x_o + y_a*y_o)  # Astigmatism
             + D*x_a*(x_o**2 + y_o**2)  # Field Curvature
             + E*x_o*(x_o**2 + y_o**2))  # Distortion

    dy = -R*(B*y_a*(x_a**2 + y_a**2)  # Spherical
             + 2*F*y_a*(x_a*x_o + y_a*y_o) + F*y_o*(x_a**2 + y_a**2)  # Coma
             + C*y_o*(x_a*x_o + y_a*y_o)  # Astigmatism
             + D*y_a*(x_o**2 + y_o**2)  # Field Curvature
             + E*y_o*(x_o**2 + y_o**2))  # Distortion

    return dx, dy


def dopd_dx(x_a, y_a, x_o, y_o, psi, coeffs, R, M, xp=np):
    """
    Evaluate the derivative of the optical path difference (OPD) with respect to x.

    Parameters:
    x_a, y_a (float or ndarray): Coordinates at the aperture.
    x_o, y_o (float or ndarray): Coordinates at the object.
    psi (float or ndarray): Angular coordinate at the aperture.
    coeffs (ndarray): Aberration coefficients.
    xp (module): Numerical library, default is numpy.

    Returns:
    float or ndarray: The derivative of the OPD with respect to x.
    """

    r_a = xp.sqrt(x_a**2 + y_a**2)
    r_o = xp.sqrt(x_o**2 + y_o**2)

    dW_dr_a = daber_dr_a(r_a, r_o, psi, coeffs, R, M, xp=xp)
    dW_dpsi = daber_dpsi(r_a, r_o, psi, coeffs, R, M, xp=xp)
    dr_a_dx, _ = grad_r_a(x_a, y_a, xp=xp)
    dpsi_dx, _ = grad_psi(x_a, y_a, xp=xp)
    dW_dx = (dW_dr_a * dr_a_dx + dW_dpsi * dpsi_dx)

    return dW_dx


def dopd_dy(x_a, y_a, x_o, y_o, psi, coeffs, R, M, xp=np):
    """
    Evaluate the derivative of the optical path difference (OPD) with respect to y.

    Parameters:
    x_a, y_a (float or ndarray): Coordinates at the aperture.
    x_o, y_o (float or ndarray): Coordinates at the object.
    psi (float or ndarray): Angular coordinate at the aperture.
    coeffs (ndarray): Aberration coefficients.
    xp (module): Numerical library, default is numpy.

    Returns:
    float or ndarray: The derivative of the OPD with respect to y.
    """

    r_a = xp.sqrt(x_a**2 + y_a**2)
    r_o = xp.sqrt(x_o**2 + y_o**2)

    dW_dr_a = daber_dr_a(r_a, r_o, psi, coeffs, R, M, xp=xp)
    dW_dpsi = daber_dpsi(r_a, r_o, psi, coeffs, R, M, xp=xp)
    _, dr_ady = grad_r_a(x_a, y_a, xp=xp)
    _, dpsi_dy = grad_psi(x_a, y_a, xp=xp)
    dW_dy = (dW_dr_a * dr_ady + dW_dpsi * dpsi_dy)
    
    return dW_dy


def compute_Tpinv(X, Y, IMAGE_POINT, REF_SPHERE_RADIUS, xp=np):
    n = len(X)

    DX = X - IMAGE_POINT[0]
    DY = Y - IMAGE_POINT[1]
    r = REF_SPHERE_RADIUS

    sf = xp.sqrt(r**2 - DY**2)
    Rmzs = xp.sqrt(-DX**2 - DY**2 + r**2)
    Somdy = xp.sqrt(1 - DY**2 / r**2)

    Tpinv = xp.zeros((n, 4, 4))

    Tpinv[:, 0, 0] = Rmzs / sf
    Tpinv[:, 0, 1] = DX * DY / (r * sf)
    Tpinv[:, 0, 2] = -DX * Somdy / sf
    Tpinv[:, 0, 3] = X

    Tpinv[:, 1, 0] = 0
    Tpinv[:, 1, 1] = -Somdy
    Tpinv[:, 1, 2] = -DY / r
    Tpinv[:, 1, 3] = Y

    Tpinv[:, 2, 0] = DX / sf
    Tpinv[:, 2, 1] = -DY * Rmzs / (r * sf)
    Tpinv[:, 2, 2] = Somdy * Rmzs / sf
    Tpinv[:, 2, 3] = r

    Tpinv[:, 3, 0] = 0
    Tpinv[:, 3, 1] = 0
    Tpinv[:, 3, 2] = 0
    Tpinv[:, 3, 3] = 1

    # matrix to transform the local gradient into the global one
    Tbar = xp.zeros((n, 2, 2))
    Tbar[:, 0, 0] = Tpinv[:, 0, 0]
    Tbar[:, 1, 0] = Tpinv[:, 0, 1]
    Tbar[:, 0, 1] = Tpinv[:, 1, 0]
    Tbar[:, 1, 1] = Tpinv[:, 1, 1]

    return Tpinv, Tbar


def transform_to_global(Tbar, gradW, W, Tpinv, xp=np):
    # Initialize arrays for nhat0, phihat0, n, and phi

    nhat0 = xp.zeros_like(Tpinv[..., 0])
    phihat0 = xp.zeros_like(Tpinv[..., 0])
    n = xp.zeros_like(Tpinv[..., 0])
    phi = xp.zeros_like(Tpinv[..., 0])

    # Step 1: Local results from global variables
    # gradWhat0 = xp.dot(Tbar, gradW)
    gradWhat0 = xp.einsum('ijk,ik->ij', Tbar, gradW)

    nhat0[:, 0:2] = -gradWhat0
    nhat0[:, 2] = xp.sqrt(1 - xp.sum(gradWhat0**2, axis=1))
    nhat0[:, 3] = 0

    phihat0 = W[:, xp.newaxis] * nhat0 + xp.array([0, 0, 0, 1])

    # Step 2: Transform to global coordinates
    # Aberrated ray direction
    n = xp.einsum('ijk,ik->ij', Tpinv, nhat0)

    # Wavefront point in global coordinates
    phi = xp.einsum('ijk,ik->ij', Tpinv, phihat0)

    return n, phi


def aberrated_sphere(x, y, xs, ys, h, R, coeffs, xp=np):

    W = opd(x, y, h, coeffs, xp=xp)
    dWdx = dopd_dx(x, y, h, coeffs, xp=xp)
    dWdy = dopd_dy(x, y, h, coeffs, xp=xp)

    Tpinv, Tbar = compute_Tpinv(x, y, [xs, ys], R, xp=xp)
    aber_ray_dir_cosine, aber_ray_coord = transform_to_global(Tbar, xp.array([dWdx, dWdy]).T, W,
                                                              Tpinv, xp=xp)

    return aber_ray_dir_cosine, aber_ray_coord, W
