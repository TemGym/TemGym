import numpy as np


def ref_sphere(X, Y, r, xs, ys, zs, xp=np):
    """
    Evaluate the reference sphere.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    r (float): Radius of the sphere.
    xs, ys, zs (float): Coordinates of the sphere center.

    Returns:
    float or ndarray: The z-coordinate on the reference sphere.
    """
    return zs - xp.sqrt(r**2 - (X - xs)**2 - (Y - ys)**2)


def aber(r_aperture, theta_aperture, r_object, coeffs, xp=np):
    C, K, A, F, D = coeffs

    Spherical = 1/4 * C * r_aperture ** 4
    Coma = (K * xp.cos(theta_aperture)) * r_aperture ** 3 * r_object
    Astig = 1/2 * (A * xp.cos(2*theta_aperture)) * r_aperture ** 2 * r_object ** 2
    Field = 1/2 * F * r_aperture ** 2 * r_object ** 2
    Dist = r_object ** 3 * r_aperture * (D * xp.cos(theta_aperture))

    W = Spherical + Coma + Astig + Field + Dist

    return W


def daber_drho(r_aperture, theta_aperture, r_object, coeffs, xp=np):
    C, K, A, F, D = coeffs

    dSpherical = C * r_aperture ** 3
    dComa = 3 * K * r_aperture ** 2 * r_object * xp.cos(theta_aperture)
    dAstig = A * r_aperture * r_object ** 2 * xp.cos(2 * theta_aperture)
    dField = F * r_aperture * r_object ** 2
    dDist = D * r_object ** 3 * xp.cos(theta_aperture)

    dWdrho = dSpherical + dComa + dAstig + dField + dDist

    return dWdrho


def daber_dtheta(r_aperture, theta_aperture, r_object, coeffs, xp=np):
    _, K, A, _, D = coeffs

    dComa = - K*r_aperture**3*r_object*xp.sin(theta_aperture)
    dAstig = -2*A*r_aperture**2*r_object**2*xp.sin(2*theta_aperture)/2
    dDist = D*r_aperture*r_object**3*xp.sin(theta_aperture)

    dWdtheta = dComa + dAstig + dDist

    return dWdtheta


def grad_Rho(X, Y, xp=np):
    """
    Calculate the gradient of RHO with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.

    Returns:
    tuple: Gradients (dRhodx, dRhody).
    """
    mask = (xp.abs(X) < 1e-12) & (xp.abs(Y) < 1e-12)
    dRhodx = xp.where(mask, 0, X / xp.sqrt(X**2 + Y**2))
    dRhody = xp.where(mask, 0, Y / xp.sqrt(X**2 + Y**2))

    return dRhodx, dRhody


def grad_Theta(X, Y, xp=np):
    """
    Calculate the gradient of THETA with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.

    Returns:
    tuple: Gradients (dThetadx, dThetady).
    """
    mask = (xp.abs(X) < 1e-12) & (xp.abs(Y) < 1e-12)
    dThetadx = xp.where(mask, 0, -Y / (X ** 2 + Y ** 2))
    dThetady = xp.where(mask, 0, 1 / (X ** 2 + Y ** 2))

    return dThetadx, dThetady


def opd(X, Y, h, coeffs, xp=np):
    """
    Evaluate the optical path difference (OPD).

    Parameters:
    X, Y (float or ndarray): Coordinates.
    IMAGE_POINT (ndarray): Image point coordinates.
    IMAGE_PLANE (float): Image plane distance.
    PUPIL_RADIUS (float): Pupil radius.
    FIELD_SIZE (float): Field size.
    coeffs (ndarray): Seidel coefficients.

    Returns:
    float or ndarray: The optical path difference (OPD).
    """
    RHO = xp.sqrt(X**2 + Y**2)
    THETA = xp.arctan2(Y, X)

    return aber(RHO, THETA, h, coeffs, xp=xp)


def dopd_dx(X, Y, h, coeffs, xp=np):
    """
    Evaluate the derivative of the optical path difference (OPD) with respect to x.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    IMAGE_POINT (ndarray): Image point coordinates.
    IMAGE_PLANE (float): Image plane distance.
    PUPIL_RADIUS (float): Pupil radius.
    FIELD_SIZE (float): Field size.
    coeffs (ndarray): Aberration coefficients.

    Returns:
    float or ndarray: The derivative of the OPD with respect to x.
    """

    RHO = xp.sqrt(X**2 + Y**2)
    THETA = xp.arctan2(Y, X)

    dWdrho = daber_drho(RHO, THETA, h, coeffs, xp=xp)
    dWdtheta = daber_dtheta(RHO, THETA, h, coeffs, xp=xp)
    dRhodx, _ = grad_Rho(X, Y, xp=xp)
    dThetadx, _ = grad_Theta(X, Y, xp=xp)
    dWdx = (dWdrho * dRhodx + dWdtheta * dThetadx)

    return dWdx


def dopd_dy(X, Y, h, coeffs, xp=np):
    """
    Evaluate the derivative of the optical path difference (OPD) with respect to y.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    IMAGE_POINT (ndarray): Image point coordinates.
    IMAGE_PLANE (float): Image plane distance.
    PUPIL_RADIUS (float): Pupil radius.
    FIELD_SIZE (float): Field size.
    coeffs (ndarray): Seidel coefficients.

    Returns:
    float or ndarray: The derivative of the OPD with respect to y.
    """

    RHO = xp.sqrt(X**2 + Y**2)
    THETA = xp.arctan2(Y, X)

    dWdrho = daber_drho(RHO, THETA, h, coeffs, xp=xp)
    dWdtheta = daber_dtheta(RHO, THETA, h, coeffs, xp=xp)
    _, dRhody = grad_Rho(X, Y, xp=xp)
    _, dThetady = grad_Theta(X, Y, xp=xp)
    dWdy = (dWdrho * dRhody + dWdtheta * dThetady)
    return dWdy


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
