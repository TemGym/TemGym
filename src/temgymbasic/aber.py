import numpy as np

import numpy as np

def ref_sphere(X, Y, r, xs, ys, zs):
    """
    Evaluate the reference sphere.

    Parameters:
    X, Y (float or ndarray): Coordinates.
    r (float): Radius of the sphere.
    xs, ys, zs (float): Coordinates of the sphere center.

    Returns:
    float or ndarray: The z-coordinate on the reference sphere.
    """
    return zs - np.sqrt(r**2 - (X - xs)**2 - (Y - ys)**2)


def aber(r_aperture, theta_aperture, r_object, coeffs):
    C, K, A, F, D = coeffs
    
    Spherical = 1/4 * C * r_aperture ** 4
    Coma = (K * np.cos(theta_aperture)) * r_aperture ** 3 * r_object
    Astig = 1/2 * (A * np.cos(2*theta_aperture)) * r_aperture ** 2 * r_object ** 2
    Field  = 1/2 * F * r_aperture ** 2 * r_object ** 2
    Dist = r_object ** 3 * r_aperture * (D * np.cos(theta_aperture))
    
    W = Spherical + Coma + Astig + Field + Dist
    
    return W

def daber_drho(r_aperture, theta_aperture, r_object, coeffs):
    C, K, A, F, D = coeffs
    
    dSpherical = C * r_aperture ** 3
    dComa = 3 *K * r_aperture ** 2 * r_object * np.cos(theta_aperture)
    dAstig = A * r_aperture * r_object **2 * np.cos(2 * theta_aperture)
    dField  = F * r_aperture * r_object ** 2
    dDist = D * r_object ** 3 * np.cos(theta_aperture)

    dWdrho = dSpherical + dComa + dAstig + dField + dDist
    
    return dWdrho

def daber_dtheta(r_aperture, theta_aperture, r_object, coeffs):
    _, K, A, _, D = coeffs
    
    dComa = - K*r_aperture**3*r_object*np.sin(theta_aperture)
    dAstig = -2*A*r_aperture**2*r_object**2*np.sin(2*theta_aperture)/2 
    dDist = D*r_aperture*r_object**3*np.sin(theta_aperture)
    
    dWdtheta = dComa + dAstig + dDist
    
    return dWdtheta


def grad_Rho(X, Y):
    """
    Calculate the gradient of RHO with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.

    Returns:
    tuple: Gradients (dRhodx, dRhody).
    """
    dRhodx = X / np.sqrt(X**2 + Y**2)
    dRhody = Y / np.sqrt(X**2 + Y**2)
    return dRhodx, dRhody

def grad_Theta(X, Y):
    """
    Calculate the gradient of THETA with respect to X and Y.

    Parameters:
    X, Y (float or ndarray): Coordinates.

    Returns:
    tuple: Gradients (dThetadx, dThetady).
    """
    dThetadx = -Y / (X ** 2 + Y ** 2)
    dThetady = 1 / (X ** 2 + Y ** 2)
    
    return dThetadx, dThetady

def opd(X, Y, h, coeffs):
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
    RHO = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)
    
    return aber(RHO, THETA, h, coeffs)

def dopd_dx(X, Y, h, coeffs):
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
    
    RHO = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)
    
    dWdrho = daber_drho(RHO, THETA, h, coeffs)
    dWdtheta = daber_dtheta(RHO, THETA, h, coeffs)
    dRhodx, _ = grad_Rho(X, Y)
    dThetadx, _ = grad_Theta(X, Y)
    dWdx = (dWdrho * dRhodx + dWdtheta * dThetadx)
    
    return dWdx

def dopd_dy(X, Y, h, coeffs):
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
    
    RHO = np.sqrt(X**2 + Y**2)
    THETA = np.arctan2(Y, X)
    
    dWdrho = daber_drho(RHO, THETA, h, coeffs)
    dWdtheta = daber_dtheta(RHO, THETA, h, coeffs)
    _, dRhody = grad_Rho(X, Y)
    _, dThetady = grad_Theta(X, Y)
    dWdy = (dWdrho * dRhody + dWdtheta * dThetady)
    return dWdy

def compute_Tpinv(X, Y, IMAGE_POINT, REF_SPHERE_RADIUS):
    n = len(X)
    
    DX = X - IMAGE_POINT[0]
    DY = Y - IMAGE_POINT[1]
    r = REF_SPHERE_RADIUS

    sf = np.sqrt(r**2 - DY**2)
    Rmzs = np.sqrt(-DX**2 - DY**2 + r**2)
    Somdy = np.sqrt(1 - DY**2 / r**2)

    Tpinv = np.zeros((n, 4, 4))

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
    
    #matrix to transform the local gradient into the global one
    Tbar = np.zeros((n, 2, 2))
    Tbar[:, 0, 0] = Tpinv[:, 0, 0]
    Tbar[:, 1, 0] = Tpinv[:, 0, 1]
    Tbar[:, 0, 1] = Tpinv[:, 1, 0]
    Tbar[:, 1, 1] = Tpinv[:, 1, 1]

    return Tpinv, Tbar

def transform_to_global(Tbar, gradW, W, Tpinv):
    # Initialize arrays for nhat0, phihat0, n, and phi
    
    nhat0 = np.zeros_like(Tpinv[..., 0])
    phihat0 = np.zeros_like(Tpinv[..., 0])
    n = np.zeros_like(Tpinv[..., 0])
    phi = np.zeros_like(Tpinv[..., 0])

    # Step 1: Local results from global variables
    #gradWhat0 = np.dot(Tbar, gradW)
    gradWhat0 = np.einsum('ijk,ik->ij', Tbar, gradW)
    
    nhat0[:, 0:2] = -gradWhat0
    nhat0[:, 2] = np.sqrt(1 - np.sum(gradWhat0**2, axis = 1))
    nhat0[:, 3] = 0

    phihat0 = W[:, np.newaxis] * nhat0 + np.array([0, 0, 0, 1])

    # Step 2: Transform to global coordinates
    # Aberrated ray direction
    n = np.einsum('ijk,ik->ij', Tpinv, nhat0)

    # Wavefront point in global coordinates
    phi = np.einsum('ijk,ik->ij', Tpinv, phihat0)
    
    return n, phi


def aberrated_sphere(x, y, xs, ys, h, R, coeffs):

    W = opd(x, y, h, coeffs)
    dWdx = dopd_dx(x, y, h, coeffs)
    dWdy = dopd_dy(x, y, h, coeffs)
    
    Tpinv, Tbar = compute_Tpinv(x, y, [xs, ys], R)
    aber_ray_dir_cosine, aber_ray_coord = transform_to_global(Tbar, np.array([dWdx, dWdy]).T, W, Tpinv)
    
    return aber_ray_dir_cosine, aber_ray_coord, W