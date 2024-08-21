import numpy as np


def differential_matrix(rayset, dPx, dPy, dHx, dHy):

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
    ABCD = np.array([
        [(x_px_T - x_cen_T) / dPx, (x_py_T - x_cen_T) / dPy,
         (x_pl_T - x_cen_T) / dHx, (x_pm_T - x_cen_T) / dHy],
        [(y_px_T - y_cen_T) / dPx, (y_py_T - y_cen_T) / dPy,
         (y_pl_T - y_cen_T) / dHx, (y_pm_T - y_cen_T) / dHy],
        [(l_px_T - l_cen_T) / dPx, (l_py_T - l_cen_T) / dPy,
         (l_pl_T - l_cen_T) / dHx, (l_pm_T - l_cen_T) / dHy],
        [(m_px_T - m_cen_T) / dPx, (m_py_T - m_cen_T) / dPy,
         (m_pl_T - m_cen_T) / dHx, (m_pm_T - m_cen_T) / dHy]
    ])

    ABCD = ABCD.transpose(2, 0, 1)
    A = ABCD[:, 0:2, 0:2]
    B = ABCD[:, 0:2, 2:4]
    C = ABCD[:, 2:4, 0:2]
    D = ABCD[:, 2:4, 2:4]

    return A, B, C, D


def propagate_qpinv_abcd(Qinv, A, B, C, D):
    num = C + D @ Qinv
    den = A + B @ Qinv
    return num @ np.linalg.inv(den)


def misalign_phase(B, A, r1m, r2, k):
    """Compute the extra phase factor from ray data for misaligned evaluation"""

    Binv = mat_inv_2x2(B)
    BA = np.einsum('...ij,...jk->...ik', Binv, A)

    misalign = np.einsum('...i,...ij,...j', r1m, BA, r1m)
    cross = -2 * np.einsum('...i,...ij,...j', r1m, Binv, r2)

    return np.exp(-1j * k / 2 * (misalign + cross))


def transversal_phase(Qpinv, r, k):
    """compute the transverse gaussian phase of a gaussian beam

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix
    r : numpy.ndarray
        N x 2 radial coordinate vector
    k : float
        wave number
    Returns
    -------
    numpy.ndarray
        phase of the gaussian profile in the plane perpendicular to the ray propagation axis
    """

    transversal = np.einsum('...i,...ij,...j->...', r, Qpinv, r)

    return np.exp(1j * k / 2 * transversal)


def phase_correction(r1m, p1m, k):
    # See https://www.tandfonline.com/doi/abs/10.1080/09500340600842237
    return np.exp(-1j * k / 2 * (r1m.T @ p1m))


def gaussian_amplitude(Qinv, A, B):
    den = A + B @ Qinv
    return 1 / np.sqrt(np.linalg.det(den))


def guoy_phase(Qpinv):
    """compute the guoy phase of a complex curvature matrix

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix

    Returns
    -------
    numpy.ndarray
        guoy phase of the complex curvature matrix
    """

    e1, e2 = eigenvalues_2x2(Qpinv)
    guoy = (np.arctan(np.real(e1) / np.imag(e1)) + np.arctan(np.real(e2) / np.imag(e2))) / 2
    return np.exp(-1j * guoy)


def propagate_misaligned_gaussian(Qinv, Qpinv, r, r1m, p1m, r2, k, A, B, path_length):

    misalign = misalign_phase(B, A, r1m, r2, k)  # First misalignment factor
    misalign_corr = phase_correction(r1m, p1m, k)  # Second misalignment factor
    aligned = transversal_phase(Qpinv, r, k)  # Phase and Amplitude at transversal plane to beam dir
    opl = np.exp(1j * k * path_length)  # Optical path length phase
    guoy = guoy_phase(Qpinv)  # Guoy phase
    amplitude = gaussian_amplitude(Qinv, A, B)  # Complex Gaussian amplitude
    return np.abs(amplitude) * aligned * opl * misalign * misalign_corr * guoy


def eigenvalues_2x2(array):
    """ Computes the eigenvalues of a 2x2 matrix using a trick

    Parameters
    ----------
    array : numpy.ndarray
        a N x 2 x 2 array that we are computing the eigenvalues of
    Returns
    -------
    e1, e2 : floats of shape N
        The eigenvalues of the array
    """

    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    determinant = a * d - b * c
    mean_ondiag = (a + d) / 2
    e1 = mean_ondiag + np.sqrt(mean_ondiag ** 2 - determinant)
    e2 = mean_ondiag - np.sqrt(mean_ondiag ** 2 - determinant)

    return e1, e2


def mat_inv_2x2(array):
    """compute inverse of 2x2 matrix, broadcasted

    Parameters
    ----------
    array : numpy.ndarray
        array containing 2x2 matrices in last dimension. Returns inverse array of shape array.shape

    Returns
    -------
    matinv
        matrix inverse array
    """
    a = array[..., 0, 0]
    b = array[..., 0, 1]
    c = array[..., 1, 0]
    d = array[..., 1, 1]

    det = a * d - b * c

    matinv = np.array([[d, -b], [-c, a]]) / det
    if matinv.ndim > 2:
        for i in range(matinv.ndim - 2):
            matinv = np.moveaxis(matinv, -1, 0)

    return matinv


def calculate_Qinv(z_r):
    N = len(z_r)
    qinv = 1/(-1j*z_r)
    Qinv = np.zeros((N, 2, 2), dtype=qinv.dtype)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


def calculate_Qpinv(A, B, C, D, Qinv):

    NUM = (C + D @ Qinv)

    DEN = mat_inv_2x2(A + B @ Qinv)

    return NUM @ DEN
