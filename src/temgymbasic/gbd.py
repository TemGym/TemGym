import numexpr as ne
import line_profiler
from .backend import xp


@line_profiler.profile
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
    ABCD = xp.array([
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
    return num @ xp.linalg.inv(den)


def Matmulvec(r0, Mat, r1):
    out = (r0[..., 0]*Mat[..., 0, 0] + r0[..., 1]*Mat[..., 1, 0])*r1[..., 0]
    out = (out + (r0[..., 0]*Mat[..., 0, 1] + r0[..., 1]*Mat[..., 1, 1])*r1[..., 1])
    return out


def misalign_phase(B, A, r1m, r2, k):
    """
    Parameters
    ----------
    B,A : numpy.ndarrays
        elements of the ray transfer matrix
    r1m : numpy.ndarray of size 2
        misalignment in position in x then y
    r2 : numpy.ndarray of dimension 2
        detector coordinates in x and y. First dimension holds x/y, second holds coordinate
    k : float, complex
        wave number of simulation
    """
    Binv = xp.linalg.inv(B)
    BinvA = Binv @ A

    misalign = (r1m[..., 0]*BinvA[..., 0, 0] + r1m[..., 1]*BinvA[..., 1, 0])*r1m[..., 0]
    misalign = (misalign + (r1m[..., 0]*BinvA[..., 0, 1]
                            + r1m[..., 1]*BinvA[..., 1, 1])*r1m[..., 1])

    cross = (r1m[..., 0]*Binv[..., 0, 0] + r1m[..., 1]*Binv[..., 1, 0])*r2[..., 0]
    cross = -2*(cross + (r1m[..., 0]*Binv[..., 0, 1] + r1m[..., 1]*Binv[..., 1, 1])*r2[..., 1])

    return xp.exp(-1j * k / 2 * (misalign + cross))


@line_profiler.profile
def transversal_phase(Qpinv, r, k):
    """compute the transverse gaussian phase of a gaussian beam

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix
    r : numpy.ndarray
        N x 2 radial coordinate vector

    Returns
    -------
    numpy.ndarray
        phase of the gaussian profile
    """
    # r: (n_px, n_gauss, 2:[x ,y])
    # Qpinv: (n_gauss, 2, 2)
    # transversal_ref = (
    #     (
    #         r[..., 0]
    #         * Qpinv[..., 0, 0]
    #     )
    #     + (
    #         r[..., 1]
    #         * Qpinv[..., 1, 0]
    #     )
    # ) * r[..., 0]
    # transversal_ref += (
    #     (
    #         r[..., 0]
    #         * Qpinv[..., 0, 1]
    #     )
    #     + (
    #         r[..., 1]
    #         * Qpinv[..., 1, 1]
    #     )
    # ) * r[..., 1]
    # transversal_ref /= 2

    # The intermediate array here is quite large
    # of shape (n_px, n_gauss, 2, 2), can be improved
    # using some form of one-step sumproduct
    # transversal = (
    #     r[..., xp.newaxis]
    #     * Qpinv[xp.newaxis, ...] / 2
    # ).sum(axis=-1)
    # transversal *= r
    # transversal = transversal.sum(axis=-1)

    # transversal = (r * Qpinv[xp.newaxis, ..., 0] / 2)
    # transversal += (r * Qpinv[xp.newaxis, ..., 1] / 2)
    # transversal *= r
    # transversal = transversal.sum(axis=-1)

    transversal = (r[..., 0] ** 2 * Qpinv[xp.newaxis, ..., 0, 0] / 2)
    transversal += (r[..., 1] ** 2 * Qpinv[xp.newaxis, ..., 1, 1] / 2)
    return transversal
    # return xp.exp(1j * k * transversal)


def phase_correction(r1m, p1m, r2m, p2m, k):
    # See https://www.tandfonline.com/doi/abs/10.1080/09500340600842237
    z1_phase = xp.sum(r1m * p1m, axis=1)
    z2_phase = xp.sum(r2m * p2m, axis=1)
    return xp.exp(-1j * k / 2 * (-z2_phase + z1_phase))


@line_profiler.profile
def gaussian_amplitude(Qinv, A, B):
    den = A + B @ Qinv
    return 1 / xp.sqrt(xp.linalg.det(den))


@line_profiler.profile
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
    guoy = (xp.arctan(xp.real(e1) / xp.imag(e1)) + xp.arctan(xp.real(e2) / xp.imag(e2))) / 2

    return guoy


@line_profiler.profile
def misalign_phase_plane_wave(r2, p2m, k):
    # r2: (n_px, n_gauss, 2:[x ,y])
    # p2m: (n_gauss, 2:[x ,y])
    # l0 = r2 * p2m
    phi = r2[:, :, 0] * p2m[:, 0] * (1 + ((p2m[:, 0] ** 2) / 2))
    # phi_y = l0_y * (1 + ((p2m[:, 1] ** 2) / 2))
    # phi = phi_x + phi_y
    phi += r2[:, :, 1] * p2m[:, 1] * (1 + ((p2m[:, 1] ** 2) / 2))
    return phi
    return (
        r2
        * p2m[xp.newaxis, ...]
        * (
            1
            + (p2m[xp.newaxis, ...] ** 2) / 2
        )
    ).sum(axis=-1)
    # return xp.exp(1j * k * phi)


@line_profiler.profile
def propagate_misaligned_gaussian(
    Qinv,
    Qpinv,
    r,
    p2m,
    k,
    A,
    B,
    path_length,
    out,
):
    # Qinv : (n_gauss, 2, 2), complex
    # Qpinv : (n_gauss, 2, 2), complex
    # r: (n_px, n_gauss, 2:[x ,y]), float => det coords relative to final, central pos
    # p2m: (n_gauss, 2:[x, y]), float => slopes of arriving central ray
    # k: scalar float
    # A: (n_gauss, 2, 2), float
    # B: (n_gauss, 2, 2), float
    # path_length: (n_gauss,), float => path length of central ray

    misaligned_phase = misalign_phase_plane_wave(r, p2m, k)
    # (n_px, n_gauss): complex
    aligned = transversal_phase(Qpinv, r, k)  # Phase and Amplitude at transversal plane to beam dir
    # (n_px, n_gauss): complex
    # opl = xp.exp(1j * k * path_length)  # Optical path length phase
    # (n_gauss,): complex
    guoy = guoy_phase(Qpinv)  # Guoy phase
    # (n_gauss,): complex
    amplitude = gaussian_amplitude(Qinv, A, B)  # Complex Gaussian amplitude
    # (n_gauss,): complex
    aligned *= 1j
    aligned += 1j * misaligned_phase
    aligned += 1j * path_length[xp.newaxis, :]
    aligned *= k
    aligned -= 1j * guoy[xp.newaxis, :]
    xp.exp(aligned, out=aligned)
    # xp.exp(aligned, out=aligned)
    aligned *= xp.abs(amplitude)
    # It should be possible to avoid this intermediate .sum
    # if we could reduce directly into out, but I can't find
    # a way to express that with numpy. Numba could be an option
    out += aligned.sum(axis=-1)
    # return aligned.sum(axis=-1)
    # (n_px,): complex
    # return field


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
    e1 = mean_ondiag + xp.sqrt(mean_ondiag ** 2 - determinant)
    e2 = mean_ondiag - xp.sqrt(mean_ondiag ** 2 - determinant)

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

    matinv = xp.array([[d, -b], [-c, a]]) / det
    if matinv.ndim > 2:
        for i in range(matinv.ndim - 2):
            matinv = xp.moveaxis(matinv, -1, 0)

    return matinv


def calculate_Qinv(z_r, num_rays):

    qinv = 1/(-1j*z_r)
    Qinv = xp.zeros((num_rays, 2, 2), dtype=xp.complex128)

    # Fill the diagonal elements
    Qinv[:, 0, 0] = qinv
    Qinv[:, 1, 1] = qinv

    return Qinv


def calculate_Qpinv(A, B, C, D, Qinv):

    NUM = (C + D @ Qinv)

    DEN = mat_inv_2x2(A + B @ Qinv)

    return NUM @ DEN
