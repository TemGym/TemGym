from typing import TYPE_CHECKING, Tuple, Sequence
from typing_extensions import TypeAlias

from numpy.typing import NDArray
from numba import njit

from scipy.constants import e, m_e, h

if TYPE_CHECKING:
    from .model import STEMModel
    from .rays import Rays
    from . import Degrees, Radians


from . import get_cupy
import numpy as np

RadiansNP: TypeAlias = np.float64

try:
    from itertools import pairwise
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        # Lifted from Python docs
        # pairwise('ABCDEFG') --> AB BC CD DE EF FG
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


def get_xp(data):
    if isinstance(data, np.ndarray):
        return np
    else:
        cp = get_cupy()
        if cp is None:
            raise ImportError("Cupy is not installed")
        return cp

def get_array_from_device(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data.get())


def P2R(radii: NDArray[np.float64], angles: NDArray[RadiansNP]) -> NDArray[np.complex128]:
    return radii * np.exp(1j*angles)


def R2P(x: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[RadiansNP]]:
    return np.abs(x), np.angle(x)


def as_gl_lines(all_rays: Sequence['Rays'], z_mult: int = 1):
    num_vertices = 0
    for r in all_rays[:-1]:
        num_vertices += r.num
    num_vertices *= 2

    vertices = np.empty(
        (num_vertices, 3),
        dtype=np.float32,
    )
    idx = 0

    def _add_vertices(r1: 'Rays', r0: 'Rays'):
        nonlocal idx, vertices

        num_endpoints = r1.num
        sl = slice(idx, idx + num_endpoints * 2, 2)
        vertices[sl, 0] = r1.x
        vertices[sl, 1] = r1.y
        vertices[sl, 2] = r1.z * z_mult
        sl = slice(1 + idx, 1 + idx + num_endpoints * 2, 2)
        # Relies on the fact that indexing
        # with None creates a new axis, only
        vertices[sl, 0] = r0.x[r1.mask].flat
        vertices[sl, 1] = r0.y[r1.mask].flat
        vertices[sl, 2] = r0.z * z_mult
        idx += num_endpoints * 2
        return idx

    r1 = all_rays[-1]
    for r0 in reversed(all_rays[:-1]):
        _add_vertices(r1, r0)
        if (r1b := r1.blocked_rays()) is not None:
            _add_vertices(r1b, r0)
        r1 = r0

    return vertices


def plot_rays(model: 'STEMModel'):
    import matplotlib.pyplot as plt
    from .components import DoubleDeflector

    # Iterate over components and their ray positions
    num_rays = 3
    yx = (0, 8)
    all_rays = tuple(model.scan_point_iter(num_rays=num_rays, yx=yx))

    fig, ax = plt.subplots()
    xvals = np.stack(tuple(r.x for r in all_rays), axis=0)
    zvals = np.asarray(tuple(r.z for r in all_rays))
    ax.plot(xvals, zvals)

    # Optional: Mark the component positions
    extent = 1.5 * np.abs(xvals).max()
    for component in model.components:
        if isinstance(component, DoubleDeflector):
            ax.hlines(
                component.first.z, -extent, extent, linestyle='--'
            )
            ax.text(-extent, component.first.z, repr(component.first), va='bottom')
            ax.hlines(
                component.second.z, -extent, extent, linestyle='--'
            )
            ax.text(-extent, component.second.z, repr(component.second), va='bottom')
        else:
            ax.hlines(component.z, -extent, extent, label=repr(component))
            ax.text(-extent, component.z, repr(component), va='bottom')

    ax.hlines(
        model.objective.ffp, -extent, extent, linestyle=':'
    )

    ax.axvline(color='black', linestyle=":", alpha=0.3)
    _, scan_pos_x = model.sample.scan_position(yx)
    ax.plot([scan_pos_x], [model.sample.z], 'ko')

    # dx = model.detector.shape[1]
    # detector_pixels = np.arange(- dx // 2., dx // 2.) * model.detector.pixel_size
    # ax.plot(
    #     detector_pixels,
    #     model.detector.z * np.ones_like(detector_pixels),
    #     'ro',
    # )

    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.invert_yaxis()
    ax.set_title(f'Ray paths for {num_rays} rays at position {yx}')
    plt.show()


def _flip_y(xp=np):
    # From libertem.corrections.coordinates v0.11.1
    return xp.array([
        (-1, 0),
        (0, 1)
    ])


def _identity(xp=np):
    # From libertem.corrections.coordinates v0.11.1
    return xp.eye(2)


def _rotate(radians: 'Radians', xp=np):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return xp.array([
        (xp.cos(radians), xp.sin(radians)),
        (-xp.sin(radians), xp.cos(radians))
    ])


def _rotate_deg(degrees: 'Degrees', xp=np):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(xp.pi / 180 * degrees, xp)


def get_pixel_coords(
    rays_x, rays_y, shape, pixel_size, flip_y=False, scan_rotation: 'Degrees' = 0., xp=np
):
    if flip_y:
        transform = _flip_y(xp)
    else:
        transform = _identity(xp)

    # Transformations are applied right to left
    transform = _rotate_deg(xp.array(scan_rotation), xp) @ transform

    y_transformed, x_transformed = (xp.array((rays_y, rays_x)).T @ transform).T

    sy, sx = shape
    pixel_coords_x = (x_transformed / pixel_size) + (sx // 2)
    pixel_coords_y = (y_transformed / pixel_size) + (sy // 2)

    return (pixel_coords_x, pixel_coords_y)


def initial_r(num_rays: int):
    r = np.zeros(
        (5, num_rays),
        dtype=np.float64,
    )  # x, theta_x, y, theta_y, 1

    r[4, :] = 1.
    return r


def initial_r_rayset(num_rays: int):
    r = np.zeros(
        (5, num_rays * 5),
        dtype=np.float64,
    )  # x, theta_x, y, theta_y, 1

    r[4, :] = 1.
    return r

@njit
def multi_cumsum_inplace(values, partitions, start):
    part_idx = 0
    current_part_len = partitions[part_idx]
    part_count = 0
    values[0] = start
    for v_idx in range(1, values.size):
        if current_part_len == part_count:
            part_count = 0
            part_idx += 1
            current_part_len = partitions[part_idx]
            values[v_idx] = start
        else:
            values[v_idx] += values[v_idx - 1]
            part_count += 1


def concentric_rings(
    num_points_approx: int,
    radius: float,
):
    num_rings = max(
        1,
        int(np.floor((-1 + np.sqrt(1 + 4 * num_points_approx / np.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = np.round(
        2 * np.pi * np.arange(1, num_rings + 1)
    ).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = np.round(num_points_kth_ring * points_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = np.linspace(
        0, radius, num_rings + 1, endpoint=True,
    )[1:]
    div_angle = 2 * np.pi / points_per_ring

    params = np.stack((radii, div_angle), axis=0)
    
    #Cupy gave an error here saying that points_per_ring must not be an array
    repeats = points_per_ring.tolist()
    
    all_params = np.repeat(params, repeats, axis=-1)
    multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return (
        all_radii * np.sin(all_angles),
        all_radii * np.cos(all_angles),
    )


def fibonacci_spiral(nb_samples: int,
                    radius: float,
                    alpha=2):
    # From https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    # Fibonacci spiral sampling in a unit circle
    # Alpha parameter determines smoothness of boundary - default of 2 means a smooth boundary
    # 0 for a rough boundary.
    # Returns a tuple of y, x coordinates of the samples

    # nb_samples * np.random.random()

    ga = np.pi * (3.0 - np.sqrt(5.0))

    # Boundary points
    np_boundary = np.round(alpha * np.sqrt(nb_samples))

    ss = np.zeros((nb_samples, 2))
    j = 0
    for i in range(nb_samples):
        if i == 0:
            r = 0
        elif i > nb_samples - (np_boundary + 1):
            r = radius
        else:
            r = radius * np.sqrt((i + 0.5) / (nb_samples - 0.5 * (np_boundary + 1)))
        phi = ga * i
        ss[j, :] = np.array([r * np.sin(phi), r * np.cos(phi)])
        j += 1

    y = ss[:, 0]
    x = ss[:, 1]

    return y, x


def random_coords(num: int, max_r: float, with_radii: bool = False):
    # generate random points uniformly sampled in x/y
    # within a centred circle of radius max_r
    # return (y, x)
    yx = np.random.uniform(
        -max_r, max_r, size=(int(num * 1.28), 2)  # 4 / np.pi
    )
    radii = np.sqrt((yx ** 2).sum(axis=1))
    mask = radii < max_r
    yx = yx[mask, :]
    return (
        yx[:, 0],
        yx[:, 1],
    )


def circular_beam(
    num_rays_approx: int,
    outer_radius: float,
    random: bool = False,
) -> NDArray:
    '''
    Generates a circular parallel initial beam
    in approximately equally-filled concentric rings

    Parameters
    ----------
    num_rays_approx : int
        The approximate number of rays
    outer_radius : float
        Outer radius of the circular beam

    Returns
    -------
    r : ndarray
        Ray position & slope matrix
    '''
    if random:
        y, x = random_coords(num_rays_approx, outer_radius)
    else:
        y, x = concentric_rings(num_rays_approx, outer_radius)
    r = initial_r(y.shape[0])
    r[0, :] = x
    r[2, :] = y
    return r


def fibonacci_beam_gauss_rayset(
    num_rays_approx: int,
    outer_radius: float,
    wo: float,
    wavelength: float,
) -> NDArray:
    '''
    Generates a circular parallel initial beam
    in approximately equally-filled concentric rings

    Parameters
    ----------
    num_rays_approx : int
        The approximate number of rays
    outer_radius : float
        Outer radius of the circular beam

    Returns
    -------
    r : ndarray
        Ray position & slope matrix
    '''
    div = wavelength / (np.pi * wo)
    dPx = wo
    dPy = wo
    dHx = div
    dHy = div

    y, x = fibonacci_spiral(num_rays_approx, outer_radius)

    r = initial_r_rayset(y.shape[0])

    r[0, 0::5] = x
    r[2, 0::5] = y

    r[0, 1::5] = x + dPx
    r[2, 1::5] = y

    r[0, 2::5] = x
    r[2, 2::5] = y + dPy

    r[0, 3::5] = x
    r[1, 3::5] += dHx
    r[2, 3::5] = y

    r[0, 4::5] = x
    r[2, 4::5] = y
    r[3, 4::5] += dHy

    return r


def point_beam(
    num_rays_approx: int,
    semiangle: float,
    random: bool = False,
) -> NDArray:
    '''
    Generates a diverging point source initial beam
    in approximately equally-filled conic shells

    Parameters
    ----------
    num_rays_approx : int
        The approximate number of rays
    semiangle : float
        The maximum semiangle of the beam

    Returns
    -------
    r : ndarray
        Ray position & slope matrix
    '''
    if random:
        y, x = random_coords(num_rays_approx, semiangle, with_radii=True)
    else:
        y, x = concentric_rings(num_rays_approx, semiangle)
    r = initial_r(y.size)
    r[1, :] = y
    r[3, :] = x
    return r


def calculate_wavelength(phi_0: float):
    return h / (2 * abs(e) * m_e * phi_0) ** (1 / 2)


def calculate_phi_0(wavelength: float):
    return h ** 2 / (2 * wavelength ** 2 * abs(e) * m_e)


def convert_slope_to_direction_cosines(dx, dy):
    l_dir_cosine = dx / np.sqrt(1 + dx ** 2 + dy ** 2)
    m_dir_cosine = dy / np.sqrt(1 + dx ** 2 + dy ** 2)
    n_dir_cosine = 1 / np.sqrt(1 + dx ** 2 + dy ** 2)
    return l_dir_cosine, m_dir_cosine, n_dir_cosine


def calculate_direction_cosines(x0, y0, z0, x1, y1, z1, xp = np):
    # Calculate the principal ray vector from ray coordinate on object to centre of lens
    vx = x1 - x0
    vy = y1 - y0
    vz = z1 - z0
    v_mag = np.sqrt(vx**2 + vy**2 + vz**2)

    # And it's direction cosines
    M = vy / v_mag
    L = vx / v_mag
    N = vz / v_mag

    return L, M, N
