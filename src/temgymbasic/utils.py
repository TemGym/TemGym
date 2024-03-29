from typing import TYPE_CHECKING, Tuple, Iterable, Optional
from typing_extensions import TypeAlias, Literal
import numpy as np
from numpy.typing import NDArray
from numba import njit

from scipy.constants import e, m_e, h

if TYPE_CHECKING:
    from .model import STEMModel
    from .rays import Rays
    from . import Degrees, Radians

RadiansNP: TypeAlias = np.float_

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


def P2R(radii: NDArray[np.float_], angles: NDArray[RadiansNP]) -> NDArray[np.complex_]:
    return radii * np.exp(1j*angles)


def R2P(x: NDArray[np.complex_]) -> Tuple[NDArray[np.float_], NDArray[RadiansNP]]:
    return np.abs(x), np.angle(x)


def as_gl_lines(all_rays: Iterable['Rays']):
    if len({r.num for r in all_rays}) != 1:
        # need to sort
        raise NotImplementedError
    num_rays = all_rays[0].num
    zpos = tuple((r0.z, r1.z) for r0, r1 in pairwise(all_rays))
    zpos = np.asarray(zpos).ravel()
    zpos = np.tile(zpos, num_rays)
    vertices = np.zeros(
        (zpos.size, 3),
        dtype=np.float32,
    )
    xvals = np.stack(tuple(
        r.x for r in all_rays
    ), axis=-1)
    xvals = np.lib.stride_tricks.sliding_window_view(
        xvals,
        2,
        axis=1,
    )
    yvals = np.stack(tuple(
        r.y for r in all_rays
    ), axis=-1)
    yvals = np.lib.stride_tricks.sliding_window_view(
        yvals,
        2,
        axis=1,
    )
    vertices[:, 0] = xvals.ravel()
    vertices[:, 1] = yvals.ravel()
    vertices[:, 2] = zpos
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


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return np.array([
        (-1, 0),
        (0, 1)
    ])


def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return np.eye(2)


def _rotate(radians: 'Radians'):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return np.array([
        (np.cos(radians), np.sin(radians)),
        (-np.sin(radians), np.cos(radians))
    ])


def _rotate_deg(degrees: 'Degrees'):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(np.pi / 180 * degrees)


def get_pixel_coords(
    rays_x, rays_y, shape, pixel_size, flip_y=False, scan_rotation: 'Degrees' = 0.,
):
    if flip_y:
        transform = _flip_y()
    else:
        transform = _identity()

    # Transformations are applied right to left
    transform = _rotate_deg(scan_rotation) @ transform

    y_transformed, x_transformed = (np.array((rays_y, rays_x)).T @ transform).T

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


def make_beam(
    num_rays_approx: int,
    beam_type: Literal['circular_beam', 'point_beam'] = 'circular_beam',
    *,
    outer_radius: Optional[float] = None,
    semiangle: Optional[float] = None,
    randomize: bool = False,
):
    '''Generates a circular paralell initial beam

    Parameters
    ----------
    r : ndarray
        Ray position and slope matrix
    outer_radius : float
        Outer radius of the circular beam

    Returns
    -------
    r : ndarray
        Updated ray position & slope matrix which create a circular beam
    num_points_kth_ring: ndarray
        Array of the number of points on each ring of our circular beam
    '''

    if beam_type == 'point_beam':
        assert semiangle is not None
        assert outer_radius is None
        outer_radius = semiangle
    elif beam_type == 'circular_beam':
        assert semiangle is None
        assert outer_radius is not None
    else:
        raise ValueError(f'Unrecognized beam type: {beam_type}')

    num_rings = max(
        1,
        int(np.floor((-1 + np.sqrt(1 + 4 * num_rays_approx / np.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = np.round(
        2 * np.pi * np.arange(1, num_rings + 1)
    ).astype(int)
    num_rings = num_points_kth_ring.size
    rays_per_unit = num_rays_approx / num_points_kth_ring.sum()
    rays_per_ring = np.round(num_points_kth_ring * rays_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = np.linspace(
        0, outer_radius, num_rings + 1, endpoint=True,
    )[1:]
    div_angle = 2 * np.pi / rays_per_ring

    params = np.stack((radii, div_angle), axis=0)
    all_params = np.repeat(params, rays_per_ring, axis=-1)
    multi_cumsum_inplace(all_params[1, :], rays_per_ring, 0.)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    r = initial_r(all_angles.size + 1)
    if beam_type == 'circular_beam':
        np.cos(all_angles, out=r[0, 1:])
        np.sin(all_angles, out=r[2, 1:])
        r[0, 1:] *= all_radii
        r[2, 1:] *= all_radii
    elif beam_type == 'point_beam':
        np.tan(all_radii, out=all_radii)
        np.cos(all_angles, out=r[1, 1:])
        np.sin(all_angles, out=r[3, 1:])
        r[1, 1:] *= all_radii
        r[3, 1:] *= all_radii

    return r


def calculate_wavelength(phi_0):
    return h / (2 * abs(e) * m_e * phi_0) ** (1 / 2)


def calculate_phi_0(wavelength):
    return h**2 / (2*wavelength**2 * abs(e) * m_e)
