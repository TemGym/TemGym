from typing import TYPE_CHECKING, Tuple, Iterable
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model import STEMModel
    from .rays import Rays

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


def P2R(radii: NDArray[np.float_], angles: NDArray[np.float_]) -> NDArray[np.complex_]:
    return radii * np.exp(1j*angles)


def R2P(x: NDArray[np.complex_]) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    return np.abs(x), np.angle(x)


def as_gl_lines(all_rays: Iterable['Rays']):
    if len(set(r.num for r in all_rays)) != 1:
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
    from . import DoubleDeflector

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


def _rotate(radians):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return np.array([
        (np.cos(radians), np.sin(radians)),
        (-np.sin(radians), np.cos(radians))
    ])


def _rotate_deg(degrees):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(np.pi/180*degrees)


def get_pixel_coords(rays_x, rays_y, shape, pixel_size, flip_y=False, scan_rotation=0.):
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
        dtype=np.float64
    )  # x, theta_x, y, theta_y, 1

    r[4, :] = np.ones(num_rays)
    return r


# FIXME resolve code duplication between circular_beam() and point_beam()
def circular_beam(num_rays, outer_radius):
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
    r = initial_r(num_rays)

    # Use the equation from stack overflow about ukrainian graves from 2014
    # to calculate the number of even rings including decimal remainder

    if num_rays < 7:
        num_circles_dec = 1.0  # Round up if the number is between 0 and 1
    else:
        num_circles_dec = (-1+np.sqrt(1+4*(num_rays)/(np.pi)))/2

    # Get the number of integer rings
    num_circles_int = int(np.floor(num_circles_dec))

    # Calculate the number of points per ring with the integer amoung of rings
    num_points_kth_ring = np.round(
        2*np.pi*(np.arange(0, num_circles_int+1))).astype(int)

    # get the remainding amount of rays
    remainder_rays = num_rays - np.sum(num_points_kth_ring)

    # Get the proportion of points in each rung
    proportion = num_points_kth_ring/np.sum(num_points_kth_ring)

    # resolve this proportion to an integer value, and reverse it
    num_rays_to_each_ring = np.ceil(proportion*remainder_rays)[::-1]

    # We need to decide on where to stop adding the remainder of rays to the
    # rest of the rings. We find this point by summing the rays in each ring
    # from outside to inside, and then getting the index where it is greater
    # than or equal to the remainder
    index_to_stop_adding_rays = np.where(
        np.cumsum(num_rays_to_each_ring) >= remainder_rays)[0][0]

    # We then get the total number of rays to add
    rays_to_add = np.cumsum(num_rays_to_each_ring)[
        index_to_stop_adding_rays].astype(np.int32)

    # The number of rays to add isn't always matching the remainder, so we
    # collect them here with this line
    final_sub = rays_to_add - remainder_rays

    # Here we take them away so we get the number of rays we want
    num_rays_to_each_ring[index_to_stop_adding_rays] -= final_sub

    # Then we add all of these rays to the correct ring
    num_points_kth_ring[::-1][:index_to_stop_adding_rays+1] += num_rays_to_each_ring[
        :index_to_stop_adding_rays+1
    ].astype(int)

    # Add one point for the centre, and take one away from the end
    num_points_kth_ring[0] = 1
    num_points_kth_ring[-1] = num_points_kth_ring[-1] - 1

    # Make get the radii for the number of circles of rays we need
    radii = np.linspace(0, outer_radius, num_circles_int+1)

    # fill in the x and y coordinates to our ray array
    idx = 0
    for i in range(len(radii)):
        for j in range(num_points_kth_ring[i]):
            radius = radii[i]
            t = j*(2 * np.pi / num_points_kth_ring[i])
            r[0, idx] = radius*np.cos(t)
            r[2, idx] = radius*np.sin(t)
            idx += 1

    return r, num_points_kth_ring
