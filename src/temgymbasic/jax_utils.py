from jax.numpy import ndarray as NDArray
import jax
import jax.numpy as jnp
from . import Degrees, Radians
# from .jax_ray import Ray, GaussianRay

from scipy.constants import e, m_e, h
from typing import (
    Tuple, TypeAlias
)

RadiansJNP = jnp.float64


def P2R(radii: NDArray,
        angles: NDArray) -> NDArray:
    return radii * jnp.exp(1j*angles)


def R2P(x: NDArray) -> Tuple[NDArray,
                             NDArray]:
    return jnp.abs(x), jnp.angle(x)

# def as_gl_lines(all_rays: Sequence['Ray'], z_mult: int = 1):
#     num_vertices = 0
#     for r in all_rays[:-1]:
#         num_vertices += r.num_display
#     num_vertices *= 2

#     jnp = all_rays[0].jnp
#     vertices = jnp.empty(
#         (num_vertices, 3),
#         dtype=jnp.float32,
#     )
#     idx = 0

#     def _add_vertices(r1:, r0:):
#         nonlocal idx, vertices

#         num_endpoints = r1.num_display
#         sl = slice(idx, idx + num_endpoints * 2, 2)
#         vertices[sl, 0] = r1.x_central
#         vertices[sl, 1] = r1.y_central
#         vertices[sl, 2] = r1.z * z_mult
#         sl = slice(1 + idx, 1 + idx + num_endpoints * 2, 2)
#         # Relies on the fact that indexing
#         # with None creates a new axis, only
#         vertices[sl, 0] = r0.x_central[r1.mask_display].ravel()
#         vertices[sl, 1] = r0.y_central[r1.mask_display].ravel()
#         vertices[sl, 2] = r0.z * z_mult
#         idx += num_endpoints * 2
#         return idx

#     r1 = all_rays[-1]
#     for r0 in reversed(all_rays[:-1]):
#         _add_vertices(r1, r0)
#         if (r1b := r1.blocked_rays()) is not None:
#             _add_vertices(r1b, r0)
#         r1 = r0

#     return get_array_from_device(vertices)


# def plot_rays(model: 'STEMModel'):
#     import matplotlib.pyplot as plt
#     from .components import DoubleDeflector

#     # Iterate over components and their ray positions
#     num_rays = 3
#     yx = (0, 8)
#     all_rays = tuple(model.scan_point_iter(num_rays=num_rays, yx=yx))

#     fig, ax = plt.subplots()
#     xvals = jnp.stack(tuple(r.x for r in all_rays), axis=0)
#     zvals = jnp.asarray(tuple(r.z for r in all_rays))
#     ax.plot(xvals, zvals)

#     # Optional: Mark the component positions
#     extent = 1.5 * jnp.abs(xvals).max()
#     for component in model.components:
#         if isinstance(component, DoubleDeflector):
#             ax.hlines(
#                 component.first.z, -extent, extent, linestyle='--'
#             )
#             ax.text(-extent, component.first.z, repr(component.first), va='bottom')
#             ax.hlines(
#                 component.second.z, -extent, extent, linestyle='--'
#             )
#             ax.text(-extent, component.second.z, repr(component.second), va='bottom')
#         else:
#             ax.hlines(component.z, -extent, extent, label=repr(component))
#             ax.text(-extent, component.z, repr(component), va='bottom')

#     ax.hlines(
#         model.objective.ffp, -extent, extent, linestyle=':'
#     )

#     ax.axvline(color='black', linestyle=":", alpha=0.3)
#     _, scan_pos_x = model.sample.scan_position(yx)
#     ax.plot([scan_pos_x], [model.sample.z], 'ko')

#     # dx = model.detector.shape[1]
#     # detector_pixels = jnp.arange(- dx // 2., dx // 2.) * model.detector.pixel_size
#     # ax.plot(
#     #     detector_pixels,
#     #     model.detector.z * jnp.ones_like(detector_pixels),
#     #     'ro',
#     # )

#     ax.set_xlabel('x position')
#     ax.set_ylabel('z position')
#     ax.invert_yaxis()
#     ax.set_title(f'Ray paths for {num_rays} rays at position {yx}')
#     plt.show()


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([
        (-1, 0),
        (0, 1)
    ])


def _identity():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.eye(2)


def _rotate(radians: 'Radians', jnp=jnp):
    # From libertem.corrections.coordinates v0.11.1
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # y, x instead of x, y
    return jnp.array([
        (jnp.cos(radians), jnp.sin(radians)),
        (-jnp.sin(radians), jnp.cos(radians))
    ])


def _rotate_deg(degrees: 'Degrees', jnp=jnp):
    # From libertem.corrections.coordinates v0.11.1
    return _rotate(jnp.pi / 180 * degrees, jnp)


def get_pixel_coords(
    rays_x, rays_y, shape, pixel_size, flip_y=False, scan_rotation: 'Degrees' = 0.
):
    if flip_y:
        transform = _flip_y()
    else:
        transform = _identity()

    # Transformations are applied right to left
    transform = _rotate_deg(jnp.array(scan_rotation), jnp) @ transform

    y_transformed, x_transformed = (jnp.array((rays_y, rays_x)).T @ transform).T

    sy, sx = shape
    pixel_coords_x = (x_transformed / pixel_size) + (sx // 2)
    pixel_coords_y = (y_transformed / pixel_size) + (sy // 2)

    return (pixel_coords_x, pixel_coords_y)


def initial_matrix(n_rays: int):
    matrix = jnp.zeros(
        (n_rays, 5),
        dtype=jnp.float64,
    )  # x, y, theta_x, theta_y, 1

    matrix = matrix.at[:, 4].set(jnp.ones(n_rays))

    return matrix


@jax.jit
def multi_cumsum_inplace(values, partitions, start):
    def body_fun(i, carry):
        vals, part_idx, part_count = carry
        current_len = partitions[part_idx]

        def reset_part(_):
            # move to the next partition, reset, set start
            new_vals = vals.at[i].set(start)
            return (new_vals, part_idx + 1, 0)

        def continue_part(_):
            # accumulate with previous value
            new_vals = vals.at[i].add(vals[i - 1])
            return (new_vals, part_idx, part_count + 1)

        return jax.lax.cond(part_count == current_len, reset_part, continue_part, None)

    values = values.at[0].set(start)
    values, _, _ = jax.lax.fori_loop(1, values.shape[0], body_fun, (values, 0, 0))
    return values


def concentric_rings(
    num_points_approx: int,
    radius: float,
):
    num_rings = max(
        1,
        int(jnp.floor((-1 + jnp.sqrt(1 + 4 * num_points_approx / jnp.pi)) / 2))
    )

    # Calculate the circumference of each ring
    num_points_kth_ring = jnp.round(
        2 * jnp.pi * jnp.arange(1, num_rings + 1)
    ).astype(int)
    num_rings = num_points_kth_ring.size
    points_per_unit = num_points_approx / num_points_kth_ring.sum()
    points_per_ring = jnp.round(num_points_kth_ring * points_per_unit).astype(int)

    # Make get the radii for the number of circles of rays we need
    radii = jnp.linspace(
        0, radius, num_rings + 1, endpoint=True,
    )[1:]
    div_angle = 2 * jnp.pi / points_per_ring

    params = jnp.stack((radii, div_angle), axis=0)

    # Cupy gave an error here saying that points_per_ring must not be an array
    repeats = points_per_ring

    all_params = jnp.repeat(params, repeats, axis=-1)
    multi_cumsum_inplace(all_params[1, :], points_per_ring, 0.)

    all_radii = all_params[0, :]
    all_angles = all_params[1, :]

    return (
        all_radii * jnp.sin(all_angles),
        all_radii * jnp.cos(all_angles),
    )


def fibonacci_spiral(
    nb_samples: int,
    radius: float,
    alpha=2,
    jnp=jnp,
):
    # From https://github.com/matt77hias/fibpy/blob/master/src/sampling.py
    # Fibonacci spiral sampling in a unit circle
    # Alpha parameter determines smoothness of boundary - default of 2 means a smooth boundary
    # 0 for a rough boundary.
    # Returns a tuple of y, x coordinates of the samples

    ga = jnp.pi * (3.0 - jnp.sqrt(5.0))

    # Boundary points
    jnp_boundary = jnp.round(alpha * jnp.sqrt(nb_samples))

    ii = jnp.arange(nb_samples)
    rr = jnp.where(
        ii > nb_samples - (jnp_boundary + 1),
        radius,
        radius * jnp.sqrt((ii + 0.5) / (nb_samples - 0.5 * (jnp_boundary + 1)))
    )
    rr[0] = 0.
    phi = ii * ga
    y = rr * jnp.sin(phi)
    x = rr * jnp.cos(phi)

    return y, x


def random_coords(num: int, jnp=jnp):
    # generate random points uniformly sampled in x/y
    # within a centred circle of radius 0.5
    # return (y, x)
    key = jax.random.PRNGKey(1)

    yx = jax.random.uniform(key, shape=(int(num * 1.28), 2), minval=-1, maxval=1)  # 1.28 =  4 / np.pi
    radii = jnp.sqrt((yx ** 2).sum(axis=1))
    mask = radii < 1
    yx = yx[mask, :]
    return (
        yx[:, 0],
        yx[:, 1],
    )


def calculate_wavelength(phi_0: float):
    return h / (2 * abs(e) * m_e * phi_0) ** (1 / 2)


def calculate_phi_0(wavelength: float):
    return h ** 2 / (2 * wavelength ** 2 * abs(e) * m_e)


def zero_phase_1D(u, idx_x):
    u_centre = u[idx_x]
    phase_difference = 0 - jnp.angle(u_centre)
    u = u * jnp.exp(1j * phase_difference)
    return u


def zero_phase(u, idx_x, idx_y):
    u_centre = u[idx_x, idx_y]
    phase_difference = 0 - jnp.angle(u_centre)
    u = u * jnp.exp(1j * phase_difference)
    return u


def FresnelPropagator(E0, ps, lambda0, z):
    """
    Parameters:
        E0 : 2D array
            The initial complex field in the x-y source plane.
        ps : float
            Pixel size in the object plane (same units as wavelength).
        lambda0 : float
            Wavelength of the light (in the same units as ps).
        z : float
            Propagation distance (in the same units as ps).

    Returns:
        Ef : 2D array
            The complex field after propagating a distance z.
    """
    n, m = E0.shape

    fx = jnp.fft.fftfreq(n, ps)
    fy = jnp.fft.fftfreq(m, ps)
    Fx, Fy = jnp.meshgrid(fx, fy)

    H = jnp.exp(1j * (
        2 * jnp.pi / lambda0) * z) * jnp.ejnp(
        -1j * jnp.pi * lambda0 * z * (Fx**2 + Fy**2))

    E0fft = jnp.fft.fft2(E0)
    G = H * E0fft
    Ef = jnp.fft.ifft2(G)

    return Ef


def lens_phase_factor(n, ps, lambda0, f):
    """
    Compute the phase factor introduced by an ideal lens.

    Parameters:
        n : int
            Number of pixels (assuming square grid, n x n).
        ps : float
            Pixel size (in the same units as wavelength and focal length).
        lambda0 : float
            Wavelength of the light (in the same units as ps).
        f : float
            focal length of the lens (in the same units as ps).

    Returns:
        phase_factor : 2D array (n x n)
            The phase factor to multiply with the field.
    """
    x = jnp.linspace(-n/2, n/2, n) * ps
    y = jnp.linspace(-n/2, n/2, n) * ps
    X, Y = jnp.meshgrid(x, y)

    phase_factor = jnp.ejnp(-1j * jnp.pi * (X**2 + Y**2) / (lambda0 * f) + 1j * jnp.pi)

    return phase_factor
