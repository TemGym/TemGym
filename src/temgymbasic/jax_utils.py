from jax.numpy import ndarray as NDArray
import jax
import jax.numpy as jnp
import numpy as np
from . import Degrees, Radians
from jax.flatten_util import ravel_pytree

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

    
def get_pytree_idx_from_model(model, parameters):

    indices = {}

    # Obtain the paths and leaves of the model
    paths, leaves =  jax.tree.flatten_with_path(model)

    # Initialise the flat index to find parameters
    flat_idx = 0

    # loop through the paths and leaves of the model to find the 
    # parameters we want to differentiate with respect to. 
    for path, leaf in zip(paths):
        component_idx = path[0]
        component_params = path[1]
        target_parameters = parameters.get(component_idx)

        if target_parameters is not None:
            # Check that target parameters exists, if not, skip
            # and also make sure it is a list
            target_parameters = target_parameters if isinstance(target_parameters, list) else [target_parameters]
            if component_params in target_parameters:
                flat_leaf, _ = ravel_pytree(leaf)

                # Add the indices to a dictionary of parameter names
                indices[component_params] = (flat_idx, flat_idx + flat_leaf.size)

    # Return the indices, the flat model and the treedef function
    return indices, flat_model, treedef


def _flip_y():
    # From libertem.corrections.coordinates v0.11.1
    return jnp.array([
        (-1., 0.),
        (0., 1.)
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
    rays_x, rays_y, shape, pixel_size, flip_y=1, scan_rotation: 'Degrees' = 0.
):
    transform = jax.lax.cond(flip_y,
                             lambda _: _flip_y(),
                             lambda _: _identity(),
                             operand=None)

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


def run_model_for_rays_and_slopes(transfer_matrices, input_slopes, scan_position):
    #Given input model and it's transfer matrix, run the model to find the positions of the rays at the sample and detector
    scan_pos_y, scan_pos_x = scan_position

    input_slopes_x = input_slopes[0]
    input_slopes_y = input_slopes[1]

    # Make the input rays we can run through one last time in the model to find positions at sample and detector
    rays_at_source_with_semi_conv = np.vstack([
        np.full(len(input_slopes_x), scan_pos_x),
        np.full(len(input_slopes_y), scan_pos_y),
        input_slopes_x,
        input_slopes_y,
        np.ones_like(input_slopes_x)
    ])

    # Propagate the point source coordinates through the forward ABCD matrices
    coord_list = [rays_at_source_with_semi_conv]
    for ABCD in transfer_matrices:
        new_coord = np.dot(ABCD[0], coord_list[-1])
        coord_list.append(new_coord)
        
    # Stack the propagated coordinates into an array for easier indexing
    coords_array = np.stack(coord_list, axis=0)
    
    xs = coords_array[:, 0, :]
    ys = coords_array[:, 1, :]
    dxs = coords_array[:, 2, :]
    dys = coords_array[:, 3, :]

    coords = np.array([xs, ys, dxs, dys])
    
    return coords


def find_input_slopes_that_hit_detpx_from_pt_source_with_semiconv(pixel_coords, scan_pos, semi_conv, transformation_matrix):
    """
    Given a set of detector pixel coordinates, a semi-convergence angle from a source, and a transformation matrix, find a mask that tells us 
    what slopes will hit the detector pixels from the point source.

    Args:
    - pixel_coords (jnp.array): A (N, 2) array of pixel coordinates in y, x format.
    - scan_pos (jnp.array): A (2,) array of the scan position in x and y format.
    - semi_conv (float): The semi-convergence angle of the point source.q
    - transformation_matrix (jnp.array): A (5, 5) transformation matrix.
    """
    # We rely on the fact that theta_x_in**2 + theta_y_in**2 = semi_conv**2 - this is essentially a parametric equation of an ellipse.  
    # which can be evaluated to find all pixels less than any semi_conv from the point source 
    # For a point source, our system of equations is:
    # [x_out, y_out, theta_x_out, theta_y_out, 1] = transformation_matrix @ [scan_pos_x, scan_pos_y, theta_x_in, theta_y_in, 1]
    # where theta_x_in and theta_y_in are given by alpha * cos(phi) and alpha * sin(phi), with alpha being the semi convergence angle
    # and phi the azimuthal angle of a ray from the point source. This is a parametric equation for a cone of rays from the point source
    # giving us the means to find a mask, which will tell us which slopes will hit the detector pixel from the point source for a given semi_conv.
    scan_pos_y, scan_pos_x = scan_pos

    A_xx, A_xy, B_xx, B_xy = transformation_matrix[0, :4] # Select first row not including the last column of values from the 5x5 transfer matrix
    A_yx, A_yy, B_yx, B_yy = transformation_matrix[1, :4] # Select second row not including the last column of values from the 5x5 transfer matrix

    delta_x, delta_y = transformation_matrix[0, 4], transformation_matrix[1, 4] # Get the shift values from the final column of the transfer matrix

    y_out, x_out = pixel_coords[:, 0], pixel_coords[:, 1]

    denom = (B_xx*B_yy - B_xy*B_yx)
    theta_x_in = (-A_xx*B_yy*scan_pos_x - A_xy*B_yy*scan_pos_y + A_yx*B_xy*scan_pos_x + 
         A_yy*B_xy*scan_pos_y + B_xy*delta_y - B_xy*y_out - B_yy*delta_x + B_yy*x_out) / denom
    
    theta_y_in = (A_xx*B_yx*scan_pos_x + A_xy*B_yx*scan_pos_y - A_yx*B_xx*scan_pos_x - 
         A_yy*B_xx*scan_pos_y - B_xx*delta_y + B_xx*y_out + B_yx*delta_x - B_yx*x_out) / denom
    
    F = (theta_x_in**2 + theta_y_in**2) - semi_conv **2
    mask = F <= 0
    input_slopes_masked = np.stack([theta_x_in, theta_y_in]) * mask
    
    return input_slopes_masked