from .jax_utils import concentric_rings, initial_matrix, random_coords, calculate_wavelength
from .jax_ray import Ray
import jax.numpy as jnp
import jax


def P2R(radii, angles):
    return radii * jnp.exp(1j*angles)


def R2P(x):
    return jnp.abs(x), jnp.angle(x)


def ParallelBeam(z,
                 radius,
                 n_rays,
                 tilt_yx=(0., 0.),
                 centre_yx=(0., 0.),
                 voltage=None,
                 random=False):

    if random:
        y, x = random_coords(n_rays) * radius
    else:
        y, x = concentric_rings(n_rays, radius)

    input_matrix = initial_matrix(n_rays)

    input_matrix = input_matrix.at[:, 0].set(x + centre_yx[1])
    input_matrix = input_matrix.at[:, 1].set(y + centre_yx[0])
    input_matrix = input_matrix.at[:, 2].set(0. + tilt_yx[1])
    input_matrix = input_matrix.at[:, 3].set(0. + tilt_yx[0])

    input_z = jnp.ones((n_rays)) * z
    input_amplitude = jnp.ones((n_rays))
    input_pathlength = jnp.zeros((n_rays))
    input_wavelength = calculate_wavelength(voltage)

    return Ray(z=input_z,
               matrix=input_matrix,
               amplitude=input_amplitude,
               pathlength=input_pathlength,
               wavelength=input_wavelength)


def PointBeam(z,
              n_rays,
              semi_angle=0.,
              tilt_yx=(0., 0.),
              centre_yx=(0., 0.),
              voltage=None,
              random=False):

    if random:
        dy, dx = random_coords(n_rays) * semi_angle
    else:
        dy, dx = concentric_rings(n_rays, semi_angle)

    input_matrix = initial_matrix(n_rays)

    input_matrix = input_matrix.at[:, 0].set(centre_yx[1])
    input_matrix = input_matrix.at[:, 1].set(centre_yx[0])

    input_matrix = input_matrix.at[:, 2].set(tilt_yx[1] + dx)
    input_matrix = input_matrix.at[:, 3].set(tilt_yx[0] + dy)

    input_z = jnp.ones((n_rays,)) * z
    input_amplitude = jnp.ones((n_rays,))
    input_pathlength = jnp.zeros((n_rays,))
    input_wavelength = calculate_wavelength(voltage)
    return Ray(z=input_z,
               matrix=input_matrix,
               amplitude=input_amplitude,
               pathlength=input_pathlength,
               wavelength=input_wavelength)


def XAxialBeam(z,
               radius,
               n_rays,
               tilt_yx=(0., 0.),
               centre_yx=(0., 0.),
               voltage=None,
               random=False):

    if random:

        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (n_rays,), minval=-radius, maxval=radius)
    else:

        x = jnp.linspace(-radius, radius, n_rays)
    y = jnp.zeros_like(x)

    # Fill matrix
    input_matrix = initial_matrix(n_rays)
    input_matrix = input_matrix.at[:, 0].set(x + centre_yx[1])
    input_matrix = input_matrix.at[:, 1].set(y + centre_yx[0])
    input_matrix = input_matrix.at[:, 2].set(tilt_yx[1])
    input_matrix = input_matrix.at[:, 3].set(tilt_yx[0])

    # Create ray
    input_z = jnp.ones((n_rays,)) * z
    input_amplitude = jnp.ones((n_rays,))
    input_pathlength = jnp.zeros((n_rays,))
    input_wavelength = calculate_wavelength(voltage)
    return Ray(z=input_z,
               matrix=input_matrix,
               amplitude=input_amplitude,
               pathlength=input_pathlength,
               wavelength=input_wavelength)


def RadialSpikesBeam(z,
                     radius,
                     n_rays,
                     tilt_yx=(0., 0.),
                     centre_yx=(0., 0.),
                     voltage=None):

    # Build radial spikes
    n_each = max(n_rays // 4, 1)
    xvals = jnp.linspace(0., radius, num=n_each, endpoint=True)
    yvals = jnp.zeros_like(xvals)
    origin_c = xvals + 1j * yvals

    # Convert to polar
    orad, oang = R2P(origin_c)
    radius1 = P2R(orad * 0.75, oang + jnp.pi * 0.4)
    radius2 = P2R(orad * 0.5, oang + jnp.pi * 0.8)
    radius3 = P2R(orad * 0.25, oang + jnp.pi * 1.2)
    r_c = jnp.concatenate([origin_c, radius1, radius2, radius3])
    x = jnp.real(r_c)
    y = jnp.imag(r_c)

    # Fill matrix
    input_matrix = initial_matrix(x.size)
    input_matrix = input_matrix.at[:, 0].set(x + centre_yx[1])
    input_matrix = input_matrix.at[:, 1].set(y + centre_yx[0])
    input_matrix = input_matrix.at[:, 2].set(tilt_yx[1])
    input_matrix = input_matrix.at[:, 3].set(tilt_yx[0])

    # Create ray
    input_z = jnp.ones((x.size,)) * z
    input_amplitude = jnp.ones((x.size,))
    input_pathlength = jnp.zeros((x.size,))
    input_wavelength = calculate_wavelength(voltage)
    return Ray(z=input_z,
               matrix=input_matrix,
               amplitude=input_amplitude,
               pathlength=input_pathlength,
               wavelength=input_wavelength)


def XPointBeam(z,
               n_rays,
               semi_angle=0.,
               tilt_yx=(0., 0.),
               centre_yx=(0., 0.),
               voltage=None,
               random=False):

    # Tilt in one axis only
    if random:
        key = jax.random.PRNGKey(2)
        tilts = jax.random.uniform(key, (n_rays,), minval=-semi_angle, maxval=semi_angle)
    else:
        tilts = jnp.linspace(-semi_angle, semi_angle, n_rays)

    # Positions at origin
    x = jnp.zeros_like(tilts)
    y = jnp.zeros_like(tilts)

    # Fill matrix
    input_matrix = initial_matrix(n_rays)
    input_matrix = input_matrix.at[:, 0].set(x + centre_yx[1])
    input_matrix = input_matrix.at[:, 1].set(y + centre_yx[0])
    input_matrix = input_matrix.at[:, 2].set(tilt_yx[1] + tilts)
    input_matrix = input_matrix.at[:, 3].set(tilt_yx[0])

    # Create ray
    input_z = jnp.ones((n_rays,)) * z
    input_amplitude = jnp.ones((n_rays,))
    input_pathlength = jnp.zeros((n_rays,))
    input_wavelength = calculate_wavelength(voltage)
    return Ray(z=input_z,
               matrix=input_matrix,
               amplitude=input_amplitude,
               pathlength=input_pathlength,
               wavelength=input_wavelength)
