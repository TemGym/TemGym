import pytest
import jax
import jax.numpy as jnp


import temgymbasic.jax_components as comp
import temgymbasic.jax_source as source
from temgymbasic.jax_run import run_model_to_end
from temgymbasic.jax_ray import Ray, propagate
from temgymbasic.jax_utils import (
    calculate_phi_0,
    calculate_wavelength,
)

from numpy.testing import assert_equal, assert_allclose

import scipy
from scipy.constants import e, m_e, c

try:
    import cupy as cp
except ImportError:
    cp = None

USE_GPU = False

ETA = (abs(e) / (2 * m_e)) ** (1 / 2)
EPSILON = abs(e) / (2 * m_e * c**2)

key = jax.random.PRNGKey(0)


@pytest.fixture()
def empty_rays():
    return Ray(
        matrix=jnp.empty(shape=(0, 5)),
        z=0.2,
        wavelength=jnp.empty([]),
        amplitude=jnp.empty([]),
        pathlength=jnp.empty([]),
    )


def single_ray(x, y, dx, dy, phi_0=1.0):
    matrix = jnp.zeros(shape=(1, 5))

    matrix = matrix.at[:, 0].set(x)
    matrix = matrix.at[:, 1].set(y)
    matrix = matrix.at[:, 2].set(dx)
    matrix = matrix.at[:, 3].set(dy)
    matrix = matrix.at[:, 4].set(jnp.ones(1))

    return Ray(
        matrix=matrix,
        z=0.2,
        pathlength=0.0,
        wavelength=calculate_wavelength(phi_0=phi_0),
    )


def random_rays(n_rays):
    matrix = jnp.zeros(shape=(n_rays, 5))
    matrix = matrix.at[:, 0].set(jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=n_rays))
    matrix = matrix.at[:, 1].set(jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=n_rays))
    matrix = matrix.at[:, 2].set(jax.random.uniform(key, minval=-0.2, maxval=0.2, shape=n_rays))
    matrix = matrix.at[:, 3].set(jax.random.uniform(key, minval=-0.2, maxval=0.2, shape=n_rays))
    matrix = matrix.at[:, 4].set(jnp.ones(shape=(n_rays)))

    return Ray(
        matrix=matrix,
        z=jnp.ones(n_rays) * 0.2,
        pathlength=jnp.zeros((n_rays,)),
    )


@pytest.fixture()
def parallel_rays():
    n_rays = 128
    matrix = jnp.zeros(shape=(n_rays, 5))

    matrix = matrix.at[:, 0].set(jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=n_rays))
    matrix = matrix.at[:, 1].set(jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=n_rays))
    matrix = matrix.at[:, 2].set(jnp.zeros(shape=(n_rays)))
    matrix = matrix.at[:, 3].set(jnp.zeros(shape=(n_rays)))
    matrix = matrix.at[:, 4].set(jnp.ones(shape=(n_rays)))

    return Ray(
        matrix=matrix,
        z=jnp.ones(n_rays) * 0.2,
        pathlength=jnp.zeros((n_rays,)),
        wavelength=jnp.ones(shape=n_rays) * calculate_wavelength(1.0),
    )


@pytest.fixture()
def point_rays():
    n_rays = 128
    matrix = jnp.zeros(shape=(n_rays, 5))

    radius = 0.5
    low = -radius / 2
    high = radius / 2

    matrix = matrix.at[:, 0].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 1].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 2].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 3].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 4].set(jnp.ones(shape=n_rays))

    return Ray(
        matrix=matrix,
        z=0.0,
        pathlength=jnp.zeros((n_rays,)),
    )


@pytest.fixture()
def paraxial_parallel_rays():
    n_rays = 128

    matrix = jnp.zeros(shape=(n_rays, 5))
    radius = 1e-3

    low = -radius / 2
    high = radius / 2

    matrix = matrix.at[:, 0].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 1].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 2].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 3].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 4].set(jnp.ones(shape=n_rays))

    return Ray(
        matrix=matrix,
        z=0.2,
        pathlength=jnp.zeros((n_rays,)),
        wavelength=jnp.ones(shape=n_rays) * calculate_wavelength(1.0),
    )


@pytest.fixture()
def paraxial_point_rays():
    n_rays = 128

    matrix = jnp.zeros(shape=(n_rays, 5))

    radius = 3e-2
    low = -radius / 2
    high = radius / 2

    matrix = matrix.at[:, 0].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 1].set(jnp.zeros(shape=n_rays))
    matrix = matrix.at[:, 2].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 3].set(jax.random.uniform(key, minval=low, maxval=high, shape=n_rays))
    matrix = matrix.at[:, 4].set(jnp.ones(shape=n_rays))

    return Ray(
        matrix=matrix,
        z=0.0,
        pathlength=jnp.zeros((n_rays,)),
    )


@pytest.mark.parametrize("n_rays", [128, 64, 32, 1])
def test_lens_random_rays(n_rays):
    rays = random_rays(n_rays)

    f = jax.random.uniform(key, 1, minval=0.0, maxval=1.0)
    out_manual_dx = rays.x * (-1 / f) + rays.dx
    out_manual_dy = rays.y * (-1 / f) + rays.dy

    lens = comp.Lens(rays.z[0], f)
    out_rays = lens.step(rays)

    # Check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)


def test_lens_focusing_to_point(parallel_rays):

    # If rays are parallel, lens focuses them to a point, so we test this behaviour here.
    f = 0.5
    out_manual_dx = parallel_rays.x * (-1 / f) + parallel_rays.dx
    out_manual_dy = parallel_rays.y * (-1 / f) + parallel_rays.dy
    lens = comp.Lens(parallel_rays.z[0], f)
    out_rays = lens.step(parallel_rays)

    # First check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.x, parallel_rays.x, atol=1e-6)
    assert_allclose(out_rays.y, parallel_rays.y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)

    # Then check that when we propagate these parallel rays,
    # they go to the 0.0 (Parallel rays go to 0.0 in the focal plane)
    propagated_rays = propagate(0.5, out_rays)
    assert_allclose(propagated_rays.x, 0.0, atol=1e-6)
    assert_allclose(propagated_rays.y, 0.0, atol=1e-6)


def test_lens_focusing_to_infinity(point_rays):
    # If rays are a point source and leave from the focal plane,
    # the lens creates a parallel ("Collimated") beam "focused at infinity",
    # so we test this behaviour here.

    f = 0.5
    propagated_rays_to_lens = propagate(f, point_rays)

    out_manual_dx = 0.0
    out_manual_dy = 0.0

    lens = comp.Lens(propagated_rays_to_lens.z, f)
    out_rays = lens.step(propagated_rays_to_lens)

    # First check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.x, propagated_rays_to_lens.x, atol=1e-6)
    assert_allclose(out_rays.y, propagated_rays_to_lens.y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)

    # Then check that when we propagate these now collimated rays a large distance,
    # they move parallel to the optic axis continuously
    propagated_rays = propagate(100000, out_rays)
    assert_allclose(propagated_rays.x, out_rays.x, atol=1e-6)
    assert_allclose(propagated_rays.y, out_rays.y, atol=1e-6)
    assert_allclose(propagated_rays.dx, out_rays.dx, atol=1e-6)
    assert_allclose(propagated_rays.dy, out_rays.dy, atol=1e-6)


def test_lens_path_length_parallel_incoming_rays_to_focal_plane(paraxial_parallel_rays):
    f = 0.5
    lens = comp.Lens(paraxial_parallel_rays.z, focal_length=f)
    out_rays = lens.step(paraxial_parallel_rays)
    rays_at_focal = propagate(f, out_rays)

    # All rays should have the same path length as the distance
    # from the reference sphere to the focal plane
    assert_allclose(rays_at_focal.pathlength, f, atol=1e-5)


@pytest.mark.parametrize("n_rays", [128, 64, 32, 1])
def test_deflector_random_rays(n_rays):
    rays = random_rays(n_rays)
    deflection = jax.random.uniform(key, minval=-0.01, maxval=0.01)
    # The last row of rays should always be 1.0, but random rays randomises
    # this, and we have no getter for the final row because it was always supposed to be 1.0
    # so we use matrix instead to make the test succeed
    out_manual_dx = rays.dx + rays.matrix[:, -1] * deflection
    out_manual_dy = rays.dy + rays.matrix[:, ] * deflection

    deflector = comp.Deflector(rays.z, def_x=deflection, def_y=deflection)
    out_rays = deflector.step(rays)

    # Check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.x, rays.x, atol=1e-6)
    assert_allclose(out_rays.y, rays.y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)


def test_deflector_deflection(parallel_rays):
    deflection = 0.5
    out_manual_dx = parallel_rays.dx + 1.0 * deflection
    out_manual_dy = parallel_rays.dy + 1.0 * deflection
    deflector = comp.Deflector(parallel_rays.z, def_x=deflection, def_y=deflection)
    out_rays = deflector.step(parallel_rays)

    assert_allclose(out_rays.x, parallel_rays.x, atol=1e-6)
    assert_allclose(out_rays.y, parallel_rays.y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)


def test_double_deflector_deflection(parallel_rays):
    separation = 0.1

    defx1 = 0.2
    defy1 = 0.2
    defx2 = -0.2
    defy2 = -0.2

    out_manual_x = parallel_rays.x + separation * defx1
    out_manual_y = parallel_rays.y + separation * defy1

    # The 1.0 multiplying defx1 and defx2 here represents the 1.0 that exists in the matmul
    out_manual_dx = parallel_rays.dx + 1.0 * defx1 + 1.0 * defx2
    out_manual_dy = parallel_rays.dy + 1.0 * defy1 + 1.0 * defy2

    z = (parallel_rays.z + separation) / 2
    double_deflector = comp.DoubleDeflector(
        z=z,
        first=comp.Deflector(
            z=parallel_rays.z, def_x=defx1, def_y=defy1
        ),
        second=comp.Deflector(
            z=parallel_rays.z + separation, def_x=defx2, def_y=defy2
        ),
    )

    out_rays = double_deflector.step(parallel_rays)

    assert_allclose(out_rays.x, out_manual_x, atol=1e-6)
    assert_allclose(out_rays.y, out_manual_y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)


def test_biprism_total_deflection_is_one(point_rays):
    # We need to test our vector calculus that we use to find the deflection vectors in x and y
    # space. If we have done this correctly, the sqrt(deflection_x**2 + deflection_y**2) we
    # calculate if the biprism is rotated should be equal to the total deflection of the biprism.
    rot = jax.random.uniform(key, minval=-180., maxval=180.)
    deflection = jax.random.uniform(key, minval=-0.01, maxval=0.01)
    biprism_z = 0.5
    biprism = comp.Biprism(z=biprism_z, deflection=deflection, rotation=rot)
    biprism_rays = propagate(biprism_z, point_rays)
    out_rays = biprism.step(biprism_rays)

    deflection_x = out_rays.dx - point_rays.dx
    deflection_y = out_rays.dy - point_rays.dy

    total_deflection = jnp.sqrt(deflection_x**2 + deflection_y**2)
    assert_allclose(
        total_deflection,
        jnp.ones(len(point_rays.x)) * jnp.abs(deflection),
        atol=1e-6
    )


def test_biprism_deflection_perpendicular(parallel_rays):
    deflection = -1.0

    out_manual_x = parallel_rays.x
    out_manual_y = parallel_rays.y
    out_manual_dx = parallel_rays.dx + jnp.sign(parallel_rays.x) * deflection
    out_manual_dy = parallel_rays.dy
    biprism = comp.Biprism(z=parallel_rays.z, deflection=deflection)

    out_rays = biprism.step(parallel_rays)

    assert_allclose(out_rays.x, out_manual_x, atol=1e-6)
    assert_allclose(out_rays.y, out_manual_y, atol=1e-6)
    assert_allclose(out_rays.dx, out_manual_dx, atol=1e-6)
    assert_allclose(out_rays.dy, out_manual_dy, atol=1e-6)


@pytest.mark.parametrize(
    "rot, inp, out",
    [
        (45, (1, 1), (-1, -1)),  # Top Right to Bottom Left
        (-45, (-1, 1), (1, -1)),  # Top Left to Bottom Right
        (45, (-1, -1), (1, 1)),  # Bottom Left to Top Right
        (-45, (1, -1), (-1, 1)),  # Bottom Right to Top Left
    ],
)
def test_biprism_deflection_by_quadrant(rot, inp, out):
    ray = single_ray(inp[0], inp[1], 0.0, 0.0)
    # Test that the ray ends up in the correct quadrant if the biprism is rotated
    deflection = -0.3
    biprism = comp.Biprism(z=ray.z, deflection=deflection, rotation=rot)
    biprism_out_rays = biprism.step(ray)
    propagated_rays = propagate(0.2, biprism_out_rays)

    assert_allclose(jnp.sign(propagated_rays.dx), out[0], atol=1e-6)
    assert_allclose(jnp.sign(propagated_rays.dy), out[1], atol=1e-6)


def test_biprism_interference():
    # This test uses an old biprism equation from light optics
    # to calculate the number of peaks in a biprism interefence pattern,
    # The biprism equation tells you the spacing between interference peaks
    # in image plane given an optical setup with a point source, a biprism
    # and a detector. The equation is given by DeltaS = ((a+b)/d) \times wavelength,
    # where a is distance from source to biprism, b is distance from biprism
    # to detector, d is the separation between spots in the source plane
    # and wavelength is the wavelength of the source

    # This tests at least that we have an interference pattern,
    # although this test is not quite so general, and will probably break
    # if any of these parameters are modified significantly
    wavelength = 0.001
    voltage = calculate_phi_0(wavelength)

    deflection = -0.1  # Deflection of biprism
    a = 0.5  # Source to biprism distance
    b = 0.5  # Biprism to image plane
    d = (
        2 * a * deflection
    )  # Source spot spacing (Biprism splits a single source into two)
    spacing = (
        (a + b) / abs(d)
    ) * wavelength  # Interference pattern peak spacing in image plane

    # For this detector size and peak spacing, we expect to see 11 peaks in the
    # detector plane
    det_shape = (1, 101)
    pixel_size = 0.001
    num_peaks = int(pixel_size * det_shape[1] / spacing) + 1

    Rays = source.XPointBeam(z=0.0,
                             n_rays=2**12,
                             semi_angle=0.1,
                             tilt_yx=(0., 0.),
                             centre_yx=(0., 0.),
                             voltage=voltage,
                             random=False)
    model = (
        comp.Biprism(
            z=a,
            offset=0.0,
            rotation=0.0,
            deflection=deflection,
        ),
        comp.Detector(
            z=a + b,
            pixel_size=pixel_size,
            shape=det_shape,
        ),
    )

    # We need enough rays that there is lots of interference in the image plane
    # so that there are definite peaks for peak finder
    out_rays = run_model_to_end(Rays, model)
    image = model[-1].get_image(out_rays)
    peaks, _ = scipy.signal.find_peaks(jnp.abs(image[0, :]) ** 2, height=0)

    assert_equal(len(peaks), num_peaks)


@pytest.mark.skipif(USE_GPU, reason="Test not supported on GPU")
def test_aperture_blocking(parallel_rays):
    blocked_matrix = jnp.ones(len(parallel_rays.x), dtype=jnp.bool_)
    aperture = comp.Aperture(z=parallel_rays.z[0], radius=0.0)
    out_rays = aperture.step(parallel_rays)

    assert jnp.array_equal(out_rays.blocked, blocked_matrix)


@pytest.mark.skipif(USE_GPU, reason="Test not supported on GPU")
def test_aperture_nonblocking(parallel_rays):
    blocked_matrix = jnp.zeros(len(parallel_rays.x), dtype=jnp.bool_)

    aperture = comp.Aperture(z=parallel_rays.z, radius=2.0)
    out_rays = aperture.step(parallel_rays)

    assert jnp.array_equal(out_rays.blocked, blocked_matrix)
