import pytest
import numpy as np

from temgymbasic.model import (
    Model,
)
import temgymbasic.components as comp
from temgymbasic.rays import Rays
from numpy.testing import assert_allclose, assert_equal
import scipy


@pytest.fixture()
def empty_rays():
    return Rays(
        data=np.empty(shape=(5, 0)),
        indices=np.empty([]),
        location=0.2,
        path_length=np.empty([]),
    )


def single_quadrant_ray(x, y):
    data = np.zeros(shape=(5, 1))

    data[0, :] = np.random.uniform(low=0, high=x)
    data[1, :] = np.zeros(1)
    data[2, :] = np.random.uniform(low=0, high=y)
    data[3, :] = np.zeros(1)
    data[4, :] = np.ones(1)

    return Rays(
        data=data,
        indices=1,
        location=0.2,
        path_length=0.0,
    )


@pytest.fixture(
    params=[128, 1, 0],
)
def random_rays(request):
    n_rays = request.param
    return Rays(
        data=np.random.uniform(size=(5, n_rays)),
        indices=np.arange(n_rays),
        location=0.2,
        path_length=np.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def parallel_rays(request):
    n_rays = request.param
    data = np.zeros(shape=(5, n_rays))

    data[0, :] = np.random.uniform(low=-1, high=1., size=n_rays)
    data[1, :] = np.zeros(shape=n_rays)
    data[2, :] = np.random.uniform(low=-1, high=1., size=n_rays)
    data[3, :] = np.zeros(shape=n_rays)
    data[4, :] = np.ones(shape=n_rays)

    return Rays(
        data=data,
        indices=np.arange(n_rays),
        location=0.2,
        path_length=np.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def point_rays(request):
    n_rays = request.param
    data = np.zeros(shape=(5, n_rays))

    data[0, :] = np.zeros(shape=n_rays)
    data[1, :] = np.random.uniform(low=-1, high=1., size=n_rays)
    data[2, :] = np.zeros(shape=n_rays)
    data[3, :] = np.random.uniform(low=-1, high=1., size=n_rays)
    data[4, :] = np.ones(shape=n_rays)

    return Rays(
        data=data,
        indices=np.arange(n_rays),
        location=0.0,
        path_length=np.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def slope_of_one_rays(request):
    n_rays = request.param
    data = np.zeros(shape=(5, n_rays))

    data[0, :] = np.random.uniform(size=n_rays)
    data[1, :] = np.ones(shape=n_rays)
    data[2, :] = np.random.uniform(size=n_rays)
    data[3, :] = np.ones(shape=n_rays)
    data[4, :] = np.ones(shape=n_rays)

    return Rays(
        data=data,
        indices=np.arange(n_rays),
        location=0.2,
        path_length=np.zeros((n_rays,)),
    )


@pytest.mark.parametrize(
    "component", [
        (comp.ParallelBeam, (0.1,)),
        (comp.Lens, (0.1,)),
        (comp.Deflector, tuple()),
        (comp.DoubleDeflector.from_params, tuple()),
        (comp.Detector, (0.001, (64, 64))),
        (comp.Aperture, tuple()),
        (comp.Biprism, tuple()),
    ]
)
def test_interface(component, random_rays):
    comp_cls, args = component
    comp = comp_cls(random_rays.location, *args)
    out_rays = tuple(comp.step(random_rays))
    out_rays = out_rays[-1]
    assert isinstance(out_rays, Rays)
    try:
        location = out_rays.location[0]
    except TypeError:
        location = out_rays.location
    assert location is comp
    assert out_rays.num <= random_rays.num


def test_lens_focusing_to_point(parallel_rays):
    # If rays are parallel, lens focuses them to a point, so we test this behaviour here.
    f = 0.5
    out_manual_dx = parallel_rays.x*(-1/f) + parallel_rays.dx
    out_manual_dy = parallel_rays.y*(-1/f) + parallel_rays.dy
    lens = comp.Lens(parallel_rays.location, f)
    out_rays = tuple(lens.step(parallel_rays))[0]

    # First check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.x, parallel_rays.x)
    assert_allclose(out_rays.y, parallel_rays.y)
    assert_allclose(out_rays.dx, out_manual_dx)
    assert_allclose(out_rays.dy, out_manual_dy)

    # Then check that when we propagate these parallel rays,
    # they go to the 0.0 (Parallel rays go to 0.0 in the focal plane)
    propagated_rays = out_rays.propagate(0.5)
    assert_allclose(propagated_rays.x, 0.0)
    assert_allclose(propagated_rays.y, 0.0)


def test_lens_focusing_to_infinity(point_rays):
    # If rays are a point source and leave from the focal plane,
    # the lens creates a parallel ("Collimated") beam "focused at infinity",
    # so we test this behaviour here.

    f = 0.5

    propagated_rays_to_lens = point_rays.propagate(f)

    out_manual_dx = 0.0
    out_manual_dy = 0.0

    lens = comp.Lens(propagated_rays_to_lens.location, f)
    out_rays = tuple(lens.step(propagated_rays_to_lens))[0]

    # First check that the lens has applied the correct deflection to rays
    assert_allclose(out_rays.x, propagated_rays_to_lens.x)
    assert_allclose(out_rays.y, propagated_rays_to_lens.y)
    assert_allclose(out_rays.dx, out_manual_dx)
    assert_allclose(out_rays.dy, out_manual_dy)

    # Then check that when we propagate these now collimated rays a large distance,
    # they move parallel to the optic axis continuously
    propagated_rays = out_rays.propagate(10000)
    assert_allclose(propagated_rays.x, out_rays.x)
    assert_allclose(propagated_rays.y, out_rays.y)
    assert_allclose(propagated_rays.dx, out_rays.dx)
    assert_allclose(propagated_rays.dy, out_rays.dy)


def test_deflector_deflection(parallel_rays):
    deflection = 0.5
    out_manual_dx = parallel_rays.dx + 1.0*deflection
    out_manual_dy = parallel_rays.dy + 1.0*deflection
    deflector = comp.Deflector(parallel_rays.location, defx=deflection, defy=deflection)
    out_rays = tuple(deflector.step(parallel_rays))[0]

    assert_allclose(out_rays.x, parallel_rays.x)
    assert_allclose(out_rays.y, parallel_rays.y)
    assert_allclose(out_rays.dx, out_manual_dx)
    assert_allclose(out_rays.dy, out_manual_dy)


def test_double_deflector_deflection(parallel_rays):
    separation = 0.1

    defx1 = 1
    defy1 = 1
    defx2 = -1
    defy2 = -1

    out_manual_x = parallel_rays.x + separation*defx1
    out_manual_y = parallel_rays.y + separation*defy1

    # The 1.0 multiplying defx1 and defx2 here represents the 1.0 that exists in the matmul
    out_manual_dx = parallel_rays.dx + 1.0*defx1 + 1.0*defx2
    out_manual_dy = parallel_rays.dy + 1.0*defy1 + 1.0*defy2

    double_deflector = comp.DoubleDeflector(
        first=comp.Deflector(z=parallel_rays.location, defx=defx1, defy=defy1, name='Upper'),
        second=comp.Deflector(z=parallel_rays.location+separation, defx=defx2,
                              defy=defy2, name='Lower'),
    )

    out_rays = tuple(double_deflector.step(parallel_rays))[1]

    assert_allclose(out_rays.x, out_manual_x)
    assert_allclose(out_rays.y, out_manual_y)
    assert_allclose(out_rays.dx, out_manual_dx)
    assert_allclose(out_rays.dy, out_manual_dy)


def test_biprism_deflection_perpendicular(parallel_rays):
    deflection = -1.0

    out_manual_x = parallel_rays.x
    out_manual_y = parallel_rays.y
    out_manual_dx = parallel_rays.dx + np.sign(parallel_rays.x)*deflection
    out_manual_dy = parallel_rays.dy
    biprism = comp.Biprism(z=parallel_rays.location, deflection=deflection)

    out_rays = tuple(biprism.step(parallel_rays))[0]

    assert_allclose(out_rays.x, out_manual_x)
    assert_allclose(out_rays.y, out_manual_y)
    assert_allclose(out_rays.dx, out_manual_dx)
    assert_allclose(out_rays.dy, out_manual_dy)


@pytest.mark.parametrize("rot, inp, out", [
    (45, (1, 1), (-1, -1)),  # Top Right to Bottom Left
    (-45, (-1, 1), (1, -1)),  # Top Left to Bottom Right
    (45, (-1, -1), (1, 1)),  # Bottom Left to Top Right
    (-45, (1, -1), (-1, 1)),  # Bottom Right to Top Left
])
def test_biprism_deflection_top_right_quadrant(rot, inp, out):
    ray = single_quadrant_ray(inp[0], inp[1])
    # Test that the ray ends up in the correct quadrant if the biprism is rotated
    deflection = -100.0
    biprism = comp.Biprism(z=ray.location, deflection=deflection, rotation=rot)
    biprism_out_rays = tuple(biprism.step(ray))[0]
    propagated_rays = biprism_out_rays.propagate(0.2)

    assert_allclose(np.sign(propagated_rays.dx), out[0])
    assert_allclose(np.sign(propagated_rays.dy), out[1])


def test_biprism_interference():
    from temgymbasic.utils import calculate_phi_0
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
    phi_0 = calculate_phi_0(wavelength)

    deflection = -0.1  # Deflection of biprism
    a = 0.5  # Source to biprism distance
    b = 0.5  # Biprism to image plane
    d = 2*a*deflection  # Source spot spacing (Biprism splits a single source into two)
    spacing = ((a+b)/abs(d))*wavelength  # Interference pattern peak spacing in image plane

    # For this detector size and peak spacing, we expect to see 11 peaks in the
    # detector plane
    det_shape = (1, 101)
    pixel_size = 0.001
    num_peaks = int(pixel_size*det_shape[1]/spacing) + 1

    components = (
        comp.XPointBeam(
            z=0.0,
            phi_0=phi_0,
            semi_angle=0.1,
        ),
        comp.Biprism(
            z=a,
            offset=0.0,
            rotation=0.0,
            deflection=deflection,
        ),
        comp.Detector(
            z=a+b,
            pixel_size=pixel_size,
            shape=det_shape,
        ),
    )

    model = Model(components)

    # We need enough rays that there is lots of interference in the image plane
    # so that there are definite peaks for peak finder
    rays = tuple(model.run_iter(num_rays=2**20))
    image = model.detector.get_image_intensity(rays[-1])
    peaks, _ = scipy.signal.find_peaks(np.abs(image[0, :])**2, height=0)

    assert_equal(len(peaks), num_peaks)


def test_aperture_blocking(parallel_rays, empty_rays):
    aperture = comp.Aperture(z=parallel_rays.location, radius_inner=2.0, radius_outer=2.0)
    out_rays = tuple(aperture.step(parallel_rays))[0]
    assert_equal(out_rays.data, empty_rays.data)


def test_aperture_nonblocking(parallel_rays):
    aperture = comp.Aperture(z=parallel_rays.location, radius_inner=0., radius_outer=2.0)
    out_rays = tuple(aperture.step(parallel_rays))[0]
    assert_equal(out_rays.data, parallel_rays.data)
