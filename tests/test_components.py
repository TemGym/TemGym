import pytest
import numpy as np

from temgymbasic.model import (
    Model,
)
import temgymbasic.components as comp
from temgymbasic.rays import Rays
from temgymbasic.utils import calculate_phi_0, calculate_wavelength, get_array_from_device
from numpy.testing import assert_allclose, assert_equal
from temgymbasic.plotting import plot_model
import scipy
# import matplotlib.pyplot as plt
from typing import Tuple, NamedTuple
from scipy.constants import e, m_e, c

try:
    import cupy as cp
except ImportError:
    cp = None

import os
# Check environment variable to decide backend
USE_GPU = os.getenv('USE_GPU', '0') == '1'

if USE_GPU:
    xp = cp
else:
    xp = np

ETA = (abs(e)/(2*m_e))**(1/2)
EPSILON = abs(e)/(2*m_e*c**2)


@pytest.fixture()
def empty_rays():
    return Rays(
        data=xp.empty(shape=(5, 0)),
        location=0.2,
        path_length=xp.empty([]),
    )


def single_random_uniform_ray(x, y, phi_0=1.0):
    data = xp.zeros(shape=(5, 1))

    data[0, :] = xp.random.uniform(low=0, high=x)
    data[1, :] = xp.zeros(1)
    data[2, :] = xp.random.uniform(low=0, high=y)
    data[3, :] = xp.zeros(1)
    data[4, :] = xp.ones(1)

    return Rays(
        data=data,
        location=0.2,
        path_length=0.0,
        wavelength=calculate_wavelength(phi_0=phi_0)
    )


@pytest.fixture(
    params=[128, 1, 0],
)
def random_rays(request):
    n_rays = request.param
    return Rays(
        data=xp.random.uniform(size=(5, n_rays)),
        location=0.2,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def parallel_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    data[0, :] = xp.random.uniform(low=-1, high=1., size=n_rays)
    data[1, :] = xp.zeros(shape=n_rays)
    data[2, :] = xp.random.uniform(low=-1, high=1., size=n_rays)
    data[3, :] = xp.zeros(shape=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.2,
        path_length=xp.zeros((n_rays,)),
        wavelength=xp.ones(shape=n_rays)*calculate_wavelength(1.0)
    )


@pytest.fixture(
    params=[128],
)
def point_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    data[0, :] = xp.zeros(shape=n_rays)
    data[1, :] = xp.random.uniform(low=-0.1, high=0.1, size=n_rays)
    data[2, :] = xp.zeros(shape=n_rays)
    data[3, :] = xp.random.uniform(low=-0.1, high=0.1, size=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.0,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def slope_of_one_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    data[0, :] = xp.random.uniform(size=n_rays)
    data[1, :] = xp.ones(shape=n_rays)
    data[2, :] = xp.random.uniform(size=n_rays)
    data[3, :] = xp.ones(shape=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays.new(
        data=data,
        location=0.2,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.mark.parametrize(
    "component", [
        (comp.ParallelBeam, (0.01,)),
        (comp.Lens, (0.01,)),
        (comp.Deflector, tuple()),
        (comp.DoubleDeflector.from_params, tuple()),
        (comp.Detector, (0.001, (64, 64))),
        (comp.Aperture, (0.1,)),
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


def test_lens_random_rays(random_rays):
    f = xp.random.uniform(0., 1)
    out_manual_dx = random_rays.x*(-1/f) + random_rays.dx
    out_manual_dy = random_rays.y*(-1/f) + random_rays.dy

    lens = comp.Lens(random_rays.location, f)
    out_rays = tuple(lens.step(random_rays))[0]
    
    # Check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)


def test_lens_focusing_to_point(parallel_rays):
    # If rays are parallel, lens focuses them to a point, so we test this behaviour here.
    f = 0.5
    out_manual_dx = parallel_rays.x*(-1/f) + parallel_rays.dx
    out_manual_dy = parallel_rays.y*(-1/f) + parallel_rays.dy
    lens = comp.Lens(parallel_rays.location, f)
    out_rays = tuple(lens.step(parallel_rays))[0]
    
    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(out_rays.x, parallel_rays.x)
    xp.testing.assert_allclose(out_rays.y, parallel_rays.y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)

    # Then check that when we propagate these parallel rays,
    # they go to the 0.0 (Parallel rays go to 0.0 in the focal plane)
    propagated_rays = out_rays.propagate(0.5)
    xp.testing.assert_allclose(propagated_rays.x, 0.0)
    xp.testing.assert_allclose(propagated_rays.y, 0.0)


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
    xp.testing.assert_allclose(out_rays.x, propagated_rays_to_lens.x)
    xp.testing.assert_allclose(out_rays.y, propagated_rays_to_lens.y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)

    # Then check that when we propagate these now collimated rays a large distance,
    # they move parallel to the optic axis continuously
    propagated_rays = out_rays.propagate(10000)
    xp.testing.assert_allclose(propagated_rays.x, out_rays.x)
    xp.testing.assert_allclose(propagated_rays.y, out_rays.y)
    xp.testing.assert_allclose(propagated_rays.dx, out_rays.dx)
    xp.testing.assert_allclose(propagated_rays.dy, out_rays.dy)


def test_deflector_random_rays(random_rays):
    deflection = xp.random.uniform(-5, 5)

    # The last row of rays should always be 1.0, but random rays randomises
    # this, and we have no getter for the final row because it was always supposed to be 1.0
    # so we use data instead to make the test succeed
    out_manual_dx = random_rays.dx + random_rays.data[-1, :]*deflection
    out_manual_dy = random_rays.dy + random_rays.data[-1, :]*deflection

    deflector = comp.Deflector(random_rays.location, defx=deflection, defy=deflection)
    out_rays = tuple(deflector.step(random_rays))[0]

    # Check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(out_rays.x, random_rays.x)
    xp.testing.assert_allclose(out_rays.y, random_rays.y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)


def test_deflector_deflection(parallel_rays):
    deflection = 0.5
    out_manual_dx = parallel_rays.dx + 1.0*deflection
    out_manual_dy = parallel_rays.dy + 1.0*deflection
    deflector = comp.Deflector(parallel_rays.location, defx=deflection, defy=deflection)
    out_rays = tuple(deflector.step(parallel_rays))[0]

    xp.testing.assert_allclose(out_rays.x, parallel_rays.x)
    xp.testing.assert_allclose(out_rays.y, parallel_rays.y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)


def test_double_deflector_deflection(parallel_rays):
    separation = 0.1

    defx1 = 0.2
    defy1 = 0.2
    defx2 = -0.2
    defy2 = -0.2

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

    xp.testing.assert_allclose(out_rays.x, out_manual_x)
    xp.testing.assert_allclose(out_rays.y, out_manual_y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)


def test_biprism_total_deflection_is_one(point_rays):

    # We need to test our vector calculus that we use to find the deflection vectors in x and y
    # space. If we have done this correctly, the sqrt(deflection_x**2 + deflection_y**2) we
    # calculate if the biprism is rotated should be equal to the total deflection of the biprism.
    rot = xp.random.uniform(-180, 180)
    deflection = xp.random.uniform(-0.01, 0.01)
    biprism_location = 0.5
    biprism = comp.Biprism(z=biprism_location, deflection=deflection, rotation=rot)
    biprism_rays = point_rays.propagate(biprism_location)
    out_rays = tuple(biprism.step(biprism_rays))[0]

    deflection_x = out_rays.dx - point_rays.dx
    deflection_y = out_rays.dy - point_rays.dy

    total_deflection = xp.sqrt(deflection_x**2 + deflection_y**2)
    xp.testing.assert_allclose(total_deflection, xp.ones(point_rays.num)*xp.abs(deflection))


def test_biprism_deflection_perpendicular(parallel_rays):
    deflection = -1.0

    out_manual_x = parallel_rays.x
    out_manual_y = parallel_rays.y
    out_manual_dx = parallel_rays.dx + xp.sign(parallel_rays.x)*deflection
    out_manual_dy = parallel_rays.dy
    biprism = comp.Biprism(z=parallel_rays.location, deflection=deflection)

    out_rays = tuple(biprism.step(parallel_rays))[0]

    xp.testing.assert_allclose(out_rays.x, out_manual_x)
    xp.testing.assert_allclose(out_rays.y, out_manual_y)
    xp.testing.assert_allclose(out_rays.dx, out_manual_dx)
    xp.testing.assert_allclose(out_rays.dy, out_manual_dy)


@pytest.mark.parametrize("rot, inp, out", [
    (45, (1, 1), (-1, -1)),  # Top Right to Bottom Left
    (-45, (-1, 1), (1, -1)),  # Top Left to Bottom Right
    (45, (-1, -1), (1, 1)),  # Bottom Left to Top Right
    (-45, (1, -1), (-1, 1)),  # Bottom Right to Top Left
])
def test_biprism_deflection_by_quadrant(rot, inp, out):
    ray = single_random_uniform_ray(inp[0], inp[1])
    # Test that the ray ends up in the correct quadrant if the biprism is rotated
    deflection = -0.3
    biprism = comp.Biprism(z=ray.location, deflection=deflection, rotation=rot)
    biprism_out_rays = tuple(biprism.step(ray))[0]
    propagated_rays = biprism_out_rays.propagate(0.2)

    xp.testing.assert_allclose(xp.sign(propagated_rays.dx), out[0])
    xp.testing.assert_allclose(xp.sign(propagated_rays.dy), out[1])


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
            voltage=phi_0,
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
    image = model.detector.get_image(rays[-1], interference='ray')
    peaks, _ = scipy.signal.find_peaks(np.abs(image[0, :])**2, height=0)

    assert_equal(len(peaks), num_peaks)

@pytest.mark.skipif(USE_GPU, reason="Test not supported on GPU") 
def test_aperture_blocking(parallel_rays, empty_rays):
    aperture = comp.Aperture(z=parallel_rays.location, radius=0.)
    out_rays = tuple(aperture.step(parallel_rays))[0]
    assert_equal(out_rays.data, empty_rays.data)

@pytest.mark.skipif(USE_GPU, reason="Test not supported on GPU") 
def test_aperture_nonblocking(parallel_rays):
    aperture = comp.Aperture(z=parallel_rays.location, radius=2.)
    out_rays = tuple(aperture.step(parallel_rays))[0]

    assert_equal(out_rays.data, parallel_rays.data)

    
@pytest.mark.skipif(USE_GPU, reason="Test not supported on GPU") 
def test_sample_potential_deflection():
    from scipy.interpolate import RegularGridInterpolator as RGI

    # Sample deflection test: The implemented method we test called in the
    # Potential Sample step function uses the relativistic deflection
    # equations explained in principles of electron optics.

    # The calculated analytical method used to compare our result comes
    # from newton's third law and the lorentz force equation for an electron:
    # We say m * \gamma * a = qE, where m is the rest mass of the electron,
    # \gamma is the relativistic correction factor, a = acceleration, q =
    # the electron charge (-ve for electron) and E is the E field.
    # a is implicitly dv/dt in this equation, but using the relation,
    # d/dt = v/(sqrt(1+x'^2 + y'^2)) * d/dz where x' denotes derivative with
    # respect to z (dx/dz), one can derive the instantaneous force as a function of z, and thus
    # velocity change on the electron in this infinitely thin plane (note dirac delta function is
    # also implied here as dz is an infinitely thin slice), thus -
    # x' += (q * (1+x'^2 + y'^2)) / (m * v^2 * gamma) * Ex
    # One must be careful about the velocity term: I believe it also needs to encode
    # the potential of the "Sample" that the electron sees. I am unsure about this for now,
    # but there is no other term that can account for the sample potential, so it makes sense to me.

    # Set initial potential of electron
    phi_0 = 1.0
    ray = single_random_uniform_ray(0., 0., phi_0=phi_0)

    # Set up sample properties and potential
    x0 = -0.25
    y0 = -0.25
    extent = (1, 1)
    gpts = (256, 256)
    x = xp.linspace(x0, x0 + extent[0], gpts[0], endpoint=True)
    y = xp.linspace(y0, y0 + extent[1], gpts[1], endpoint=True)
    xx, _ = xp.meshgrid(x, y)

    # Set initial potential
    phi_r = xx - x0

    # Find voltage at central pixel
    phi_centre = phi_0 - x0
    phi_hat_centre = (phi_centre)*(1 + EPSILON * (phi_centre))

    # Calculate gamma
    gamma = xp.sqrt(1 + 4 * EPSILON * phi_hat_centre)

    # Calculate velocity two ways for sanity check
    v = 2 * ETA * xp.sqrt(phi_hat_centre) / gamma
    v_acc = 1 * c * (1 - (1 - (- e*(phi_centre)) / (m_e * (c ** 2))) ** (-2)) ** (1/2)

    xp.testing.assert_allclose(v, v_acc, atol=1e-3)
    
    Ey, Ex = xp.gradient(phi_r, x, y)
    x = get_array_from_device(x)
    y = get_array_from_device(y)
    phi_r = get_array_from_device(phi_r)
    Ey, Ex = get_array_from_device(Ey), get_array_from_device(Ex) 
    

    # Interpolate potential
    pot_interp = RGI([x, y], phi_r, method='linear', bounds_error=False, fill_value=0.0)
    Ex_interp = RGI([x, y], Ex, method='linear', bounds_error=False, fill_value=0.0)
    Ey_interp = RGI([x, y], Ey, method='linear', bounds_error=False, fill_value=0.0)

    sample = comp.PotentialSample(
        z=ray.location,
        potential=pot_interp,
        Ex=Ex_interp,
        Ey=Ey_interp,
    )

    # rho encodes the slope of the ray into a single parameter and is needed for when the calc
    # is performed as a function of d/dz, instead of d/dt
    rho = xp.sqrt(1 + ray.dx ** 2 + ray.dy ** 2)

    # Step rays
    import pdb; pdb.set_trace()
    out_rays = tuple(sample.step(ray))[0]

    # Analytical calculation to the slope change - See Szilagyi Ion and Electron Optics also
    # but their derivation is not well explained, and is verbose, so this is what we have.
    dx_analytical_one = xp.float64((e * rho ** 2)/(gamma * m_e * v * v)) * xp.max(Ex)

    # This other equation comes from solving the equation for an electrosatic deflector,
    # I need to do the derivation from scratch in the
    # relativistic sense, as there is another gamma that I don't quite understand
    dx_analytical_two = (1.0*gamma)/(2*phi_hat_centre)

    xp.testing.assert_allclose(out_rays.dx, dx_analytical_one, atol=1e-7)
    xp.testing.assert_allclose(out_rays.dx, dx_analytical_two, atol=1e-7)


@pytest.mark.skip(reason="No way to numerically test this now, so visualise plot below to check")
def test_sample_phase_shift():
    from scipy.interpolate import RegularGridInterpolator as RGI, interp1d

    class PlotParams(NamedTuple):
        num_rays: int = 10
        ray_color: str = 'dimgray'
        fill_color: str = 'aquamarine'
        fill_color_pair: Tuple[str, str] = ('khaki', 'deepskyblue')
        fill_alpha: float = 0.0
        ray_alpha: float = 1.
        component_lw: float = 1.
        edge_lw: float = 1.
        ray_lw: float = 1.
        label_fontsize: int = 12
        figsize: Tuple[int, int] = (6, 6)
        extent_scale: float = 1.1

    phi_0 = 1.1
    x0 = -0.5
    y0 = -0.5

    extent = (1, 1)
    gpts = (256, 256)
    x = xp.linspace(x0, x0 + extent[0], gpts[0], endpoint=True)
    y = xp.linspace(y0, y0 + extent[1], gpts[1], endpoint=True)
    xx, yy = xp.meshgrid(x, y, indexing='ij')

    phi_r = xx - x0

    Ey, Ex = xp.gradient(phi_r, x, y)
    
    x = get_array_from_device(x)
    y = get_array_from_device(y)
    phi_r = get_array_from_device(phi_r)
    Ey, Ex = get_array_from_device(Ey), get_array_from_device(Ex) 
    
    # Interpolate potential
    pot_interp = RGI([x, y], phi_r, method='linear', bounds_error=False, fill_value=0.0)
    Ex_interp = RGI([x, y], Ex, method='linear', bounds_error=False, fill_value=0.0)
    Ey_interp = RGI([x, y], Ey, method='linear', bounds_error=False, fill_value=0.0)

    components = (
        comp.XAxialBeam(
            z=0.0,
            radius=0.5,
            voltage=phi_0
        ),
        comp.PotentialSample(
            z=0.5,
            potential=pot_interp,
            Ex=Ex_interp,
            Ey=Ey_interp,
        ),
        comp.Detector(
            z=6,
            pixel_size=0.01,
            shape=(100, 100),
        ),
    )

    num_rays = 100
    plot_params = PlotParams(num_rays=num_rays)

    model = Model(components)
    fig, ax = plot_model(model, plot_params=plot_params)
    rays = tuple(model.run_iter(num_rays=num_rays))
    print(rays[1].path_length[1], rays[1].path_length[2])
    x = xp.stack(tuple(r.x for r in rays), axis=0)
    z = xp.asarray(tuple(r.z for r in rays))
    opl = xp.asarray(tuple(r.path_length for r in rays))

    opls = xp.linspace(0.5, 5, 11)

    for idx in range(num_rays):

        # Interpolation for x and z as functions of path length
        z_of_L = interp1d(opl[:, idx], z, kind='linear')
        x_of_z = interp1d(z, x[:, idx])

        # Find x and z for the given path length L'
        z_prime = z_of_L(opls)
        x_prime = x_of_z(z_prime)

        ax.plot(x_prime, z_prime, '.r')

    # Uncomment these plotting lines to see if the wavefront looks correct

    # plt.axis('equal')
    # plt.show()
