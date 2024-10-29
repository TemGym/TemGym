import pytest
import numpy as np

import temgymbasic.components as comp
from temgymbasic.rays import Rays
from temgymbasic.utils import (
    calculate_wavelength,
    concentric_rings,
)
from scipy.constants import e, m_e, c
import os

import cupy as cp
# Check environment variable to decide backend
USE_GPU = os.getenv("USE_GPU", "0") == "0"

if USE_GPU:
    xp = cp
else:
    xp = np

xp = np
ETA = (abs(e) / (2 * m_e)) ** (1 / 2)
EPSILON = abs(e) / (2 * m_e * c**2)


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
        wavelength=calculate_wavelength(phi_0=phi_0),
    )


def single_ray(x, dx, y, dy, phi_0=1.0):
    n_rays = len(x)
    data = xp.zeros(shape=(5, n_rays))

    data[0, :] = x
    data[1, :] = dx
    data[2, :] = y
    data[3, :] = dy
    data[4, :] = xp.ones(1)

    return Rays(
        data=data,
        location=0.2,
        path_length=0.0,
        wavelength=calculate_wavelength(phi_0=phi_0),
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

    data[0, :] = xp.random.uniform(low=-1, high=1.0, size=n_rays)
    data[1, :] = xp.zeros(shape=n_rays)
    data[2, :] = xp.random.uniform(low=-1, high=1.0, size=n_rays)
    data[3, :] = xp.zeros(shape=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.2,
        path_length=xp.zeros((n_rays,)),
        wavelength=xp.ones(shape=n_rays) * calculate_wavelength(1.0),
    )


@pytest.fixture(
    params=[128],
)
def paraxial_parallel_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))
    radius = 1e-3
    low = -radius / 2
    high = radius / 2
    data[0, :] = xp.random.uniform(low=low, high=high, size=n_rays)
    data[1, :] = xp.zeros(shape=n_rays)
    data[2, :] = xp.random.uniform(low=low, high=high, size=n_rays)
    data[3, :] = xp.zeros(shape=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.2,
        path_length=xp.zeros((n_rays,)),
        wavelength=xp.ones(shape=n_rays) * calculate_wavelength(1.0),
    )


@pytest.fixture(
    params=[128],
)
def point_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    radius = 0.5
    low = -radius / 2
    high = radius / 2

    data[0, :] = xp.zeros(shape=n_rays)
    data[1, :] = xp.random.uniform(low=low, high=high, size=n_rays)
    data[2, :] = xp.zeros(shape=n_rays)
    data[3, :] = xp.random.uniform(low=low, high=high, size=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.0,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def increasing_slope_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    y, x = concentric_rings(n_rays, 1e-2)
    data[0, :] = xp.zeros(shape=n_rays)
    data[1, :] = x
    data[2, :] = xp.zeros(shape=n_rays)
    data[3, :] = y
    data[4, :] = xp.ones(shape=n_rays)
    return Rays(
        data=data,
        location=0.0,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.fixture(
    params=[128],
)
def paraxial_point_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    radius = 3e-2
    low = -radius / 2
    high = radius / 2

    data[0, :] = xp.zeros(shape=n_rays)
    data[1, :] = xp.random.uniform(low=low, high=high, size=n_rays)
    data[2, :] = xp.zeros(shape=n_rays)
    data[3, :] = xp.random.uniform(low=low, high=high, size=n_rays)
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


@pytest.fixture(
    params=[128],
)
def square_lattice_rays(request):
    n_rays = request.param
    data = xp.zeros(shape=(5, n_rays))

    n = int(n_rays ** 0.5)
    x = xp.linspace(-0.001, 0.001, n)
    y = xp.linspace(-0.001, 0.001, n)

    x, y = xp.meshgrid(x, y)

    data[0, :] = x.flatten()
    data[1, :] = xp.zeros(shape=n_rays)
    data[2, :] = y.flatten()
    data[3, :] = xp.zeros(shape=n_rays)
    data[4, :] = xp.ones(shape=n_rays)

    return Rays(
        data=data,
        location=0.2,
        path_length=xp.zeros((n_rays,)),
    )


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.0], [0.01], [0.0], [0.0]),
        (
            np.zeros(100),
            np.random.uniform(-0.01, 0.01, 100),
            np.zeros(100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_matrix_lens_spherical_aberration(x, dx, y, dy):
    z_o = -10
    z_i = 11

    R = z_i
    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    x_a = lens_rays.x
    y_a = lens_rays.y

    B = 10

    coeffs = comp.LensAberrations(B, 0.0, 0.0, 0.0, 0.0)

    lens = comp.Lens(z=lens_rays.location, z1=z_o, z2=z_i, aber_coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * (1.0 * B * x_a * (x_a**2 + y_a**2))
    dy = y_i + -R * (1.0 * B * y_a * (x_a**2 + y_a**2))

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-9)

    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(delta_x_i, delta_y_i, 'or')
    # plt.savefig('test_aberrated_lens_spherical.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.01], [0.01], [0.0], [0.0]),
        (
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_matrix_lens_coma(x, dx, y, dy):
    z_o = -10
    z_i = 11

    R = z_i
    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    x_a = lens_rays.x
    y_a = lens_rays.y

    F = 10

    coeffs = comp.LensAberrations(0, F, 0.0, 0.0, 0.0)

    lens = comp.Lens(z=lens_rays.location, z1=z_o, z2=z_i, aber_coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * (
        2 * F * x_a * (x_a * x_o + y_a * y_o) + F * x_o * (x_a**2 + y_a**2)
    )
    dy = y_i + -R * (
        2 * F * y_a * (x_a * x_o + y_a * y_o) + F * y_o * (x_a**2 + y_a**2)
    )

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-9)

    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(delta_x_i, delta_y_i, 'or')
    # plt.savefig('test_aberrated_lens_spherical.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.01], [0.01], [0.0], [0.0]),
        (
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_matrix_lens_astigmatism(x, dx, y, dy):
    z_o = -10
    z_i = 11

    R = z_i
    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    x_a = lens_rays.x
    y_a = lens_rays.y

    C = 10

    coeffs = comp.LensAberrations(0.0, 0.0, C, 0.0, 0.0)

    lens = comp.Lens(z=lens_rays.location, z1=z_o, z2=z_i, aber_coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * C*x_o*(x_a*x_o + y_a*y_o)  # Astigmatism
    dy = y_i + -R * C*y_o*(x_a*x_o + y_a*y_o)  # Astigmatism

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-9)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(dx, dy, 'or')
    # plt.savefig('test_aberrated_lens_astigmatism.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.01], [0.01], [0.0], [0.0]),
        (
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_matrix_lens_field_curvature(x, dx, y, dy):
    z_o = -10
    z_i = 11

    R = z_i
    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    x_a = lens_rays.x
    y_a = lens_rays.y

    D = 10

    coeffs = comp.LensAberrations(0.0, 0.0, 0.0, D, 0.0)

    lens = comp.Lens(z=lens_rays.location, z1=z_o, z2=z_i, aber_coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * D*x_a*(x_o**2 + y_o**2)  # Field Curvature
    dy = y_i + -R * D*y_a*(x_o**2 + y_o**2)  # Field Curvature

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-9)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(dx, dy, 'or')
    # plt.savefig('test_aberrated_lens_astigmatism.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.01], [0.01], [0.0], [0.0]),
        (
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_matrix_lens_distortion(x, dx, y, dy):
    z_o = -10
    z_i = 11

    R = z_i
    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    E = 10

    coeffs = comp.LensAberrations(0.0, 0.0, 0.0, 0.0, E)

    lens = comp.Lens(z=lens_rays.location, z1=z_o, z2=z_i, aber_coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * E*x_o * (x_o**2 + y_o**2)  # Field Curvature
    dy = y_i + -R * E*y_o * (x_o**2 + y_o**2)  # Field Curvature

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-9)

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(dx, dy, 'or')
    # plt.savefig('test_aberrated_lens_astigmatism.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.0], [0.01], [0.0], [0.0]),
        (
            np.zeros(100),
            np.random.uniform(-0.01, 0.01, 100),
            np.zeros(100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_lens_spherical_aberration(x, dx, y, dy):
    z_o = -10
    z_i = 11
    f = 4

    M = z_i / z_o

    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y
    x_o_slope = input_rays.dx
    y_o_slope = input_rays.dy

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))

    B = 10

    coeffs = [B, 0.0, 0.0, 0.0, 0.0]
    lens = comp.AberratedLens(z=lens_rays.location, f=f, z1=z_o, z2=z_i, coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    delta_x_i = x_i + M * (B * x_o_slope * (x_o_slope**2 + y_o_slope**2))
    delta_y_i = y_i + M * (B * y_o_slope * (x_o_slope**2 + y_o_slope**2))

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, delta_x_i, atol=1e-9)
    xp.testing.assert_allclose(propagated_rays.y, delta_y_i, atol=1e-9)

    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(delta_x_i, delta_y_i, 'or')
    # plt.savefig('test_aberrated_lens_spherical.png')


@pytest.mark.parametrize(
    "x, dx, y, dy",
    [
        ([0.01], [0.01], [0.0], [0.0]),
        (
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
            np.random.uniform(-0.001, 0.001, 100),
            np.random.uniform(-0.01, 0.01, 100),
        ),
    ],
)
def test_aberrated_lens_coma(x, dx, y, dy):
    z_o = -10
    z_i = 11
    f = 4

    M = z_i / z_o

    R = z_i
    input_rays = single_ray(x, dx, y, dy)

    x_o = input_rays.x
    y_o = input_rays.y

    x_i = M * x_o
    y_i = M * y_o

    lens_rays = input_rays.propagate(abs(z_o))
    x_a = lens_rays.x
    y_a = lens_rays.y

    F = 1

    coeffs = [0, F, 0.0, 0.0, 0.0]
    B, F, C, D, E = coeffs
    lens = comp.AberratedLens(z=lens_rays.location, f=f, z1=z_o, z2=z_i, coeffs=coeffs)
    out_rays = tuple(lens.step(lens_rays))[0]
    propagated_rays = out_rays.propagate(z_i)

    dx = x_i + -R * (
        1.0 * B * x_a * (x_a**2 + y_a**2)
        + 1.0 * C * x_o * (x_a * x_o + y_a * y_o)
        + 1.0 * D * x_a * (x_o**2 + y_o**2)
        + E * x_o * (x_o**2 + y_o**2)
        + 2 * F * x_a * (x_a * x_o + y_a * y_o)
        + F * x_o * (x_a**2 + y_a**2)
    )
    dy = y_i + -R * (
        1.0 * B * y_a * (x_a**2 + y_a**2)
        + 1.0 * C * y_o * (x_a * x_o + y_a * y_o)
        + 1.0 * D * y_a * (x_o**2 + y_o**2)
        + E * y_o * (x_o**2 + y_o**2)
        + 2 * F * y_a * (x_a * x_o + y_a * y_o)
        + F * y_o * (x_a**2 + y_a**2)
    )

    # First check that the lens has applied the correct deflection to rays
    xp.testing.assert_allclose(propagated_rays.x, dx, atol=1e-5)
    xp.testing.assert_allclose(propagated_rays.y, dy, atol=1e-5)

    # plt.figure()
    # plt.plot(propagated_rays.x, propagated_rays.y, 'og')
    # plt.plot(delta_x_i, delta_y_i, 'or')
    # plt.savefig('test_aberrated_lens_spherical.png')
