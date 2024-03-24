import pytest
import numpy as np

import temgymbasic.components as comp
from temgymbasic.rays import Rays
from temgymbasic.model import Model
from numpy.testing import assert_allclose


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


def test_lens_focusing():
    num_rays = 512
    components = (
        comp.ParallelBeam(
            z=0.0,
            radius=0.5,
        ),
        comp.Lens(
            z=0.5,
            f=0.5,
        ),
        comp.Detector(
            z=1.0,
            pixel_size=0.1,
            shape=(10, 10),
        ),
    )
    model = Model(components)
    rays = model.run_to_end(num_rays=num_rays)
    assert_allclose(rays.x, 0.0)
    assert_allclose(rays.y, 0.0)


def test_deflector_deflection():
    deflection = 1.0
    num_rays = 512
    components = (
        comp.ParallelBeam(
            z=0.0,
            radius=0.5,
        ),
        comp.Deflector(
            z=0.5,
            defx=deflection,
            defy=deflection,
        ),
        comp.Detector(
            z=1.0,
            pixel_size=0.0005,
            shape=(200, 200),
        ),
    )
    model = Model(components)
    rays = model.run_to_end(num_rays=num_rays)
    assert_allclose(rays.dx, deflection)
    assert_allclose(rays.dy, deflection)


def test_double_deflector_deflection():
    num_rays = 512
    components = (
        comp.ParallelBeam(
            z=0.0,
            radius=0.5,
        ),
        comp.DoubleDeflector(
            first=comp.Deflector(z=0.1, defx=1, defy=1, name='Upper'),
            second=comp.Deflector(z=0.2, defx=-1, defy=-1, name='Lower'),
        ),
        comp.Detector(
            z=0.3,
            pixel_size=0.0005,
            shape=(200, 200),
        ),
    )

    model = Model(components)
    rays = model.run_to_end(num_rays=num_rays)
    assert_allclose(rays.dx, 0.0)
    assert_allclose(rays.dy, 0.0)


def test_biprism_deflection():
    num_rays = 512
    components = (
        comp.ParallelBeam(
            z=0.0,
            radius=0.5,
        ),
        comp.Biprism(
            z=0.1,
            defx=1.0
        ),
        comp.Detector(
            z=0.3,
            pixel_size=0.0005,
            shape=(200, 200),
        ),
    )

    model = Model(components)
    rays = tuple(model.run_iter(num_rays=num_rays))
    biprism_rays = rays[1]
    positive_ray_idcs = biprism_rays.x > 0.0
    negative_ray_idcs = biprism_rays.x < 0.0

    assert_allclose(rays[2].dx[0], 0.0)
    assert_allclose(rays[2].dx[positive_ray_idcs], 1.0)
    assert_allclose(rays[2].dx[negative_ray_idcs], -1.0)
    assert_allclose(rays[2].dy, 0.0)


def test_aperture_blocking():
    num_rays = 512
    components = (
        comp.ParallelBeam(
            z=0.0,
            radius=0.5,
        ),
        comp.Aperture(
            z=0.1,
            radius_inner=0.0,
            radius_outer=1.0,
        ),
        comp.Detector(
            z=0.3,
            pixel_size=0.0005,
            shape=(200, 200),
        ),
    )

    model = Model(components)
    rays = tuple(model.run_iter(num_rays=num_rays))

    #Don't understand how aperture works
    assert_allclose(1, 0)


