import pytest
import numpy as np

import temgymbasic.components as comp
from temgymbasic.rays import Rays


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
