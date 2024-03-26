

from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp


import matplotlib.pyplot as plt
import numpy as np
from ase.build import graphene
from typing import Tuple, NamedTuple
import abtem
from temgymbasic.plotting import plot_model


class PlotParams(NamedTuple):
    num_rays: int = 1024
    ray_color: str = 'dimgray'
    fill_color: str = 'aquamarine'
    fill_color_pair: Tuple[str, str] = ('khaki', 'deepskyblue')
    fill_alpha: float = 0.5
    ray_alpha: float = 1.
    component_lw: float = 1.
    edge_lw: float = 1.
    ray_lw: float = 0.1
    label_fontsize: int = 12
    figsize: Tuple[int, int] = (6, 6)
    extent_scale: float = 1e-10


atoms = abtem.orthogonalize_cell(graphene(vacuum=0.5))

atoms *= (5, 3, 1)

atoms.numbers[17] = 14
potential = abtem.Potential(atoms, sampling=[0.05, 0.05])

potential_array = potential.build().compute()
pot_array = potential_array.array[0]

components = (
    comp.ParallelBeam(
        z=0.0,
        phi_0=100e3,
        radius=1e-9,
    ),
    comp.Sample(
        z=0.1,
        potential=pot_array,
        pixel_size=potential.sampling[0]*1e-10,
        shape=pot_array.shape,
    ),
    comp.Detector(
        z=0.2,
        pixel_size=0.000000005,
        shape=(200, 200),
    ),
)

model = Model(components)
rays = tuple(model.run_iter(num_rays=1024))

plot_model(model, plot_params=PlotParams())

plt.figure()
plt.imshow(model.detector.get_image(rays[-1]).astype(np.bool_))
plt.show()
