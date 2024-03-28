

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
from scipy.interpolate import RegularGridInterpolator as RGI

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
print(potential_array.sampling, potential_array.gpts)
pot_array = potential_array.array[0]
extent = potential_array.extent*1e-10
x = np.linspace(0., potential_array.extent[0], potential_array.gpts[0])*1e-10
y = np.linspace(0., potential_array.extent[1], potential_array.gpts[1])*1e-10
xx, yy = np.meshgrid(x, y)
pot_interp = RGI((x, y), pot_array, method='linear', bounds_error=False, fill_value=0.0)
Ey, Ex = np.gradient(pot_array)
Ex_interp = RGI((x, y), Ex, method='linear', bounds_error=False, fill_value=0.0)
Ey_interp = RGI((x, y), Ey, method='linear', bounds_error=False, fill_value=0.0)

components = (
    comp.ParallelBeam(
        z=0.0,
        phi_0=100e3,
        radius=1e-9,
    ),
    comp.PotentialSample(
        z=0.1,
        potential=pot_interp,
        Ex=Ex_interp,
        Ey=Ey_interp,
    ),
    comp.Detector(
        z=0.2,
        pixel_size=0.000000005,
        shape=(200, 200),
    ),
)

model = Model(components)
rays = tuple(model.run_iter(num_rays=1024))

# plot_model(model, plot_params=PlotParams())

plt.figure()
plt.imshow(model.detector.get_image(rays[-1]).astype(np.bool_))
plt.show()
