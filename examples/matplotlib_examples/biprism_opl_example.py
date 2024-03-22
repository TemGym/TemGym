import sys
from temgymbasic.model import (
    Model,
)
from typing import Tuple, NamedTuple
from temgymbasic import components as comp
from temgymbasic.plotting import plot_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

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

components = (
    comp.XAxialBeam(
        z=0.0,
        radius=0.01,
    ),
    comp.Biprism(
        z=0.5,
        offset=0.0,
        rotation=np.pi/2,
        deflection=0.1,
    ),
    comp.Detector(
        z=1.,
        pixel_size=0.01,
        shape=(128, 128),
    ),
)

num_rays = 10
plot_params = PlotParams(num_rays = num_rays)

model = Model(components)
fig, ax = plot_model(model, plot_params=plot_params)
rays = tuple(model.run_iter(num_rays=num_rays))
print(rays[1].path_length[1], rays[1].path_length[2])
x = np.stack(tuple(r.x for r in rays), axis=0)
z = np.asarray(tuple(r.z for r in rays))
opl = np.asarray(tuple(r.path_length for r in rays))

opls = np.linspace(0, 1, 11)

for idx in range(num_rays):

    # Interpolation for x and z as functions of path length
    z_of_L = interp1d(opl[:, idx], z, kind='linear')
    x_of_z = interp1d(z, x[:, idx]) 

    # Find x and z for the given path length L'
    z_prime = z_of_L(opls)
    x_prime = x_of_z(z_prime)

    ax.plot(x_prime, z_prime, '.r')

plt.axis('equal')
plt.show()
