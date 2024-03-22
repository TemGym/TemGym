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
    fill_alpha: float = 0.5
    ray_alpha: float = 1.
    component_lw: float = 1.
    edge_lw: float = 1.
    ray_lw: float = 0.1
    label_fontsize: int = 12
    figsize: Tuple[int, int] = (6, 6)
    extent_scale: float = 1.1

wavelength = 0.001
deflection = 0.1
a = 0.5
b = 0.5
d=2*a*deflection
spacing = (a+b)/d*wavelength
print(spacing)

components = (
    comp.PointBeam(
        z=0.0,
        wavelength=wavelength,
        semi_angle=0.1,
        random=True,
    ),
    comp.Biprism(
        z=a,
        offset=0.0,
        rotation=np.pi/2,
        deflection=deflection,
    ),
    comp.Detector(
        z=a+b,
        pixel_size=0.0005,
        shape=(200, 200),
    ),
)

num_rays = 2*20
model = Model(components)
rays = tuple(model.run_iter(num_rays=num_rays))
image = model.detector.get_image_intensity(rays[-1])

plt.figure()
plt.imshow(np.abs(image)**2/np.max(np.abs(image)**2))
plt.show()
