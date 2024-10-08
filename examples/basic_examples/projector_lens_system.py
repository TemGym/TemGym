from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import numpy as np
from temgymbasic.utils import calculate_phi_0

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import sys

n_rays = 1000
wavelength = 0.01

size = 256
det_shape = (size, size)
pixel_size = 0.005

lens_dist = 10
prop_dist = 5

centre_yx = [0.0, 0.0]
coeffs = [0., 0., 0., 0., 0.]
components = (
    comp.PointBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        semi_angle=0.5,
    ),
    comp.ProjectorLensSystem(
        first=comp.Lens(z=2, name='PL1', z1=-2, z2=1),
        second=comp.Lens(z=4, name='PL2', z1=-1, z2=1),
        name='Projector Lens System',
    ),
    comp.Detector(
        z=prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='ray'
    ),
)

model = Model(components, backend='cpu')
# rays = tuple(model.run_iter(num_rays=n_rays, random = False))
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
