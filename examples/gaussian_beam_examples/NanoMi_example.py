from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import numpy as np

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import sys


n_rays=1000
voltage = 60e3

size = 256
det_shape = (size, size)
pixel_size = 10e-6

theta_x = 0
theta_y = 0
deg_yx = np.deg2rad((theta_y, theta_x))
tilt_yx = np.tan(deg_yx)

lens_dist = 10
prop_dist = 5

z_o = -lens_dist
z_i = lens_dist + prop_dist
f = (1 / z_i - 1 / z_o) ** -1
coeffs = comp.LensAberrations(1e5, 0., 0., 0., 0.)

components = (
    comp.PointBeam(
        z=0.0,
        voltage=voltage,
        semi_angle=0.0005,
    ),
    comp.Lens(
        z=0.1,
        f=0.1,
        z1=-0.1,
        z2=0.1,
        name='Condenser-Lens',
    ),
    comp.Deflector(
        z=0.2,
        name='Deflector',
    ),
    comp.Lens(
        z=0.4,
        z1=-0.1,
        z2=0.4,
        f=0.07,
        name='Objective-Lens',
        aber_coeffs=coeffs,
    ),
    comp.Lens(
        z=0.7,
        z1=-0.4,
        z2=0.7,
        f=0.2,
        name='Stigmator',
    ),
    comp.Detector(
        z=1.0,
        name='Detector',
        shape=det_shape,
        pixel_size=pixel_size,
        interference=None,
    )
)

model = Model(components, backend='gpu')

# Run Model Once
rays = tuple(model.run_iter(num_rays=n_rays, random=False))
image = model.detector.get_image(rays[-1])

# Run Model Again With GUI
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
