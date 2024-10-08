from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp
import numpy as np

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import sys

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
coeffs = comp.LensAberrations(0., 0., 0., 0., 0.)

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=voltage,
        radius=50e-6,
        wo=200e-6,
        tilt_yx=tilt_yx,
        semi_angle=0.0005,
    ),
    comp.DoubleDeflector(
        first=comp.Deflector(z=0.15, name='Upper OBJ Deflector'),
        second=comp.Deflector(z=0.2, name='Lower OBJ Deflector'),
    ),
    comp.Lens(
        z=0.4,
        z1=-0.05,
        z2=0.15,
        aber_coeffs=coeffs,
        name='Objective Lens'
    ),
    comp.DoubleDeflector(
        first=comp.Deflector(z=0.45, name='Upper Image Deflector'),
        second=comp.Deflector(z=0.5, name='Lower Image Deflector'),
    ),
    comp.Lens(
        z=0.7,
        z1=-0.15,
        f=0.12,
        name='Projector Lens 1',
    ),
    comp.Lens(
        z=1.0,
        z1=-0.1,
        f=0.07,
        name='Projector Lens 2',
    ),
    comp.AccumulatingDetector(
        z=1.5,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=1,
        interference='gauss',
    ),
)


components[0].random = True
model = Model(components, backend='gpu')
# rays = tuple(model.run_iter(num_rays=n_rays, random = False))
# image = model.detector.get_image(rays[-1])
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
