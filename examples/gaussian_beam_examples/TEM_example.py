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
coeffs = comp.LensAberrations(0, 0, 0., 0., 0.)

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=voltage,
        radius=10e-6,
        wo=50e-6,
        tilt_yx=tilt_yx,
        semi_angle=0.0005,
    ),
    comp.DoubleDeflector(
        first=comp.Deflector(z=0.15, name='Upper OBJ Deflector'),
        second=comp.Deflector(z=0.2, name='Lower OBJ Deflector'),
    ),
    comp.Lens(
        z=0.3,
        z1=-0.05,
        z2=0.15,
        aber_coeffs=coeffs,
        name='Objective Lens'
    ),
    comp.DoubleDeflector(
        first=comp.Deflector(z=0.45, name='Upper Image Deflector'),
        second=comp.Deflector(z=0.5, name='Lower Image Deflector'),
    ),
    comp.ProjectorLensSystem(
        first=comp.Lens(z=0.6, z1=-0.15, z2=0.75),
        second=comp.Lens(z=1.0, z1=-0.2, z2=0.5),
        name='Projector Lens System',
    ),
    comp.AccumulatingDetector(
        z=1.5,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=1,
        interference='gauss',
    ),
)


components[0].random = False
model = Model(components, backend='cpu')
# rays = tuple(model.run_iter(num_rays=1000, random=False))
# image = model.detector.get_image(rays[-1])
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
