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

wo = 0.1
theta_x = 0
theta_y = 0
deg_yx = np.deg2rad((theta_y, theta_x))
tilt_yx = np.tan(deg_yx)

radius = 0.2
lens_dist = 10
prop_dist = 5

z_o = -lens_dist
z_i = lens_dist + prop_dist
focal = 3

f = (1 / z_i - 1 / z_o) ** -1

centre_yx = [0.0, 0.0]
coeffs = [0., 0., 0., 0., 0.]
components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        radius=radius,
        wo=wo,
        tilt_yx=tilt_yx,
        semi_angle=0.0
    ),
    comp.Lens(
        z=lens_dist,
        z1=-lens_dist,
        z2=prop_dist,
        f=focal,
        aber_coeffs=coeffs
    ),
    comp.Detector(
        z=lens_dist + prop_dist,
        pixel_size=pixel_size,
        shape=det_shape,
        interference='gauss'
    ),
)

model = Model(components, backend='gpu')
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
