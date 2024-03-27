import sys
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
import matplotlib.pyplot as plt
import numpy as np

wavelength = 0.001
deflection = -0.1
a = 0.5
b = 0.5
d = 2*a*deflection
spacing = ((a+b)/abs(d))*wavelength
print(spacing)
print(1/spacing)
det_shape = (1, 101)
pixel_size = 0.001
components = (
    comp.XPointBeam(
        z=0.0,
        wavelength=wavelength,
        semi_angle=0.1,
        random=False,
    ),
    comp.Biprism(
        z=a,
        offset=0.0,
        rotation=0.0,
        deflection=deflection,
    ),
    # comp.Biprism(
    #     z=0.75,
    #     offset=0.0,
    #     rotation=np.pi/2,
    #     deflection=deflection,
    # ),
    comp.Detector(
        z=a+b,
        pixel_size=pixel_size,
        shape=det_shape,
    ),
)

model = Model(components)
rays = tuple(model.run_iter(num_rays=2**20))
image = model.detector.get_image_intensity(rays[-1])
det_x = np.linspace(-det_shape[1]//2*pixel_size, det_shape[1]//2*pixel_size, det_shape[1])

plt.axis('equal')
plt.figure()
plt.imshow(np.abs(image)**2/np.max(np.abs(image)**2))
plt.xscale

AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()

plt.show()
