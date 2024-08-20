import sys
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow
# import matplotlib.pyplot as plt
import numpy as np
from temgymbasic.utils import calculate_phi_0

wavelength = 0.001
deflection = -0.1
a = 0.5
b = 0.5
d = 2*a*deflection
spacing = ((a+b)/abs(d))*wavelength
print(spacing)
print(1/spacing)
det_shape = (101, 101)
pixel_size = 0.001

components = (
    comp.GaussBeam(
        z=0.0,
        voltage=calculate_phi_0(wavelength),
        radius=0.1,
        wo=0.2
    ),
    comp.Biprism(
        z=a,
        offset=0.0,
        rotation=0.0,
        deflection=deflection,
    ),
    # comp.Biprism(
    #     z=0.70,
    #     offset=0.0,
    #     rotation=np.pi/2,
    #     deflection=deflection,
    # ),
    comp.AccumulatingDetector(
        z=a+b,
        pixel_size=pixel_size,
        shape=det_shape,
        buffer_length=64,
    ),
)

model = Model(components)
rays = tuple(model.run_iter(num_rays=1000))

end_rays = rays[-1].data
image = model.detector.get_image(rays[-1])
det_x = np.linspace(-det_shape[1]//2*pixel_size, det_shape[1]//2*pixel_size, det_shape[1])

# plt.axis('equal')
# plt.figure()
# plt.imshow(np.abs(image)**2/np.max(np.abs(image)**2))
# plt.show()

AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
