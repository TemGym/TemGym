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
        z=0.25,
        offset=0.0,
        rotation=np.pi/2,
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
        pixel_size=0.0005,
        shape=(200, 200),
    ),
)

model = Model(components)
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
