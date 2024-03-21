import sys
from temgymbasic.model import (
    Model,
)
from temgymbasic import components as comp

from PySide6.QtWidgets import QApplication
from temgymbasic.gui import TemGymWindow

components = (
    comp.ParallelBeam(
        z=0.0,
        radius=0.01,
    ),
    comp.Biprism(
        z=0.5,
        offset=0.0,
        rotation=0.0,
        deflection=0.1,
    ),
    comp.Detector(
        z=1.,
        pixel_size=0.01,
        shape=(128, 128),
    ),
)

model = Model(components)
AppWindow = QApplication(sys.argv)
viewer = TemGymWindow(model)
viewer.show()
AppWindow.exec()
