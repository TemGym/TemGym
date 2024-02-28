import sys
from .model import (
    STEMModel,
)


def main_window():
    from PySide6.QtWidgets import QApplication
    from .gui import TemGymWindow

    model = STEMModel()
    AppWindow = QApplication(sys.argv)
    viewer = TemGymWindow(model)
    viewer.show()
    AppWindow.exec()
