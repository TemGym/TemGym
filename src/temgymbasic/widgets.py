from typing import Optional

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QSlider,
    QLabel,
    QHBoxLayout,
    QLayout,
    QWidget,
    QVBoxLayout,
    QLineEdit,
)
from PySide6.QtGui import (
    QIntValidator,
)

from superqt import QLabeledDoubleSlider, QLabeledSlider


def slider_config(slider: QSlider, value: int, vmin: int, vmax: int):
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    vmin, vmax = sorted((vmin, vmax))
    slider.setRange(vmin, max(vmax, vmin + 1e-5))
    slider.setValue(value)


def labelled_slider(
    value: int,
    vmin: int,
    vmax: int,
    name: Optional[str] = None,
    prefix: str = '',
    insert_into: Optional[QLayout] = None,
    spacing: int = 15,
    decimals: int = 0,
):
    if decimals > 0:
        slider = QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
    else:
        slider = QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
    slider_config(slider, value, vmin, vmax)

    if isinstance(insert_into, QHBoxLayout):
        hbox = insert_into
    else:
        hbox = QHBoxLayout()

    if name is not None:
        slider_namelabel = QLabel(name)
        hbox.addWidget(slider_namelabel)

    hbox.addWidget(slider)
    # hbox.addSpacing(spacing)
    # hbox.addWidget(slider_valuelabel)

    if insert_into is not None and not isinstance(insert_into, QHBoxLayout):
        insert_into.addLayout(hbox)

    return slider, hbox


class LabelledIntField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText(title)
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value is not None:
            self.lineEdit.setText(str(initial_value))
        layout.addWidget(self.lineEdit)
        layout.addStretch()

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)

    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    def getValue(self, default: int = 1):
        try:
            return int(self.lineEdit.text())
        except ValueError:
            return default

    def insert_into(self, layout):
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        return self
