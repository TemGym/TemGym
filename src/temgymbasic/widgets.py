from typing import Optional

from PySide6 import QtCore
from PySide6.QtCore import Slot, Signal
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


class DoubleSlider(QSlider):
    # create our our signal that we can connect to if necessary
    doubleValueChanged = Signal(float)

    def __init__(self, *args, decimals=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super().value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super().value()) / self._multi

    def setMinimum(self, value):
        return super().setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super().setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super().setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super().singleStep()) / self._multi

    def setValue(self, value):
        super().setValue(int(value * self._multi))


class QNumericLabel(QLabel):
    def setPrefix(self, prefix: str):
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        try:
            return self._prefix
        except AttributeError:
            return ''

    @Slot(int)
    @Slot(float)
    @Slot(complex)
    def setText(self, v):
        super().setText(f"{self.prefix}{v}")


def slider_config(slider: QSlider, value: int, vmin: int, vmax: int):
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    slider.setMinimum(vmin)
    slider.setMaximum(vmax)
    slider.setValue(value)
    slider.setTickPosition(QSlider.TicksBelow)


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
        slider = DoubleSlider(QtCore.Qt.Orientation.Horizontal, decimals=decimals)
    else:
        slider = QSlider(QtCore.Qt.Orientation.Horizontal)
    slider_config(slider, value, vmin, vmax)
    slider_valuelabel = QNumericLabel(prefix + str(slider.value()))
    slider_valuelabel.setPrefix(prefix)
    slider_valuelabel.setMinimumWidth(80)
    if decimals > 0:
        slider.doubleValueChanged.connect(slider_valuelabel.setText)
    else:
        slider.valueChanged.connect(slider_valuelabel.setText)

    if isinstance(insert_into, QHBoxLayout):
        hbox = insert_into
    else:
        hbox = QHBoxLayout()

    if name is not None:
        slider_namelabel = QLabel(name)
        hbox.addWidget(slider_namelabel)

    hbox.addWidget(slider)
    hbox.addSpacing(spacing)
    hbox.addWidget(slider_valuelabel)

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
