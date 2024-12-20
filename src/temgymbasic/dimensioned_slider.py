from math import log10, floor
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from qtpy.QtWidgets import (
    QHBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QFrame,
)
from qtpy import QtCore
from qtpy.QtGui import QPainter

from superqt import QLabeledDoubleSlider, QDoubleSlider

if TYPE_CHECKING:
    import pint


def bounds(x, pow10):
    lb = floor(x / pow10) * pow10
    return lb, lb + pow10


class QDoubleSliderTicked(QDoubleSlider):
    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        style = self.style()

        st_slider = QStyleOptionSlider()
        st_slider.initFrom(self)
        st_slider.orientation = self.orientation()
        available = style.pixelMetric(QStyle.PM_SliderSpaceAvailable, st_slider, self)

        s_min = self.minimum()
        s_max = self.maximum()

        for idx, val in enumerate((s_min, s_max)):
            x_loc = QStyle.sliderPositionFromValue(
                s_min,
                s_max,
                val,
                available,
            )
            # left bound of the text = center - half of text width + L_margin
            v_str = f"{val:.0f}"
            rect = p.drawText(
                QtCore.QRect(),
                QtCore.Qt.TextFlag.TextDontClip | QtCore.Qt.TextFlag.TextDontPrint,
                v_str,
            )
            left = x_loc
            if idx == 1:
                left -= (rect.width() / 2)
            left += 2
            middle = (self.rect().bottom() + self.rect().top()) // 2
            p.drawText(
                QtCore.QRect(left, middle + 4, 10, 10),
                QtCore.Qt.TextFlag.TextDontClip,
                v_str,
            )


class QDimensionedFloatSlider(QLabeledDoubleSlider):
    _slider_class = QDoubleSliderTicked

    def __init__(
        self,
        quant_name: str,
        quantity: 'pint.Quantity',
        *args,
        negative: bool = False,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        slider_layout = QHBoxLayout()
        self._slider.setLayout(slider_layout)
        self._slider.setMinimumHeight(35)
        slider_layout.setContentsMargins(0, 0, 0, 10)

        self.quantity = quantity.to_compact()
        self._fullrange = (-1000 if negative else 0, 1000)
        self.setRange(*self._fullrange)
        self.setDecimals(1)
        self.setValue(self.current_q_val())
        self.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.setTickInterval(200 if negative else 100)
        self._slider_name = QLabel(parent=self)
        self._slider_name.setText(quant_name)
        self._slider_unit = QLabel(parent=self)
        self._slider_unit.setText(self.current_q_str())
        self._slider_unit.setFixedWidth(30)
        marg = (0, 0, 0, 0)
        self._down_button = QPushButton("/10")
        self._down_button.setFlat(True)
        self._down_button.setFixedWidth(22)
        self._down_button.clicked.connect(self._on_scale_down)
        self._down_button.setContentsMargins(*marg)
        self._up_button = QPushButton("x10")
        self._up_button.setFlat(True)
        self._up_button.setFixedWidth(22)
        self._up_button.clicked.connect(self._on_scale_up)
        self._up_button.setContentsMargins(*marg)
        self._zoom_out_button = QPushButton("-")
        self._zoom_out_button.setFlat(True)
        self._zoom_out_button.setFixedWidth(10)
        self._zoom_out_button.clicked.connect(self._on_zoom_out)
        self._zoom_out_button.setContentsMargins(*marg)
        self._zoom_in_button = QPushButton("+")
        self._zoom_in_button.setFlat(True)
        self._zoom_in_button.setFixedWidth(10)
        self._zoom_in_button.clicked.connect(self._on_zoom_in)
        self._zoom_in_button.setContentsMargins(*marg)
        self._zero_button = None
        if negative:
            self._zero_button = QPushButton("0")
            self._zero_button.setFlat(True)
            self._zero_button.setFixedWidth(10)
            self._zero_button.clicked.connect(self._to_zero)
            self._zero_button.setContentsMargins(*marg)
        self.setOrientation(self._slider.orientation())

    def setOrientation(self, orientation: Qt.Orientation) -> None:
        """Set orientation, value will be 'horizontal' or 'vertical'."""
        self._slider.setOrientation(orientation)
        marg = (0, 0, 0, 0)
        layout = QHBoxLayout()  # type: ignore
        try:
            layout.addWidget(self._slider_name)
        except AttributeError:
            pass
        layout.addWidget(self._slider)
        layout.addWidget(self._label)
        self._label.setContentsMargins(0, 0, 0, 0)
        self._label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        try:
            layout.addWidget(self._slider_unit)
            self._slider_unit.setContentsMargins(0, 0, 0, 0)
            self._slider_unit.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            line = QLabel()
            line.setFrameStyle(QFrame.Shape.VLine)
            line.setLineWidth(1)
            line.setFixedHeight(15)
            line.setFixedWidth(2)
            line.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(line)
            if self._zero_button is not None:
                layout.addWidget(self._zero_button)
            layout.addWidget(self._zoom_out_button)
            layout.addWidget(self._zoom_in_button)
            layout.addWidget(self._down_button)
            layout.addWidget(self._up_button)
        except AttributeError:
            pass

        old_layout = self.layout()
        if old_layout is not None:
            QWidget().setLayout(old_layout)

        layout.setContentsMargins(*marg)
        self.setLayout(layout)

    def current_q_str(self):
        return f"{self.quantity.u:~}"

    def current_q_val(self):
        return round(self.quantity.m, 1)

    def _on_scale(self, scale: float, clip) -> None:
        val = self.value()
        sign = 1 if val > 0 else -1
        val = abs(val)
        if val == 0:
            if scale > 1:
                quantity = (1000 * self.quantity.u).to_compact()
            else:
                quantity = (0.001 * self.quantity.u).to_compact()
            self.quantity = 0 * quantity.u
        else:
            self.quantity = ((val * scale) * self.quantity.u).to_compact()
        self._slider_unit.setText(self.current_q_str())
        self._setRange(*self._fullrange)
        self.setValue(sign * clip(self.current_q_val()))

    @QtCore.Slot()
    def _on_scale_up(self) -> None:
        self._on_scale(10., lambda x: min(1000., x))

    @QtCore.Slot()
    def _on_scale_down(self) -> None:
        self._on_scale(1 / 10, lambda x: max(0., x))

    @QtCore.Slot()
    def _to_zero(self) -> None:
        self.setValue(0)

    @QtCore.Slot()
    def _on_zoom(self, step: int):
        mini, maxi = self.minimum(), self.maximum()
        current_range = min(maxi - mini, 1000)
        new_range = 10 ** (floor(log10(current_range)) + step)
        new_range = max(10, min(new_range, 1000))
        current_si = (self.value() * self.quantity.u).to_compact()
        lb, ub = bounds(current_si.m, new_range)
        if lb in (-1000, 0) and ub in (0, 1000):
            lb, ub = self._fullrange
        self._setRange(lb, ub)
        self.setValue(current_si.m)

    def _setRange(self, lb, ub):
        self.setRange(lb, ub)
        self.setTickInterval((ub - lb) / 10)

    @QtCore.Slot()
    def _on_zoom_out(self):
        self._on_zoom(1)

    @QtCore.Slot()
    def _on_zoom_in(self):
        self._on_zoom(-1)


if __name__ == "__main__":
    import pint  # noqa
    u = pint.UnitRegistry()
    app = QApplication([])

    slider = QDimensionedFloatSlider(
        "Focal length",
        10 * u.micrometers,
        negative=True,
    )
    slider._slider.setStyleSheet(
        R"""
QSlider::groove:horizontal {
    border: 1px solid #999999;
    height: 10px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #66b2ff, stop:1 #cce5ff);
    margin: 4px 0;
}

QSlider::groove:horizontal:disabled {
    border: 1px solid #999999;
    height: 10px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f4f4f4, stop:1 #8f8f8f);
    margin: 4px 0;
}

QSlider::handle:horizontal:disabled {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f4f4f4, stop:1 #8f8f8f);
    border: 1px solid #5c5c5c;
    width: 12px;
    margin: -2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
    border: 1px solid #5c5c5c;
    width: 12px;
    margin: -2px 0;
    border-radius: 3px;
}
""")
    slider.show()

    app.exec_()
