from typing import Optional, Union

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QSlider,
    QLabel,
    QHBoxLayout,
    QLayout,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
)
from PySide6.QtGui import (
    QIntValidator,
)

from superqt import QLabeledDoubleSlider, QLabeledSlider
import OpenGL.GL as gl
from OpenGL.GL import (
    GL_PROXY_TEXTURE_2D,
    GL_TEXTURE_WIDTH,
    GL_RGBA,
    GL_UNSIGNED_BYTE,
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_LINEAR,
    GL_NEAREST,
    GL_TEXTURE_WRAP_S,
    GL_CLAMP_TO_BORDER,
    GL_TEXTURE_WRAP_T,
    GL_TEXTURE_MAG_FILTER,
    GL_QUADS,
)
import numpy as np

from pyqtgraph.dockarea import DockLabel
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


class MyDockLabel(DockLabel):
    def updateStyle(self):
        r = '3px'
        if self.dim:
            fg = '#aaa'
            bg = '#44a'
            border = '#339'
        else:
            fg = '#fff'
            bg = '#333'
            border = '#444'

        if self.orientation == 'vertical':
            self.vStyle = """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: 0px;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: %s;
                border-width: 0px;
                border-right: 2px solid %s;
                padding-top: 3px;
                padding-bottom: 3px;
                font-size: %s;
            }""" % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.vStyle)
        else:
            self.hStyle = """DockLabel {
                background-color : %s;
                color : %s;
                border-top-right-radius: %s;
                border-top-left-radius: %s;
                border-bottom-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-width: 0px;
                border-bottom: 2px solid %s;
                padding-left: 3px;
                padding-right: 3px;
                font-size: %s;
            }""" % (bg, fg, r, r, border, self.fontSize)
            self.setStyleSheet(self.hStyle)


def slider_config(slider: QSlider, value: int, vmin: int, vmax: int, tick_interval: Optional[int]):
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    vmin, vmax = sorted((vmin, vmax))
    slider.setRange(vmin, max(vmax, vmin + 1e-5))
    slider.setValue(value)
    if tick_interval is not None:
        slider.setTickInterval(tick_interval)


def labelled_slider(
    value: int,
    vmin: int,
    vmax: int,
    name: Optional[str] = None,
    insert_into: Optional[QLayout] = None,
    decimals: int = 0,
    tick_interval: Optional[int] = None,
    reset_to: Optional[Union[int, bool]] = True,
):
    if decimals > 0:
        slider = QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
    else:
        slider = QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
    slider_config(slider, value, vmin, vmax, tick_interval)
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
"""
    )

    if isinstance(insert_into, QHBoxLayout):
        hbox = insert_into
    else:
        hbox = QVBoxLayout()

    if name is not None:
        if reset_to is not None:

            @QtCore.Slot()
            def reset():
                if reset_to is True:
                    slider.setValue(value)
                else:
                    slider.setValue(reset_to)

            button = QPushButton(name)
            button.setFlat(True)
            button.setMaximumWidth(175)
            button.setStyleSheet("text-align:left;")
            # button.setSizePolicy(
            #     QSizePolicy.Policy.Minimum,
            #     QSizePolicy.Policy.Minimum,
            # )
            button.clicked.connect(reset)
            hbox.addWidget(button)
        else:
            slider_namelabel = QLabel(name)
            hbox.addWidget(slider_namelabel)

    hbox.addWidget(slider)

    if insert_into is not None and not isinstance(insert_into, QHBoxLayout):
        insert_into.addLayout(hbox)

    return slider, hbox


class LabelledIntField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText(title)
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value is not None:
            self.lineEdit.setText(str(initial_value))
        layout.addWidget(self.lineEdit)

    def getValue(self, default: int = 1):
        try:
            return int(self.lineEdit.text())
        except ValueError:
            return default

    def insert_into(self, layout):
        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        return self


class GLImageItem(GLGraphicsItem):
    def __init__(self, vertices, data, smooth=False, glOptions='translucent', parentItem=None):
        self.smooth = smooth
        self._needUpdate = False
        super().__init__(parentItem=parentItem)
        self.setData(data)
        self.setVertices(vertices)
        self.setGLOptions(glOptions)
        self.texture = None

    def initializeGL(self):
        if self.texture is not None:
            return
        gl.glEnable(GL_TEXTURE_2D)
        self.texture = gl.glGenTextures(1)

    def setData(self, data):
        self.data = data
        self._needUpdate = True
        self.update()

    def setVertices(self, vertices, update: bool = True):
        self.vertices = vertices
        if update:
            self.update()

    def _updateTexture(self):
        gl.glBindTexture(GL_TEXTURE_2D, self.texture)
        if self.smooth:
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        else:
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        # glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        shape = self.data.shape

        # Test texture dimensions first
        gl.glTexImage2D(
            GL_PROXY_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None
        )
        if gl.glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH) == 0:
            raise Exception(
                "OpenGL failed to create 2D texture "
                "(%dx%d); too large for this hardware." % shape[:2]
            )

        data = np.ascontiguousarray(self.data.transpose((1, 0, 2)))
        gl.glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, shape[0], shape[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, data
        )
        gl.glDisable(GL_TEXTURE_2D)

    def paint(self):
        if self._needUpdate:
            self._updateTexture()
            self._needUpdate = False
        gl.glEnable(GL_TEXTURE_2D)
        gl.glBindTexture(GL_TEXTURE_2D, self.texture)

        self.setupGLState()

        gl.glColor4f(1, 1, 1, 1)

        gl.glBegin(GL_QUADS)
        gl.glTexCoord2f(0, 0)
        gl.glVertex3f(*self.vertices[0])
        gl.glTexCoord2f(1, 0)
        gl.glVertex3f(*self.vertices[1])
        gl.glTexCoord2f(1, 1)
        gl.glVertex3f(*self.vertices[2])
        gl.glTexCoord2f(0, 1)
        gl.glVertex3f(*self.vertices[3])
        gl.glEnd()
        gl.glDisable(GL_TEXTURE_2D)
