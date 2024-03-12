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

from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem


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
):
    if decimals > 0:
        slider = QLabeledDoubleSlider(QtCore.Qt.Orientation.Horizontal)
    else:
        slider = QLabeledSlider(QtCore.Qt.Orientation.Horizontal)
    slider_config(slider, value, vmin, vmax, tick_interval)

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
