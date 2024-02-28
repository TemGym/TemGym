from typing import List, Iterable, TYPE_CHECKING

import PyQt5
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSlider,
    QLabel,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QLineEdit,
    QComboBox,
)
from PyQt5.QtGui import QDoubleValidator
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import numpy as np

from . import shapes as comp_geom
from .utils import as_gl_lines

if TYPE_CHECKING:
    from .model import Model
    from . import components as comp


LABEL_RADIUS = 0.3
Z_ORIENT = -1


class ComponentGUIWrapper:
    def __init__(self, component: 'comp.Component'):
        self.component = component
        self.box = QGroupBox(component.name)
        self.table = QGroupBox(component.name)

    def get_label(self) -> gl.GLTextItem:
        return gl.GLTextItem(
            pos=np.array([
                -LABEL_RADIUS,
                LABEL_RADIUS,
                Z_ORIENT * self.component.z
            ]),
            text=self.component.name,
            color='w',
        )

    def get_geom(self) -> Iterable[gl.GLLinePlotItem]:
        raise NotImplementedError()


class TemGymWindow(QMainWindow):
    '''
    Create the UI Window
    '''
    def __init__(self, model: 'Model'):

        '''Init important parameters

        Parameters
        ----------
        model : class
            Microscope model
        '''
        super().__init__()
        self.model = model
        self.model_gui = ModelGUI()

        self.gui_components: List[ComponentGUIWrapper] = []
        # Loop through all components, and display the GUI for each
        for component in self.model.components:
            gui_component_c = component.gui_wrapper()
            if gui_component_c is None:
                continue
            self.gui_components.append(gui_component_c(component))

        # Set some main window's properties
        self.setWindowTitle("TemGymBasic")
        self.resize(1600, 1200)

        # Create Docks
        self.tem_dock = Dock("3D View")
        self.detector_dock = Dock("Detector", size=(5, 5))
        self.gui_dock = Dock("GUI", size=(10, 3))
        self.table_dock = Dock("Parameter Table", size=(5, 5))

        self.centralWidget = DockArea()
        self.setCentralWidget(self.centralWidget)
        self.centralWidget.addDock(self.tem_dock, "left")
        self.centralWidget.addDock(self.table_dock, "bottom", self.tem_dock)
        self.centralWidget.addDock(self.gui_dock, "right")
        self.centralWidget.addDock(self.detector_dock, "above", self.table_dock)

        # Create the display and the buttons
        self.create3DDisplay()
        self.createDetectorDisplay()
        self.createGUI()

        # Draw rays and det image
        self.update_rays()

    def create3DDisplay(self):
        '''Create the 3D Display
        '''
        # # create detector
        # scale = 1.  # self.model.detector_size/2
        # vertices = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0],
        #                     [1, -1, 0], [1, 1, 0]]) * scale

        # self.detector_outline = gl.GLLinePlotItem(pos=vertices, color="w", mode='line_strip')

        # Create the 3D TEM Widnow, and plot the components in 3D
        self.tem_window = gl.GLViewWidget()

        # Define Camera Parameters
        self.initial_camera_params = {'center': PyQt5.QtGui.QVector3D(0.5, 0.5, 0.0),
                                 'fov': 25, 'azimuth': -45.0, 'distance': 10, 'elevation': 25.0, }

        self.x_camera_params = {'center': PyQt5.QtGui.QVector3D(0.0, 0.0, 0.5),
                           'fov': 7e-07, 'azimuth': 90.0, 'distance': 143358760, 'elevation': 0.0}

        self.y_camera_params = {'center': PyQt5.QtGui.QVector3D(0.0, 0.0, 0.5),
                           'fov': 7e-07, 'azimuth': 0, 'distance': 143358760, 'elevation': 0.0}

        # Make an axis and addit to the 3D window. Also set up the ray geometry placeholder
        # and detector outline.
        # axis = gl.GLAxisItem()
        # self.tem_window.addItem(axis)

        pos = np.empty((4, 3))
        size = np.empty(4)
        color = np.empty((4, 4))

        pos[0] = (1, 0, 0)
        size[0] = 0.1
        color[0] = (1.0, 0.0, 0.0, 0.5)  # x
        pos[1] = (0, 1, 0)
        size[1] = 0.1
        color[1] = (0.0, 1.0, 0.0, 0.5)  # y
        pos[2] = (0, 0, 1)
        size[2] = 0.1
        color[2] = (0.0, 0.0, 1.0, 0.5)  # z
        pos[3] = (0.125, 0.125, 0.5)
        size[3] = 0.1
        color[3] = (1.0, 1.0, 1.0, 0.5)  # z

        # sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        # self.tem_window.addItem(sp1)

        self.tem_window.setBackgroundColor((150, 150, 150, 255))
        self.tem_window.setCameraPosition(distance=5)
        self.ray_geometry = gl.GLLinePlotItem(mode='lines', width=2)
        self.tem_window.addItem(self.ray_geometry)
        self.tem_window.setCameraParams(**self.initial_camera_params)

        # Loop through all of the model components, and add their geometry to the TEM window.
        for component in self.gui_components:
            for geometry in component.get_geom():
                self.tem_window.addItem(geometry)
            self.tem_window.addItem(component.get_label())

        # Add the ray geometry GLLinePlotItem to the list of geometries for that window
        self.tem_window.addItem(self.ray_geometry)

        # Add the window to the dock
        self.tem_dock.addWidget(self.tem_window)

    def createDetectorDisplay(self):
        '''Create the detector display
        '''
        # Create the detector window, which shows where rays land at the bottom
        self.detector_window = pg.GraphicsLayoutWidget()
        self.detector_window.setAspectLocked(1.0)
        self.spot_img = pg.ImageItem(border="b")
        v2 = self.detector_window.addViewBox()
        v2.setAspectLocked()

        # Invert coordinate system so spot moves up when it should
        v2.invertY()
        v2.addItem(self.spot_img)

        self.detector_dock.addWidget(self.detector_window)

    def createGUI(self):
        '''Create the gui display
        '''
        # Create the window which houses the GUI
        scroll = QScrollArea()
        scroll.setWidgetResizable(1)
        content = QWidget()
        scroll.setWidget(content)
        self.gui_layout = QVBoxLayout(content)

        self.gui_dock.addWidget(scroll, 1, 0)

        self.gui_layout.addWidget(self.model_gui.box, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(1)
        content = QWidget()
        scroll.setWidget(content)
        self.table_layout = QVBoxLayout(content)
        self.table_dock.addWidget(scroll, 1, 0)

        for idx, gui_component in enumerate(self.gui_components):
            self.gui_layout.addWidget(gui_component.box, idx)
            self.table_layout.addWidget(gui_component.table, 0)

    def update_rays(self):
        all_rays = tuple(self.model.run_iter(64))
        vertices = as_gl_lines(all_rays)
        vertices[:, 2] *= Z_ORIENT
        self.ray_geometry.setData(
            pos=vertices,
            color=(0, 0.8, 0, 0.05),
        )

        if self.model.detector is not None:
            image = self.model.detector.get_image(all_rays[-1])
            self.spot_img.setImage(image)


class ModelGUI():
    '''Overall GUI of the model
    '''
    def __init__(self):
        '''

        Parameters
        ----------
        num_rays : int
            Number of rays in the model
        beam_type : str
            Type of initial beam: Axial, paralell of point.
        gun_beam_semi_angle : float
            Semi angle of the beam
        beam_tilt_x : float
            Initial x tilt of the beam in radians
        beam_tilt_y : float
            Initial y tilt of the beam in radians
        '''
        self.box = QGroupBox('Model Settings')

        vbox = QVBoxLayout()
        vbox.addStretch()

        self.beamSelect = QComboBox()
        self.beamSelect.addItem("Axial Beam")
        self.beamSelect.addItem("Point Beam")
        self.beamSelect.addItem("Paralell Beam")
        self.beamSelectLabel = QLabel("Beam type")

        hbox = QHBoxLayout()
        hbox.addWidget(self.beamSelectLabel)
        hbox.addWidget(self.beamSelect)
        vbox.addLayout(hbox)

        self.view_label = QLabel('Set Camera View')
        self.init_button = QPushButton('Initial View')
        self.x_button = QPushButton('X View')
        self.y_button = QPushButton('Y View')

        hbox_label = QHBoxLayout()
        hbox_label.addWidget(self.view_label)
        hbox_push_buttons = QHBoxLayout()
        hbox_push_buttons.addWidget(self.init_button)
        hbox_push_buttons.addSpacing(15)
        hbox_push_buttons.addWidget(self.x_button)
        hbox_push_buttons.addSpacing(15)
        hbox_push_buttons.addWidget(self.y_button)
        vbox.addLayout(hbox_label)
        vbox.addLayout(hbox_push_buttons)
        self.box.setLayout(vbox)


class LensGUI(ComponentGUIWrapper):
    def __init__(self, lens: 'comp.Lens'):
        '''GUI for the Lens component
        ----------
        name : str
            Name of component
        f : float
            Focal length
        '''
        super().__init__(component=lens)

        self.fslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.fslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fslider.setMinimum(-10)
        self.fslider.setMaximum(10)
        self.fslider.setValue(1)
        self.fslider.setTickPosition(QSlider.TicksBelow)

        self.flineedit = QLineEdit(f"{lens.f:.4f}")
        self.flineeditstep = QLineEdit(f"{0.1:.4f}")

        self.fwobblefreqlineedit = QLineEdit(f"{1:.4f}")
        self.fwobbleamplineedit = QLineEdit(f"{0.5:.4f}")

        qdoublevalidator = QDoubleValidator()
        self.flineedit.setValidator(qdoublevalidator)
        self.flineeditstep.setValidator(qdoublevalidator)
        self.fwobblefreqlineedit.setValidator(qdoublevalidator)
        self.fwobbleamplineedit.setValidator(qdoublevalidator)

        self.fwobble = QCheckBox('Wobble Lens Current')

        hbox = QHBoxLayout()
        hbox_lineedit = QHBoxLayout()
        hbox_lineedit.addWidget(QLabel('Focal Length = '))
        hbox_lineedit.addWidget(self.flineedit)
        hbox_lineedit.addWidget(QLabel('Slider Step = '))
        hbox_lineedit.addWidget(self.flineeditstep)
        hbox_slider = QHBoxLayout()
        hbox_slider.addWidget(self.fslider)
        hbox_wobble = QHBoxLayout()
        hbox_wobble.addWidget(self.fwobble)
        hbox_wobble.addWidget(QLabel('Wobble Frequency'))
        hbox_wobble.addWidget(self.fwobblefreqlineedit)
        hbox_wobble.addWidget(QLabel('Wobble Amplitude'))
        hbox_wobble.addWidget(self.fwobbleamplineedit)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_lineedit)
        vbox.addLayout(hbox_slider)
        vbox.addLayout(hbox_wobble)
        vbox.addStretch()

        self.box.setLayout(vbox)

        self.flabel_table = QLabel('Focal Length = ' + f"{lens.f:.2f}")
        self.flabel_table.setMinimumWidth(80)
        hbox = QHBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(self.flabel_table)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        self.table.setLayout(vbox)

    def get_geom(self):
        vertices = comp_geom.lens(
            0.2,
            Z_ORIENT * self.component.z,
            64,
        )
        return [
            gl.GLLinePlotItem(
                pos=vertices.T,
                color="white",
                width=5
            )
        ]


class ParallelBeamGUI(ComponentGUIWrapper):
    def __init__(self, beam: 'comp.ParallelBeam'):
        super().__init__(beam)

        num_rays = 64
        self.box = QGroupBox('Beam Settings')
        self.rayslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.rayslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.rayslider.setMinimum(4)
        self.rayslider.setMaximum(15)
        self.rayslider.setValue(int(np.log2(num_rays)))
        self.rayslider.setTickPosition(QSlider.TicksBelow)

        self.raylabel = QLabel(str(num_rays))
        self.raylabel.setMinimumWidth(80)
        self.modelraylabel = QLabel('Number of Rays')

        self.anglelabel = QLabel('Beam Tilt Offset')

        beam_tilt_y, beam_tilt_x = 0., 0.
        self.xanglelabel = QLabel(
            'Beam Tilt X (Radians) = ' + f"{beam_tilt_x:.3f}")
        self.xangleslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.xangleslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.xangleslider.setMinimum(-200)
        self.xangleslider.setMaximum(200)
        self.xangleslider.setValue(0)
        self.xangleslider.setTickPosition(QSlider.TicksBelow)

        self.yanglelabel = QLabel(
            'Beam Tilt Y (Radians) = ' + f"{beam_tilt_y:.3f}")
        self.yangleslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.yangleslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.yangleslider.setMinimum(-200)
        self.yangleslider.setMaximum(200)
        self.yangleslider.setValue(0)
        self.yangleslider.setTickPosition(QSlider.TicksBelow)

        self.beamwidthslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.beamwidthslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.beamwidthslider.setMinimum(-100)
        self.beamwidthslider.setMaximum(99)
        self.beamwidthslider.setValue(1)
        self.beamwidthslider.setTickPosition(QSlider.TicksBelow)

        self.beamwidthlabel = QLabel('0')
        self.beamwidthlabel.setMinimumWidth(80)
        self.modelbeamwidthlabel = QLabel('Paralell Beam Width')

        vbox = QVBoxLayout()
        vbox.addStretch()

        hbox = QHBoxLayout()
        hbox.addWidget(self.rayslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.raylabel)
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.modelraylabel)
        hbox_labels.addStretch()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.beamwidthslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.beamwidthlabel)
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.modelbeamwidthlabel)
        hbox_labels.addStretch()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.anglelabel)
        hbox.addWidget(self.xangleslider)
        hbox.addWidget(self.xanglelabel)
        hbox.addWidget(self.yangleslider)
        hbox.addWidget(self.yanglelabel)
        hbox_labels.addStretch()
        vbox.addLayout(hbox_labels)
        vbox.addLayout(hbox)

        self.box.setLayout(vbox)

    def get_geom(self):
        vertices = comp_geom.lens(
            self.component.radius,
            Z_ORIENT * self.component.z,
            64,
        )
        return [
            gl.GLLinePlotItem(
                pos=vertices.T,
                color="green",
                width=2,
            )
        ]


class DetectorGUI(ComponentGUIWrapper):
    def get_geom(self):
        vertices = comp_geom.square(
            w=0.5,
            x=0.,
            y=0.,
            z=Z_ORIENT * self.component.z,
        )
        mesh = gl.GLMeshItem(
            vertexes=vertices,
            smooth=True,
            drawEdges=True,
            drawFaces=False,
        )
        mesh.setColor("white")
        return [mesh]
