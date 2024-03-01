from typing import List, Iterable, TYPE_CHECKING
from typing_extensions import Self
import weakref

from PySide6.QtGui import QVector3D
from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
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
from PySide6.QtGui import (
    QDoubleValidator,
    QKeyEvent,
)
from PySide6.QtCore import Qt
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import numpy as np

from . import shapes as comp_geom
from .utils import as_gl_lines
from .widgets import labelled_slider, LabelledIntField

if TYPE_CHECKING:
    from .model import Model
    from . import components as comp


LABEL_RADIUS = 0.3
Z_ORIENT = -1


class ComponentGUIWrapper:
    def __init__(self, component: 'comp.Component', window: 'TemGymWindow'):
        self.component = component
        self.window = weakref.ref(window)
        self.box = QGroupBox(component.name)
        self.table = QGroupBox(component.name)

    def try_update(self):
        window = self.window()
        if window is not None:
            window.update_rays()

    @Slot(int)
    @Slot(float)
    @Slot(str)
    def try_update_slot(self, val):
        self.try_update()

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
            self.gui_components.append(
                gui_component_c(component, self).build()
            )
        assert isinstance(self.gui_components[0], SourceGUI)

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

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()

    def create3DDisplay(self):
        '''Create the 3D Display
        '''
        # Create the 3D TEM Widnow, and plot the components in 3D
        self.tem_window = gl.GLViewWidget()
        self.tem_window.setBackgroundColor((150, 150, 150, 255))

        # Get the model mean height to centre the camera origin
        mean_z = sum(c.z for c in self.model.components) / len(self.model.components)
        mean_z *= Z_ORIENT

        # Define Camera Parameters
        self.initial_camera_params = {
            'center': QVector3D(0.0, 0.0, mean_z),
            'fov': 25,
            'azimuth': -45.0,
            'distance': 5,
            'elevation': 25.0,
        }

        self.x_camera_params = {
            'center': QVector3D(0.0, 0.0, mean_z),
            'fov': 25,
            'azimuth': 90.0,
            'distance': 5,
            'elevation': 0.0,
        }

        self.y_camera_params = {
            'center': QVector3D(0.0, 0.0, mean_z),
            'fov': 25,
            'azimuth': 0,
            'distance': 5,
            'elevation': 0.0,
        }
        self.tem_window.setCameraParams(**self.initial_camera_params)

        self.ray_geometry = gl.GLLinePlotItem(
            mode='lines',
            width=2
        )
        self.tem_window.addItem(self.ray_geometry)

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

    @Slot()
    def update_rays(self):
        all_rays = tuple(self.model.run_iter(
            self.gui_components[0].num_rays
        ))
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
    @property
    def lens(self) -> 'comp.Lens':
        return self.component

    @Slot(float)
    def set_f(self, val: float):
        if abs(val) < 1e-6:
            return
        self.lens.f = val
        self.try_update()

    def build(self) -> Self:
        qdoublevalidator = QDoubleValidator()
        vbox = QVBoxLayout()

        self.fslider, _ = labelled_slider(
            self.lens.f, -5., 5., name="Focal Length", insert_into=vbox, decimals=2,
        )
        self.fslider.doubleValueChanged.connect(self.set_f)

        self.fwobblefreqlineedit = QLineEdit(f"{1:.4f}")
        self.fwobbleamplineedit = QLineEdit(f"{0.5:.4f}")
        self.fwobblefreqlineedit.setValidator(qdoublevalidator)
        self.fwobbleamplineedit.setValidator(qdoublevalidator)
        self.fwobble = QCheckBox('Wobble Lens Current')
        hbox_wobble = QHBoxLayout()
        hbox_wobble.addWidget(self.fwobble)
        hbox_wobble.addWidget(QLabel('Wobble Frequency'))
        hbox_wobble.addWidget(self.fwobblefreqlineedit)
        hbox_wobble.addWidget(QLabel('Wobble Amplitude'))
        hbox_wobble.addWidget(self.fwobbleamplineedit)
        vbox.addLayout(hbox_wobble)
        vbox.addStretch()
        self.box.setLayout(vbox)

        self.flabel_table = QLabel('Focal Length = ' + f"{self.lens.f:.2f}")
        self.flabel_table.setMinimumWidth(80)
        hbox = QHBoxLayout()
        hbox.addWidget(self.flabel_table)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        self.table.setLayout(vbox)

        return self

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


class SourceGUI(ComponentGUIWrapper):
    @property
    def num_rays(self) -> int:
        return self.rayslider.value()


class ParallelBeamGUI(SourceGUI):
    @property
    def beam(self) -> 'comp.ParallelBeam':
        return self.component

    @Slot(float)
    def set_radius(self, val):
        self.beam.radius = val
        self.try_update()

    @Slot(float)
    def set_tilt(self, val):
        self.beam.tilt_yx = (
            self.yangleslider.value(),
            self.xangleslider.value(),
        )
        self.try_update()

    def build(self) -> Self:
        num_rays = 64
        beam_tilt_y, beam_tilt_x = 0, 0
        beam_radius = 0.01

        vbox = QVBoxLayout()
        vbox.addStretch()

        self.rayslider, _ = labelled_slider(
            num_rays, 1, 512, name="Number of Rays", insert_into=vbox,
        )
        self.rayslider.valueChanged.connect(self.try_update_slot)

        common_args = dict(
            vmin=-np.pi / 4, vmax=np.pi / 4, spacing=0, decimals=2,
        )
        self.xangleslider, hbox_angles = labelled_slider(
            value=beam_tilt_x, name="Beam Tilt X (rad)", **common_args
        )
        self.yangleslider, _ = labelled_slider(
            value=beam_tilt_y, name="Beam Tilt Y (rad)", **common_args,
            insert_into=hbox_angles
        )
        vbox.addLayout(hbox_angles)
        self.xangleslider.doubleValueChanged.connect(self.set_tilt)
        self.yangleslider.doubleValueChanged.connect(self.set_tilt)

        self.beamwidthslider, _ = labelled_slider(
            beam_radius, 0.001, 0.1, name='Parallel Beam Width', insert_into=vbox, decimals=3
        )
        self.beamwidthslider.doubleValueChanged.connect(self.set_radius)

        self.box.setLayout(vbox)
        return self

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


class SampleGUI(ComponentGUIWrapper):
    def get_geom(self):
        vertices, faces = comp_geom.square(
            w=0.25,
            x=0.,
            y=0.,
            z=Z_ORIENT * self.component.z,
        )

        colors = np.ones((vertices.shape[0], 4))
        colors[..., 3] = 0.1

        mesh = gl.GLMeshItem(
            meshdata=gl.MeshData(
                vertexes=vertices,
                faces=faces,
                vertexColors=colors,
            ),
            smooth=True,
            drawEdges=False,
            drawFaces=True,
            shader='shaded',
        )
        return [mesh]


class STEMSampleGUI(SampleGUI):
    @property
    def sample(self) -> 'comp.STEMSample':
        return self.component

    def build(self) -> Self:
        self.scanpixelsslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scanpixelsslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scanpixelsslider.setMinimum(2)
        self.scanpixelsslider.setMaximum(8)
        self.scanpixelsslider.setValue(256)

        self.scanpixelslabel = QLabel('Scan pixels = ' + str(int(self.scanpixelsslider.value())))
        self.scanpixelslabel.setMinimumWidth(80)

        self.overfocuslabel = QLabel('Overfocus = Not Set')
        self.overfocuslabel.setMinimumWidth(80)

        self.cameralengthlabel = QLabel('Camera length = Not Set')
        self.cameralengthlabel.setMinimumWidth(80)

        self.semiconvlabel = QLabel('Semi conv = Not Set')
        self.semiconvlabel.setMinimumWidth(80)

        self.scanpixelsizelabel = QLabel('Scan pixel size = Not Set')
        self.scanpixelsizelabel.setMinimumWidth(80)

        vbox = QVBoxLayout()
        vbox.addStretch()

        hbox = QHBoxLayout()
        hbox.addWidget(self.scanpixelsslider)
        hbox.addSpacing(15)
        hbox.addWidget(self.scanpixelslabel)
        hbox.addSpacing(15)
        hbox.addWidget(self.overfocuslabel)
        hbox.addSpacing(15)
        hbox.addWidget(self.semiconvlabel)
        hbox.addSpacing(15)
        hbox.addWidget(self.scanpixelsizelabel)
        hbox.addSpacing(15)
        hbox.addWidget(self.cameralengthlabel)

        vbox.addLayout(hbox)

        self.FOURDSTEM_experiment_button = QPushButton('Run 4D STEM Experiment')

        hbox_push_buttons = QHBoxLayout()
        hbox_push_buttons.addWidget(self.FOURDSTEM_experiment_button)
        vbox.addLayout(hbox_push_buttons)

        self.box.setLayout(vbox)
        return self


class DoubleDeflectorGUI(ComponentGUIWrapper):
    @property
    def d_deflector(self) -> 'comp.DoubleDeflector':
        return self.component

    @Slot(float)
    def set_updefx(self, val: float):
        self.d_deflector.first.defx = val
        self.try_update()

    @Slot(float)
    def set_updefy(self, val: float):
        self.d_deflector.first.defy = val
        self.try_update()

    @Slot(float)
    def set_lowdefx(self, val: float):
        self.d_deflector.second.defx = val
        self.try_update()

    @Slot(float)
    def set_lowdefy(self, val: float):
        self.d_deflector.second.defy = val
        self.try_update()

    def build(self) -> Self:
        updefx = self.d_deflector.first.defx
        updefy = self.d_deflector.first.defy
        lowdefx = self.d_deflector.second.defx
        lowdefy = self.d_deflector.second.defy

        vbox = QVBoxLayout()

        common_args = dict(
            vmin=-0.1, vmax=0.1, spacing=0, decimals=2,
        )
        self.updefxslider, hbox = labelled_slider(
            value=updefx, name="Upper X Deflection", **common_args
        )
        self.updefyslider, _ = labelled_slider(
            value=updefy, name="Upper Y Deflection", **common_args,
            insert_into=hbox
        )
        vbox.addLayout(hbox)
        self.updefxslider.doubleValueChanged.connect(self.set_updefx)
        self.updefyslider.doubleValueChanged.connect(self.set_updefy)

        self.lowdefxslider, hbox = labelled_slider(
            value=lowdefx, name="Lower X Deflection", **common_args
        )
        self.lowdefyslider, _ = labelled_slider(
            value=lowdefy, name="Lower Y Deflection", **common_args,
            insert_into=hbox
        )
        vbox.addLayout(hbox)
        self.lowdefxslider.doubleValueChanged.connect(self.set_lowdefx)
        self.lowdefyslider.doubleValueChanged.connect(self.set_lowdefy)

        # self.xbuttonwobble = QCheckBox("Wobble Upper Deflector X")
        # self.defxwobblefreqlineedit = QLineEdit(f"{1:.4f}")
        # self.defxwobbleamplineedit = QLineEdit(f"{0.5:.4f}")
        # self.defxratiolabel = QLabel('Deflector X Response Ratio = ')
        # self.defxratiolineedit = QLineEdit(f"{0.0:.4f}")
        # self.defxratiolineeditstep = QLineEdit(f"{0.1:.4f}")

        # self.defxratiolineedit.setValidator(qdoublevalidator)
        # self.defxratiolineeditstep.setValidator(qdoublevalidator)

        # self.defxratioslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        # self.defxratioslider.setMinimum(-10)
        # self.defxratioslider.setMaximum(10)
        # self.defxratioslider.setValue(1)
        # self.defxratioslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.defxratioslider.setTickPosition(QSlider.TicksBelow)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.xbuttonwobble)
        # hbox.addWidget(QLabel('Wobble X Frequency'))
        # hbox.addWidget(self.defxwobblefreqlineedit)
        # hbox.addWidget(QLabel('Wobble X Amplitude'))
        # hbox.addWidget(self.defxwobbleamplineedit)
        # vbox.addLayout(hbox)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.defxratiolabel)
        # hbox.addWidget(self.defxratiolineedit)
        # hbox.addWidget(QLabel('Def Ratio X Response Slider Step = '))
        # hbox.addWidget(self.defxratiolineeditstep)
        # vbox.addLayout(hbox)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.defxratioslider)
        # vbox.addLayout(hbox)

        # self.ybuttonwobble = QCheckBox("Wobble Upper Deflector Y")
        # self.defywobblefreqlineedit = QLineEdit(f"{1:.4f}")
        # self.defywobbleamplineedit = QLineEdit(f"{0.5:.4f}")
        # self.defyratiolabel = QLabel('Deflector Y Response Ratio = ')
        # self.defyratiolineedit = QLineEdit(f"{0.0:.4f}")
        # self.defyratiolineeditstep = QLineEdit(f"{0.1:.4f}")
        # self.defyratiolineedit.setValidator(qdoublevalidator)
        # self.defyratiolineeditstep.setValidator(qdoublevalidator)

        # self.defyratioslider = QSlider(QtCore.Qt.Orientation.Horizontal)
        # self.defyratioslider.setMinimum(-10)
        # self.defyratioslider.setMaximum(10)
        # self.defyratioslider.setValue(1)
        # self.defyratioslider.setTickPosition(QSlider.TickPosition.TicksBelow)
        # self.defyratioslider.setTickPosition(QSlider.TicksBelow)

        # self.usedefratio = QCheckBox("Use Def Ratio")

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.ybuttonwobble)
        # hbox.addWidget(QLabel('Wobble Y Frequency'))
        # hbox.addWidget(self.defywobblefreqlineedit)
        # hbox.addWidget(QLabel('Wobble Y Amplitude'))
        # hbox.addWidget(self.defywobbleamplineedit)
        # vbox.addLayout(hbox)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.defyratiolabel)
        # hbox.addWidget(self.defyratiolineedit)
        # hbox.addWidget(QLabel('Def Ratio Y Response Slider Step = '))
        # hbox.addWidget(self.defyratiolineeditstep)
        # vbox.addLayout(hbox)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.defyratioslider)
        # vbox.addLayout(hbox)
        # hbox = QHBoxLayout()
        # hbox.addWidget(self.usedefratio)
        # vbox.addLayout(hbox)

        self.box.setLayout(vbox)

        hbox = QHBoxLayout()

        self.updefxlabel_table = QLabel('Upper X Deflection = ' + f"{updefx:.2f}")
        self.updefxlabel_table.setMinimumWidth(80)
        self.updefylabel_table = QLabel('Upper Y Deflection = ' + f"{updefy:.2f}")
        self.updefylabel_table.setMinimumWidth(80)
        self.lowdefxlabel_table = QLabel('Lower X Deflection = ' + f"{lowdefx:.2f}")
        self.lowdefxlabel_table.setMinimumWidth(80)
        self.lowdefylabel_table = QLabel('Lower Y Deflection = ' + f"{lowdefy:.2f}")
        self.lowdefylabel_table.setMinimumWidth(80)
        self.defyratiolabel_table = QLabel('Y Deflector Ratio = ' + f"{1:.2f}")
        self.defxratiolabel_table = QLabel('X Deflector Ratio = ' + f"{1:.2f}")

        hbox_labels = QHBoxLayout()
        hbox_labels.addWidget(self.updefxlabel_table)
        hbox_labels.addWidget(self.updefylabel_table)
        hbox_labels.addWidget(self.lowdefxlabel_table)
        hbox_labels.addWidget(self.lowdefylabel_table)
        hbox_labels.addWidget(self.defxratiolabel_table)
        hbox_labels.addWidget(self.defyratiolabel_table)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_labels)
        self.table.setLayout(vbox)
        return self

    def get_geom(self):
        elements = []
        phi = np.pi / 2
        radius = 0.25
        n_arc = 64

        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.entrance_z,
            n_arc=n_arc,
        )
        elements.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        elements.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.exit_z,
            n_arc=n_arc,
        )
        elements.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        elements.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        return elements


class DetectorGUI(ComponentGUIWrapper):
    @property
    def detector(self) -> 'comp.Detector':
        return self.component

    @Slot(float)
    def set_pixelsize(self, val: float):
        self.detector.pixel_size = val
        self.try_update()

    @Slot(float)
    def set_rotation(self, val: float):
        self.detector.rotation = np.deg2rad(val)
        self.try_update()

    @Slot(bool)
    def set_flip_y(self, val: bool):
        self.detector.flip_y = bool(val)
        self.try_update()

    @Slot(str)
    @Slot(int)
    @Slot(float)
    def set_shape(self, val: bool):
        self.detector.shape = (
            abs(self.ysize.getValue()),
            abs(self.xsize.getValue()),
        )
        self.try_update()

    def build(self) -> Self:
        vbox = QVBoxLayout()

        self.pixelsizeslider, _ = labelled_slider(
            self.detector.pixel_size, 0.001, 0.1, name="Pixel size",
            insert_into=vbox, decimals=3,
        )
        self.pixelsizeslider.valueChanged.connect(self.set_pixelsize)

        self.rotationslider, _ = labelled_slider(
            self.detector.rotation, -180., 180., name="Rotation",
            insert_into=vbox, decimals=1,
        )
        self.rotationslider.valueChanged.connect(self.set_rotation)

        hbox = QHBoxLayout()
        self.xsize = LabelledIntField("X-dim", initial_value=self.detector.shape[1])
        self.ysize = LabelledIntField("Y-dim", initial_value=self.detector.shape[0])
        self.xsize.insert_into(hbox)
        self.ysize.insert_into(hbox)
        self.xsize.lineEdit.textChanged.connect(self.set_shape)
        self.ysize.lineEdit.textChanged.connect(self.set_shape)
        self.flipy_cbox = QCheckBox("Flip-y")
        self.flipy_cbox.setChecked(self.detector.flip_y)
        self.flipy_cbox.stateChanged.connect(self.set_flip_y)
        hbox.addWidget(self.flipy_cbox)
        hbox.addStretch()
        vbox.addLayout(hbox)

        self.box.setLayout(vbox)
        return self

    def get_geom(self):
        vertices, faces = comp_geom.square(
            w=0.5,
            x=0.,
            y=0.,
            z=Z_ORIENT * self.component.z,
        )
        colors = np.ones((vertices.shape[0], 4))
        colors[..., 3] = 0.9
        mesh = gl.GLMeshItem(
            meshdata=gl.MeshData(
                vertexes=vertices,
                faces=faces,
                vertexColors=colors,
            ),
            smooth=False,
            drawEdges=False,
            drawFaces=True,
            shader='shaded',
        )
        return [mesh]
