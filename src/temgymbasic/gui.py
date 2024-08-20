from typing import List, Iterable, TYPE_CHECKING, NamedTuple, Optional, Tuple
from typing_extensions import Self
import weakref
from contextlib import nullcontext, contextmanager

from PySide6.QtGui import QVector3D
from PySide6.QtCore import (
    Slot,
    Qt,
    QTimer,
)
from PySide6.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QLabel,
    QHBoxLayout,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QComboBox,
)
from PySide6.QtGui import (
    QKeyEvent,
)
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import numpy as np

from . import shapes as comp_geom
from .utils import as_gl_lines, P2R, R2P
from .widgets import labelled_slider, LabelledIntField, GLImageItem

if TYPE_CHECKING:
    from .model import Model, STEMModel
    from . import components as comp
    from . import Radians


LABEL_RADIUS = 0.3
Z_ORIENT = -1
RAY_COLOR = (0., 0.8, 0.)


class GridGeomParams(NamedTuple):
    w: float
    h: float
    cx: float
    cy: float
    rotation: 'Radians'
    z: float
    shape: Optional[Tuple[int, int]]


class GridGeomMixin:
    def _get_extents(self) -> GridGeomParams:
        # (cx, cy, w, h, rotation, z)
        raise NotImplementedError()

    def _get_image(self):
        return np.asarray(
            (((255, 128, 128, 255),),),
            dtype=np.uint8,
        )

    def _get_grid_verts(self):
        geom = self._get_extents()
        rotation = geom.rotation
        shape = geom.shape
        if shape is None or any(s <= 0 for s in shape):
            return None
        vertices = self._get_mesh(rotation=0.)
        min_x, min_y, z = vertices.min(axis=0)
        max_x, max_y, _ = vertices.max(axis=0)
        ny, nx = shape
        # this can be done with clever striding / reshaping
        xvals = np.linspace(min_x, max_x, num=nx + 1, endpoint=True)[1:-1]
        xfill = np.asarray((min_y, max_y))
        xfill = np.tile(xfill, xvals.size)
        xvals = np.repeat(xvals, 2)
        yvals = np.linspace(min_y, max_y, num=ny + 1, endpoint=True)[1:-1]
        yfill = np.asarray((min_x, max_x))
        yfill = np.tile(yfill, yvals.size)
        yvals = np.repeat(yvals, 2)

        if rotation != 0.:
            mag, ang = R2P(xvals + xfill * 1j)
            xcplx = P2R(mag, ang + rotation)
            xvals, xfill = xcplx.real, xcplx.imag
            mag, ang = R2P(yfill + yvals * 1j)
            ycplx = P2R(mag, ang + rotation)
            yfill, yvals = ycplx.real, ycplx.imag

        xlines = np.stack((xvals, xfill, np.full_like(xvals, z)), axis=1)
        ylines = np.stack((yfill, yvals, np.full_like(yvals, z)), axis=1)
        return np.concatenate((xlines, ylines), axis=0)

    def get_geom(self):
        vertices = self._get_mesh()
        self.geom_border = gl.GLLinePlotItem(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0., 0., 0., 8.),
            antialias=True,
            mode='line_strip',
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            self.geom_grid = gl.GLLinePlotItem(
                pos=grid_verts,
                color=(0., 0., 0., 0.2),
                antialias=True,
                mode='lines',
            )
        self.geom_image = GLImageItem(
            vertices,
            self._get_image(),
        )
        return [self.geom_image, self.geom_grid, self.geom_border]

    def update_geometry(self):
        vertices = self._get_mesh()
        self.geom_image.setVertices(
            vertices,
        )
        grid_verts = self._get_grid_verts()
        if grid_verts is not None:
            self.geom_grid.setData(
                pos=grid_verts,
                color=(0., 0., 0., 0.3),
                antialias=True,
            )
        self.geom_border.setData(
            pos=np.concatenate((vertices, vertices[:1, :]), axis=0),
            color=(0., 0., 0., 1.),
            antialias=True,
        )

    def _get_mesh(self, rotation=None):
        geom = self._get_extents()
        if rotation is None:
            rotation = geom.rotation
        vertices, _ = comp_geom.rectangle(
            w=geom.w,
            h=geom.h,
            x=geom.cx,
            y=geom.cy,
            z=Z_ORIENT * geom.z,
            rotation=rotation,
        )
        return vertices


class ComponentGUIWrapper:
    def __init__(self, component: 'comp.Component', window: 'TemGymWindow'):
        self.component = component
        self.window = weakref.ref(window)
        self.box = QGroupBox(component.name)
        self.table = QGroupBox(component.name)
        self.blocked = False

    def update_geometry(self):
        pass

    @property
    def model(self):
        window = self.window()
        if window is not None:
            return window.model
        return None

    @property
    def model_gui(self):
        window = self.window()
        if window is not None:
            return window.model_gui
        return None

    def try_update(self, rays: bool = True, geom: bool = False):
        if self.blocked:
            return
        window = self.window()
        if window is not None:
            if rays:
                window.update_rays()
            if geom:
                window.update_geometry()

    @Slot(int)
    @Slot(float)
    @Slot(str)
    def try_update_slot(self, val):
        self.try_update()

    def sync(self, block: bool = True):
        pass

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

    @contextmanager
    def _update_blocker(self, *args):
        self.blocked = True
        yield
        self.blocked = False

    def _get_blocker(self, block: bool):
        if block:
            return self._update_blocker
        else:
            return nullcontext


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
        self.add_geometry()
        self.update_rays()

        # Any model-specific GUI setup called in finalize
        self.model_gui.finalize(self)

        # Update rays timer
        self.rays_timer = QTimer(parent=self)
        self.rays_timer.timeout.connect(self.update_rays)
        self.rays_timer.start(30)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
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

        # Add the window to the dock
        self.tem_dock.addWidget(self.tem_window)

    def set_camera_x(self):
        self.tem_window.setCameraParams(**self.x_camera_params)

    def set_camera_y(self):
        self.tem_window.setCameraParams(**self.y_camera_params)

    def set_camera_initial(self):
        self.tem_window.setCameraParams(**self.initial_camera_params)

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
        scroll = QScrollArea()
        scroll.setWidgetResizable(1)
        content = QWidget()
        scroll.setWidget(content)
        self.table_layout = QVBoxLayout(content)
        self.table_dock.addWidget(scroll, 1, 0)

        # Create the window which houses the GUI
        scroll = QScrollArea()
        scroll.setWidgetResizable(1)
        content = QWidget()
        scroll.setWidget(content)
        self.gui_dock.addWidget(scroll, 1, 0)

        self.gui_layout = QVBoxLayout(content)
        self.model_gui = self.model.gui_wrapper()(window=self).build()
        self.gui_layout.addWidget(self.model_gui.box)

        for gui_component in self.gui_components:
            self.gui_layout.addWidget(gui_component.box)
            self.table_layout.addWidget(gui_component.table)

        self.gui_layout.addStretch()

    @Slot()
    def update_rays(self):
        all_rays = tuple(self.model.run_iter(
            self.gui_components[0].num_rays,
            random=True,
        ))

        if self.model.components[0] == comp.GaussBeam:
            all_rays = all_rays[5::]

        vertices = as_gl_lines(all_rays, z_mult=Z_ORIENT)
        self.ray_geometry.setData(
            pos=vertices,
            color=RAY_COLOR + (0.05,),
        )

        if self.model.detector is not None:
            image = self.model.detector.get_image(
                all_rays[-1]
            )
            self.spot_img.setImage(image.T)

    def add_geometry(self):
        self.tem_window.clear()
        # Loop through all of the model components
        # and add their geometry to the TEM window.
        # FIXME Add in reverse to simulate better depth stacking
        for component in reversed(self.gui_components):
            for geometry in component.get_geom():
                self.tem_window.addItem(geometry)
        # Add labels next so they appear above geometry
        for component in reversed(self.gui_components):
            self.tem_window.addItem(component.get_label())
        # Add the ray geometry last so it is always on top
        self.tem_window.addItem(self.ray_geometry)

    @Slot()
    def update_geometry(self):
        for component in self.gui_components:
            component.update_geometry()


class ModelComponent(NamedTuple):
    name: str = "Model"


class ModelGUI(ComponentGUIWrapper):
    def __init__(self, window: TemGymWindow, component=ModelComponent()):
        super().__init__(component=component, window=window)

    def build(self):
        vbox = QVBoxLayout()

        self.beamSelect = QComboBox()
        self.beamSelect.addItem("Parallel Beam")
        self.beamSelect.addItem("Point Beam")
        self.beamSelect.addItem("Axial Beam")
        self.beamSelectLabel = QLabel("Beam type")

        hbox = QHBoxLayout()
        hbox.addWidget(self.beamSelectLabel)
        hbox.addWidget(self.beamSelect)
        vbox.addLayout(hbox)

        self.view_label = QLabel('Set Camera View')
        self.init_button = QPushButton('Initial View')
        self.init_button.pressed.connect(self.set_camera_initial)
        self.x_button = QPushButton('X View')
        self.x_button.pressed.connect(self.set_camera_x)
        self.y_button = QPushButton('Y View')
        self.y_button.pressed.connect(self.set_camera_y)

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
        return self

    @Slot()
    def set_camera_x(self):
        window = self.window()
        if window is not None:
            window.set_camera_x()

    @Slot()
    def set_camera_y(self):
        window = self.window()
        if window is not None:
            window.set_camera_y()

    @Slot()
    def set_camera_initial(self):
        window = self.window()
        if window is not None:
            window.set_camera_initial()

    def finalize(self, window: TemGymWindow):
        pass

    def sync(self, block: bool = True):
        pass


class STEMModelGUI(ModelGUI):
    def __init__(
        self,
        window: TemGymWindow,
        component=ModelComponent(name="STEMModel")
    ):
        super().__init__(component=component, window=window)
        self.beam: ParallelBeamGUI = window.gui_components[0]
        self.scan_coils: DoubleDeflectorGUI = window.gui_components[1]
        self.lens: LensGUI = window.gui_components[2]
        self.sample: STEMSampleGUI = window.gui_components[3]
        self.descan_coils: DoubleDeflectorGUI = window.gui_components[4]

    @property
    def model(self) -> Optional['STEMModel']:
        window = self.window()
        if window is not None:
            return window.model
        return None

    def build(self):
        super().build()
        self.beamSelect.removeItem(2)
        self.beamSelect.removeItem(1)
        return self

    def sync(self, block: bool = True):
        model = self.model
        if model is not None:
            _, overfocus_max = model._overfocus_bounds()
            self.sample.overfocus_slider.setRange(-overfocus_max, overfocus_max)
            min_f, max_f = sorted((
                model._get_objective_f(overfocus_max),
                model._get_objective_f(-overfocus_max),
            ))
            self.lens.fslider.setRange(min_f, max_f)

            min_rad, max_rad = sorted((
                model._get_radius(self.sample.semiconv_slider.minimum()),
                model._get_radius(self.sample.semiconv_slider.maximum()),
            ))
            self.beam.beamwidthslider.setRange(min_rad, max_rad)

            mindef, maxdef = self.model._minmax_def()
            self.scan_coils.updefyslider.setRange(mindef[0], maxdef[0])
            self.scan_coils.updefxslider.setRange(mindef[1], maxdef[1])
            self.scan_coils.lowdefyslider.setRange(mindef[2], maxdef[2])
            self.scan_coils.lowdefxslider.setRange(mindef[3], maxdef[3])
            self.descan_coils.updefyslider.setRange(mindef[4], maxdef[4])
            self.descan_coils.updefxslider.setRange(mindef[5], maxdef[5])
            self.descan_coils.lowdefyslider.setRange(mindef[6], maxdef[6])
            self.descan_coils.lowdefxslider.setRange(mindef[7], maxdef[7])

        self.beam.sync(block=block)
        self.scan_coils.sync(block=block)
        self.lens.sync(block=block)
        self.sample.sync(block=block)
        self.descan_coils.sync(block=block)

    def finalize(self, window: TemGymWindow):
        for widget in (
            self.beam.beamwidthslider,
            self.beam.xangleslider,
            self.beam.yangleslider,
            self.scan_coils.updefxslider,
            self.scan_coils.updefyslider,
            self.scan_coils.lowdefxslider,
            self.scan_coils.lowdefyslider,
            self.lens.fslider,
            self.descan_coils.updefxslider,
            self.descan_coils.updefyslider,
            self.descan_coils.lowdefxslider,
            self.descan_coils.lowdefyslider,
        ):
            widget.setEnabled(False)
            widget.valueChanged.disconnect()

        self.sync(block=False)


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

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.fslider):
            self.fslider.setValue(self.lens.f)

    def build(self) -> Self:
        vbox = QVBoxLayout()

        self.fslider, _ = labelled_slider(
            self.lens.f, -5., 5., name="Focal Length", insert_into=vbox, decimals=2,
        )
        self.fslider.valueChanged.connect(self.set_f)
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


class PerfectLensGUI(LensGUI):
    @property
    def perfectlens(self) -> 'comp.PerfectLens':
        return self.component

    @Slot(float)
    def set_m(self, val):
        self.perfectlens._m = (
            self.mslider.value(),
        )
        self.perfectlens.update_m_and_principal_planes(None, None, val)
        self.try_update()

    @Slot(float)
    def set_f(self, val: float):
        if abs(val) < 1e-6:
            return

        self.perfectlens.f = val
        self.perfectlens.update_m_and_principal_planes(None, None, self.perfectlens._m)
        self.try_update()

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.fslider):
            self.fslider.setValue(self.perfectlens.f)
        with blocker(self.mslider):
            self.mslider.setValue(self.perfectlens._m)

    def build(self) -> Self:

        vbox = QVBoxLayout()

        self.fslider, _ = labelled_slider(
            self.perfectlens.f, -5., 5., name="Focal Length",
            insert_into=vbox, decimals=2,
        )
        self.fslider.valueChanged.connect(self.set_f)
        self.box.setLayout(vbox)

        self.mslider, _ = labelled_slider(
            self.perfectlens._m, -5., 5., name="Magnification",
            insert_into=vbox, decimals=2,
        )
        self.mslider.valueChanged.connect(self.set_m)
        self.box.setLayout(vbox)

        # self.flabel_table = QLabel('Focal Length = ' + f"{self.perfectlens.f:.2f}")
        # self.flabel_table.setMinimumWidth(80)

        # self.mlabel_table = QLabel('Magnification = ' + f"{self.perfectlens._m:.2f}")
        # self.mlabel_table.setMinimumWidth(80)

        # self.z1label_table = QLabel('Principal Plane z1 = ' + f"{self.perfectlens._z1:.2f}")
        # self.z1label_table.setMinimumWidth(80)

        # self.z2label_table = QLabel('Principal Plane z2 = ' + f"{self.perfectlens._z2:.2f}")
        # self.z2label_table.setMinimumWidth(80)

        # hbox = QHBoxLayout()
        # hbox.addWidget(self.flabel_table)
        # hbox.addWidget(self.mlabel_table)
        # hbox.addWidget(self.z1label_table)
        # hbox.addWidget(self.z2label_table)

        vbox = QVBoxLayout()
        # vbox.addLayout(hbox)
        self.table.setLayout(vbox)

        return self


class SourceGUI(ComponentGUIWrapper):
    @property
    def beam(self) -> 'comp.Source':
        return self.component

    @property
    def num_rays(self) -> int:
        return self.rayslider.value()

    @Slot(float)
    def set_tilt(self, val):
        self.beam.tilt_yx = (
            self.yangleslider.value(),
            self.xangleslider.value(),
        )
        self.try_update()

    def _build(self):
        num_rays = 64
        beam_tilt_y, beam_tilt_x = self.beam.tilt_yx

        self.rayslider, _ = labelled_slider(
            num_rays, 1, 4096, name="Number of Rays"
        )
        self.rayslider.valueChanged.connect(self.try_update_slot)

        common_args = dict(
            vmin=-np.pi / 4, vmax=np.pi / 4, decimals=2,
        )
        self.xangleslider, _ = labelled_slider(
            value=beam_tilt_x, name="Beam Tilt X", **common_args
        )
        self.yangleslider, _ = labelled_slider(
            value=beam_tilt_y, name="Beam Tilt Y", **common_args,
        )

        self.xangleslider.valueChanged.connect(self.set_tilt)
        self.yangleslider.valueChanged.connect(self.set_tilt)

    def _get_geom(self):
        raise NotImplementedError("Needs to be defined")

    def get_geom(self):
        self.geom = gl.GLLinePlotItem(
            pos=self._get_geom(),
            color=RAY_COLOR + (0.9,),
            width=2,
            antialias=True,
        )
        return [self.geom]

    def update_geometry(self):
        self.geom.setData(
            pos=self._get_geom(),
            color=RAY_COLOR + (0.9,),
            antialias=True,
        )


class ParallelBeamGUI(SourceGUI):
    @property
    def beam(self) -> 'comp.ParallelBeam':
        return self.component

    @Slot(float)
    def set_radius(self, val):
        self.beam.radius = val
        self.try_update(geom=True)

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.beamwidthslider):
            self.beamwidthslider.setValue(self.beam.radius)
        with blocker(self.xangleslider):
            self.xangleslider.setValue(self.beam.tilt_yx[1])
        with blocker(self.yangleslider):
            self.yangleslider.setValue(self.beam.tilt_yx[0])

    def build(self) -> Self:

        vbox = QVBoxLayout()

        self._build()

        vbox.addWidget(self.rayslider)

        beam_radius = self.beam.radius
        self.beamwidthslider, _ = labelled_slider(
            beam_radius, 0.001, 0.1, name='Parallel Beam Width', insert_into=vbox, decimals=3
        )
        self.beamwidthslider.valueChanged.connect(self.set_radius)

        hbox = QHBoxLayout()
        hbox.addWidget(self.xangleslider)
        hbox.addWidget(self.yangleslider)
        vbox.addLayout(hbox)
        self.box.setLayout(vbox)
        return self

    def _get_geom(self):
        return comp_geom.lens(
            self.beam.radius,
            Z_ORIENT * self.component.z,
            64,
        ).T


class GaussBeamGUI(SourceGUI):
    @property
    def beam(self) -> 'comp.GaussBeam':
        return self.component

    @Slot(float)
    def set_radius(self, val):
        self.beam.radius = val
        self.try_update(geom=True)

    @Slot(float)
    def set_wo(self, val):
        self.beam.wo = val
        self.try_update(geom=True)

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.beamwidthslider):
            self.beamwidthslider.setValue(self.beam.radius)
        with blocker(self.xangleslider):
            self.xangleslider.setValue(self.beam.tilt_yx[1])
        with blocker(self.yangleslider):
            self.yangleslider.setValue(self.beam.tilt_yx[0])
        with blocker(self.woslider):
            self.woslider.setValue(self.beam.wo)

    def build(self) -> Self:

        vbox = QVBoxLayout()

        self._build()

        vbox.addWidget(self.rayslider)

        beam_radius = self.beam.radius
        self.beamwidthslider, _ = labelled_slider(
            beam_radius, 0.001, 0.1, name='Parallel Beam Width', insert_into=vbox, decimals=3
        )
        self.beamwidthslider.valueChanged.connect(self.set_radius)

        wo = self.beam.wo
        self.woslider, _ = labelled_slider(
            wo, 0.001, 1, name='Gaussian Beam Width', insert_into=vbox, decimals=3
        )
        self.woslider.valueChanged.connect(self.set_wo)

        hbox = QHBoxLayout()
        hbox.addWidget(self.xangleslider)
        hbox.addWidget(self.yangleslider)
        vbox.addLayout(hbox)
        self.box.setLayout(vbox)
        return self

    def _get_geom(self):
        return comp_geom.lens(
            self.beam.radius,
            Z_ORIENT * self.component.z,
            64,
        ).T


class PointBeamGUI(SourceGUI):
    @property
    def beam(self) -> 'comp.PointBeam':
        return self.component

    @Slot(float)
    def set_semi_angle(self, val):
        self.beam.semi_angle = val
        self.try_update(geom=True)

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.beamsemiangleslider):
            self.beamsemiangleslider.setValue(self.beam.semi_angle)

    def build(self) -> Self:
        beam_semi_angle = self.beam.semi_angle

        vbox = QVBoxLayout()

        self._build()

        vbox.addWidget(self.rayslider)

        self.beamsemiangleslider, _ = labelled_slider(
            beam_semi_angle,
            0.0001,
            0.3,
            name='Point Beam Semi Angle',
            insert_into=vbox,
            decimals=3,
        )
        self.beamsemiangleslider.valueChanged.connect(self.set_semi_angle)

        hbox = QHBoxLayout()
        hbox.addWidget(self.xangleslider)
        hbox.addWidget(self.yangleslider)
        vbox.addLayout(hbox)
        self.box.setLayout(vbox)

        return self

    def _get_geom(self):
        return comp_geom.lens(
            0.,
            Z_ORIENT * self.component.z,
            64,
        ).T


class SampleGUI(GridGeomMixin, ComponentGUIWrapper):
    @property
    def sample(self) -> 'comp.Sample':
        return self.component

    def _get_image(self):
        return np.asarray(
            (((0, 128, 255, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy, sx = self.sample.shape
        py = self.sample.pixel_size
        px = self.sample.pixel_size
        return GridGeomParams(
            cx=-1 * px / 2.,
            cy=-1 * py / 2.,
            w=sx * px,
            h=sy * py,
            rotation=self.sample.rotation,
            z=self.component.z,
            shape=self.sample.shape,
        )

    def build(self):
        return self


class STEMSampleGUI(SampleGUI):
    @property
    def sample(self) -> 'comp.STEMSample':
        return self.component

    def set_stem_generic(self, update_kwargs=None, **kwargs):
        if update_kwargs is None:
            update_kwargs = {}
        model: 'STEMModel' = self.model
        if model is not None:
            model.set_stem_params(**kwargs)
            self.try_update(**update_kwargs)
        model_gui: 'STEMModelGUI' = self.model_gui
        if model_gui is not None:
            model_gui.sync()

    @Slot(float)
    def set_overfocus(self, val):
        self.set_stem_generic(
            update_kwargs=dict(geom=True),
            overfocus=val,
        )

    @Slot(float)
    def set_semiconv(self, val):
        self.set_stem_generic(
            update_kwargs=dict(geom=True),
            semiconv_angle=val,
        )

    @Slot(float)
    def set_scan_rotation(self, val):
        self.set_stem_generic(
            scan_rotation=val,
            update_kwargs=dict(geom=True),
        )

    @Slot(float)
    def set_camera_length(self, val):
        self.set_stem_generic(
            update_kwargs=dict(geom=True),
            camera_length=val,
        )

    @Slot(float)
    def set_scanstep(self, val):
        self.set_stem_generic(scan_step_yx=(
            self.scanstep_y.value(),
            self.scanstep_x.value(),
        ), update_kwargs=dict(geom=True),
        )

    @Slot(int)
    def set_scanshape(self, val):
        self.set_stem_generic(scan_shape=(
            abs(self.ysize.getValue()),
            abs(self.xsize.getValue()),
        ), update_kwargs=dict(geom=True)
        )
        ny, nx = self.sample.scan_shape
        self.scanpos_x.setValue(min(nx - 1, self.scanpos_x.value()))
        self.scanpos_y.setValue(min(ny - 1, self.scanpos_y.value()))
        self.scanpos_x.setMaximum(nx - 1)
        self.scanpos_y.setMaximum(ny - 1)

    @Slot(int)
    def set_scanpos(self, val):
        model: 'STEMModel' = self.model
        if model is not None:
            model.move_to((
                abs(self.scanpos_y.value()),
                abs(self.scanpos_x.value()),
            ))
            self.try_update()
        model_gui: 'STEMModelGUI' = self.model_gui
        if model_gui is not None:
            model_gui.sync()

    def build(self) -> Self:
        vbox = QVBoxLayout()

        self.overfocus_slider, _ = labelled_slider(
            value=self.sample.overfocus,
            vmin=-0.1,
            vmax=0.1,
            decimals=2,
            name="Overfocus",
            insert_into=vbox,
        )
        self.overfocus_slider.valueChanged.connect(self.set_overfocus)

        self.semiconv_slider, _ = labelled_slider(
            value=self.sample.semiconv_angle,
            vmin=0.001,
            vmax=0.100,
            decimals=3,
            name="Semiconv (rad)",
            insert_into=vbox,
        )
        self.semiconv_slider.valueChanged.connect(self.set_semiconv)

        if self.model is not None:
            camera_length = self.model.camera_length
        self.cameralength_slider, _ = labelled_slider(
            value=camera_length,
            vmin=0.01,
            vmax=1.,
            decimals=2,
            name="Camera length",
            insert_into=vbox,
        )
        self.cameralength_slider.valueChanged.connect(self.set_camera_length)

        self.scan_rotation_slider, _ = labelled_slider(
            value=self.sample.scan_rotation,
            vmin=-180.,
            vmax=180.,
            decimals=1,
            name="Scan rotation",
            insert_into=vbox,
            tick_interval=10.,
        )
        self.scan_rotation_slider.valueChanged.connect(self.set_scan_rotation)

        x_hbox = QHBoxLayout()
        y_hbox = QHBoxLayout()
        self.xsize = LabelledIntField(
            "Scan Dim-X", initial_value=self.sample.scan_shape[1]
        )
        self.ysize = LabelledIntField(
            "Scan Dim-Y", initial_value=self.sample.scan_shape[0]
        )
        self.xsize.insert_into(x_hbox)
        self.ysize.insert_into(y_hbox)
        self.xsize.lineEdit.textChanged.connect(self.set_scanshape)
        self.ysize.lineEdit.textChanged.connect(self.set_scanshape)

        self.scanstep_x, _ = labelled_slider(
            value=self.sample.scan_step_yx[1],
            vmin=-0.05,
            vmax=0.05,
            decimals=2,
            name="Step-X",
            insert_into=x_hbox,
        )
        self.scanstep_y, _ = labelled_slider(
            value=self.sample.scan_step_yx[1],
            vmin=-0.05,
            vmax=0.05,
            decimals=2,
            name="Step-Y",
            insert_into=y_hbox,
        )
        self.scanstep_x.valueChanged.connect(self.set_scanstep)
        self.scanstep_y.valueChanged.connect(self.set_scanstep)

        ny, nx = self.sample.scan_shape
        self.scanpos_x, _ = labelled_slider(
            value=0,
            vmin=0,
            vmax=nx - 1,
            name="Pos-X",
            insert_into=x_hbox,
        )
        self.scanpos_y, _ = labelled_slider(
            value=0,
            vmin=0,
            vmax=ny - 1,
            name="Pos-Y",
            insert_into=y_hbox,
        )
        self.scanpos_x.valueChanged.connect(self.set_scanpos)
        self.scanpos_y.valueChanged.connect(self.set_scanpos)

        vbox.addLayout(x_hbox)
        vbox.addLayout(y_hbox)

        self.box.setLayout(vbox)
        return self

    def _get_image(self):
        return np.asarray(
            (((0, 128, 255, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy, sx = self.sample.scan_shape
        py, px = self.sample.scan_step_yx
        return GridGeomParams(
            cx=-1 * px / 2.,
            cy=-1 * py / 2.,
            w=sx * px,
            h=sy * py,
            rotation=self.sample.scan_rotation_rad,
            z=self.component.z,
            shape=self.sample.scan_shape,
        )


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

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.updefxslider):
            self.updefxslider.setValue(self.d_deflector.first.defx)
        with blocker(self.updefyslider):
            self.updefyslider.setValue(self.d_deflector.first.defy)
        with blocker(self.lowdefxslider):
            self.lowdefxslider.setValue(self.d_deflector.second.defx)
        with blocker(self.lowdefyslider):
            self.lowdefyslider.setValue(self.d_deflector.second.defy)

    def build(self) -> Self:
        updefx = self.d_deflector.first.defx
        updefy = self.d_deflector.first.defy
        lowdefx = self.d_deflector.second.defx
        lowdefy = self.d_deflector.second.defy

        vbox = QVBoxLayout()

        common_args = dict(
            vmin=-0.1, vmax=0.1, decimals=2,
        )
        self.updefxslider, hbox = labelled_slider(
            value=updefx, name="Upper X Deflection", **common_args
        )
        self.updefyslider, _ = labelled_slider(
            value=updefy, name="Upper Y Deflection", **common_args,
            insert_into=hbox
        )
        vbox.addLayout(hbox)
        self.updefxslider.valueChanged.connect(self.set_updefx)
        self.updefyslider.valueChanged.connect(self.set_updefy)

        self.lowdefxslider, hbox = labelled_slider(
            value=lowdefx, name="Lower X Deflection", **common_args
        )
        self.lowdefyslider, _ = labelled_slider(
            value=lowdefy, name="Lower Y Deflection", **common_args,
            insert_into=hbox
        )
        vbox.addLayout(hbox)
        self.lowdefxslider.valueChanged.connect(self.set_lowdefx)
        self.lowdefyslider.valueChanged.connect(self.set_lowdefy)

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
        self.geom = []
        phi = np.pi / 2
        radius = 0.25
        n_arc = 64

        arc1, arc2 = comp_geom.deflector(
            r=radius,
            phi=phi,
            z=Z_ORIENT * self.component.entrance_z,
            n_arc=n_arc,
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        self.geom.append(
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
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc1.T, color="r", width=5
            )
        )
        self.geom.append(
            gl.GLLinePlotItem(
                pos=arc2.T, color="b", width=5
            )
        )
        return self.geom


class DetectorGUI(GridGeomMixin, ComponentGUIWrapper):
    @property
    def detector(self) -> 'comp.Detector':
        return self.component

    @Slot(float)
    def set_pixelsize(self, val: float):
        self.detector.pixel_size = val
        self.try_update(geom=True)

    @Slot(float)
    def set_rotation(self, val: float):
        self.detector.rotation = val
        self.try_update(geom=True)

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
        self.try_update(geom=True)

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.xsize):
            self.xsize.lineEdit.setValue(str(self.detector.shape[1]))
        with blocker(self.ysize):
            self.ysize.lineEdit.setValue(str(self.detector.shape[0]))
        with blocker(self.pixelsizeslider):
            self.pixelsizeslider.setValue(self.detector.pixel_size)
        with blocker(self.flipy_cbox):
            self.flipy_cbox.setChecked(self.detector.flip_y)
        with blocker(self.rotationslider):
            self.rotationslider.setValue(self.detector.rotation)

    def build(self) -> Self:
        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        self.xsize = LabelledIntField("X-dim", initial_value=self.detector.shape[1])
        self.ysize = LabelledIntField("Y-dim", initial_value=self.detector.shape[0])
        self.xsize.insert_into(hbox)
        self.ysize.insert_into(hbox)
        self.xsize.lineEdit.textChanged.connect(self.set_shape)
        self.ysize.lineEdit.textChanged.connect(self.set_shape)
        self.pixelsizeslider, _ = labelled_slider(
            self.detector.pixel_size, 0.001, 0.03, name="Pixel size",
            insert_into=hbox, decimals=5,
        )
        self.pixelsizeslider.valueChanged.connect(self.set_pixelsize)
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        self.flipy_cbox = QCheckBox("Flip-y")
        self.flipy_cbox.setChecked(self.detector.flip_y)
        self.flipy_cbox.stateChanged.connect(self.set_flip_y)
        hbox.addWidget(self.flipy_cbox)
        self.rotationslider, _ = labelled_slider(
            self.detector.rotation, -180., 180., name="Rotation",
            insert_into=hbox, decimals=1, tick_interval=10.,
        )
        self.rotationslider.valueChanged.connect(self.set_rotation)
        vbox.addLayout(hbox)

        self.box.setLayout(vbox)
        return self

    def _get_image(self):
        return np.asarray(
            (((10, 190, 100, 170),),),
            dtype=np.uint8,
        )

    def _get_extents(self):
        sy, sx = self.detector.shape
        pixelsize = self.detector.pixel_size
        # cx, cy, w, h, rotation, z, px_shape
        return GridGeomParams(
            cx=-1 * pixelsize / 2.,
            cy=-1 * pixelsize / 2.,
            w=sx * pixelsize,
            h=sy * pixelsize,
            rotation=self.detector.rotation_rad,
            z=self.component.z,
            shape=self.detector.shape,
        )


class ApertureGUI(LensGUI):
    @property
    def aperture(self) -> 'comp.Aperture':
        return self.component

    def build(self) -> Self:

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        self.radiusslider, _ = labelled_slider(
            self.aperture.radius, 0.0, 0.25,
            name="Radius", insert_into=hbox, decimals=2,
        )
        vbox.addLayout(hbox)

        hbox = QHBoxLayout()
        self.xpos_slider, _ = labelled_slider(
            self.aperture.x, -0.25, -0.25,
            name="Pos-X", insert_into=hbox, decimals=2,
        )
        self.ypos_slider, _ = labelled_slider(
            self.aperture.y, -0.25, -0.25,
            name="Pos-Y", insert_into=hbox, decimals=2,
        )
        vbox.addLayout(hbox)

        self.box.setLayout(vbox)
        return self


class BiprismGUI(ComponentGUIWrapper):
    @property
    def biprism(self) -> 'comp.Biprism':
        return self.component

    @Slot(float)
    def set_deflection(self, val: float):
        self.biprism.deflection = val
        self.try_update()

    @Slot(float)
    def set_offset(self, val: float):
        self.biprism.offset = val
        self.try_update(geom=True)

    @Slot(float)
    def set_rotation(self, val: float):
        self.biprism.rotation = val
        self.try_update(geom=True)

    def sync(self, block: bool = True):
        blocker = self._get_blocker(block)
        with blocker(self.deflection_slider):
            self.deflection_slider.setValue(self.biprism.deflection)
        with blocker(self.offset_slider):
            self.offset_slider.setValue(self.biprism.offset)
        with blocker(self.rotation_slider):
            self.rotation_slider.setValue(self.biprism.rotation)

    def build(self) -> Self:
        deflection = self.biprism.deflection
        offset = self.biprism.offset
        rotation = self.biprism.rotation

        vbox = QVBoxLayout()

        hbox = QHBoxLayout()
        common_args = dict(
            vmin=-0.5, vmax=0.5, decimals=2,
        )
        self.deflection_slider, hbox = labelled_slider(
            value=deflection, name="Deflection", **common_args
        )
        self.offset_slider, _ = labelled_slider(
            offset, -0.25, 0.25,
            name="X-Offset", insert_into=hbox, decimals=2,
        )
        self.rotation_slider, _ = labelled_slider(
            rotation, -180, 180,
            name="Rotation", insert_into=hbox, decimals=2,
        )

        self.deflection_slider.valueChanged.connect(self.set_deflection)
        self.offset_slider.valueChanged.connect(self.set_offset)
        self.rotation_slider.valueChanged.connect(self.set_rotation)

        vbox.addLayout(hbox)
        self.box.setLayout(vbox)

        return self

    def _get_geom(self):
        return comp_geom.biprism(
            Z_ORIENT*self.biprism.z,
            1,
            self.biprism.rotation_rad,
            self.biprism.offset,
        )

    def get_geom(self):
        self.geom = gl.GLLinePlotItem(
            pos=self._get_geom(),
            color='white',
            width=5,
            antialias=True,
        )
        return [self.geom]

    def update_geometry(self):
        self.geom.setData(
            pos=self._get_geom(),
            color='white',
            antialias=True,
        )
