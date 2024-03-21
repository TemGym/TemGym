import abc
from typing import (
    Generator, Tuple, Optional, Type,
    TYPE_CHECKING
)

import numpy as np
from numpy.typing import NDArray

from . import (
    UsageError,
    InvalidModelError,
    PositiveFloat,
    NonNegativeFloat,
    Radians
)
from .rays import Rays
from .utils import (
    P2R, R2P,
    circular_beam,
)

if TYPE_CHECKING:
    from .gui import ComponentGUIWrapper


class Component(abc.ABC):
    def __init__(self, z: float, name: Optional[str] = None):
        if name is None:
            name = type(self).__name__
        self._name = name
        self._z = z

    def _validate_component(self):
        pass

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, new_z: float):
        raise UsageError("Do not set z on a component directly, use Model methods")

    def _set_z(self, new_z: float):
        self._z = new_z

    @property
    def entrance_z(self) -> float:
        return self.z

    @property
    def exit_z(self) -> float:
        return self.z

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}: {self._name} @ z = {self.z}'

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        raise NotImplementedError

    @staticmethod
    def gui_wrapper() -> Optional[Type['ComponentGUIWrapper']]:
        return None


class Lens(Component):
    def __init__(self, z: float, f: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)
        self._f = f

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def ffp(self) -> float:
        return self.z - abs(self.f)

    @staticmethod
    def lens_matrix(f):
        '''
        Lens ray transfer matrix

        Parameters
        ----------
        f : float
            Focal length of lens

        Returns
        -------
        ndarray
            Output Ray Transfer Matrix
        '''
        f = f + 1e-6  # div-0
        return np.array(
            [[1,      0, 0,      0, 0],
             [-1 / f, 1, 0,      0, 0],
             [0,      0, 1,      0, 0],
             [0,      0, -1 / f, 1, 0],
             [0,      0, 0,      0, 1]]
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Just straightforward matrix multiplication
        yield Rays(
            data=np.matmul(self.lens_matrix(self.f), rays.data),
            indices=rays.indices,
            location=self,
            path_length=rays.path_length
        )

    @staticmethod
    def gui_wrapper():
        from .gui import LensGUI
        return LensGUI


class Sample(Component):
    def __init__(self, z: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Sample has no effect, yet
        # Could implement ray intensity / attenuation ??
        rays.location = self
        yield rays

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI


class STEMSample(Sample):
    def __init__(
        self,
        z: float,
        overfocus: NonNegativeFloat = 0.,
        semiconv_angle: PositiveFloat = 0.01,
        scan_shape: Tuple[int, int] = (8, 8),
        scan_step_yx: Tuple[float, float] = (0.01, 0.01),
        scan_rotation: Radians = 0.,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx
        self.scan_rotation = scan_rotation

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step_yx
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y
        if self.scan_rotation != 0.:
            pos_r, pos_a = R2P(scan_position_x + scan_position_y * 1j)
            pos_c = P2R(pos_r, pos_a + self.scan_rotation)
            scan_position_y, scan_position_x = pos_c.imag, pos_c.real
        return (scan_position_y, scan_position_x)

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.scan_shape,
            # This needs to be refactored...
            pixel_size=self.scan_step_yx[0],
            rotation=self.scan_rotation,
            as_int=as_int,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import STEMSampleGUI
        return STEMSampleGUI


class Source(Component):
    def __init__(
        self, z: float, tilt_yx: Tuple[float, float] = (0., 0.), name: Optional[str] = None
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx

    @abc.abstractmethod
    def get_rays(self, num_rays: int) -> Rays:
        raise NotImplementedError

    def _make_rays(self, r: NDArray, indices: Optional[NDArray] = None) -> Rays:
        if indices is None:
            indices = np.arange(r.shape[1])
        r[1, :] += self.tilt_yx[1]
        r[3, :] += self.tilt_yx[0]
        return Rays(
            data=r, indices=indices, location=self, path_length=np.zeros((r.shape[1],))
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Source has no effect after get_rays was called
        rays.location = self
        yield rays


class ParallelBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name)
        self.radius = radius

    def get_rays(self, num_rays: int) -> Rays:
        r, _ = circular_beam(num_rays, self.radius)
        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import ParallelBeamGUI
        return ParallelBeamGUI


class XAxialBeam(ParallelBeam):
    def get_rays(self, num_rays: int) -> Rays:
        r = np.zeros((5, num_rays))
        r[0, :] = np.linspace(
            -self.radius, self.radius, num=num_rays, endpoint=True
        )
        return self._make_rays(r)


class RadialSpikesBeam(ParallelBeam):
    def get_rays(self, num_rays: int) -> Rays:
        xvals = np.linspace(
            0., self.radius, num=num_rays // 4, endpoint=True
        )
        yvals = np.zeros_like(xvals)
        origin_c = xvals + yvals * 1j

        orad, oang = R2P(origin_c)
        radius1 = P2R(orad * 0.75, oang + np.pi * 0.4)
        radius2 = P2R(orad * 0.5, oang + np.pi * 0.8)
        radius3 = P2R(orad * 0.25, oang + np.pi * 1.2)
        r_c = np.concatenate((origin_c, radius1, radius2, radius3))

        r = np.zeros((5, r_c.size))
        r[0, :] = r_c.real
        r[2, :] = r_c.imag
        return self._make_rays(r)


# class PointSource(Source):
#     def __init__(
#         self,
#         z: float,
#         semi_angle: Optional[float] = 0.,
#         tilt_yx: Tuple[float, float] = (0., 0.),
#         name: Optional[str] = None,
#     ):
#         super().__init__(name=name, z=z)
#         self.semi_angle = semi_angle
#         self.tilt_yx = tilt_yx

#     def get_rays(self, num_rays: int) -> Rays:
#         r, _ = point_beam(num_rays, self.semi_angle)
#         return self._make_rays(r)


class Detector(Component):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        rotation: Radians = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        """
        The intention of rotation is to rotate the detector
        realative to the common y/x coordinate system of the optics.
        A positive rotation would rotate the detector clockwise
        looking down a ray , and the image will appear
        to rotate anti-clockwise.

        In STEMModel the scan grid is aligned with the optics
        y/x coordinate system default, but can also
        be rotated using the "scan_rotation" parameter.

        The detector flip_y acts only at the image generation step,
        the scan grid itself can be flipped by setting negative
        scan step values
        """
        super().__init__(name=name, z=z)
        self.pixel_size = pixel_size
        self.shape = shape
        self.rotation = rotation  # degrees
        self.flip_y = flip_y
        self.center = center

    def set_center_px(self, center_px: Tuple[int, int]):
        """
        For the desired image center in pixels (after any flip / rotation)
        set the image center in the physical coordinates of the microscope

        The continuous coordinate can be set directly on detector.center
        """
        iy, ix = center_px
        sy, sx = self.shape
        cont_y = (iy - sy // 2) * self.pixel_size
        cont_x = (ix - sx // 2) * self.pixel_size
        if self.flip_y:
            cont_y = -1 * cont_y
        mag, angle = R2P(cont_x + 1j * cont_y)
        coord: complex = P2R(mag, angle + np.deg2rad(self.rotation))
        self.center = coord.imag, coord.real

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Detector has no effect on rays
        rays.location = self
        yield rays

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def get_image(self, rays: Rays) -> NDArray:
        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = self.on_grid(rays, as_int=True)
        sy, sx = self.shape
        mask = np.logical_and(
            np.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            np.logical_and(
                0 <= pixel_coords_x,
                pixel_coords_x < sx
            )
        )
        image = np.zeros(
            self.shape,
            dtype=int,
        )
        flat_icds = np.ravel_multi_index(
            [
                pixel_coords_y[mask],
                pixel_coords_x[mask],
            ],
            image.shape
        )
        # Increment at each pixel for each ray that hits
        np.add.at(
            image.ravel(),
            flat_icds,
            1,
        )
        return image

    @staticmethod
    def gui_wrapper():
        from .gui import DetectorGUI
        return DetectorGUI


class Deflector(Component):
    '''Creates a single deflector component and handles calls to GUI creation, updates to GUI
        and stores the component matrix. See Double Deflector component for a more useful version
    '''
    def __init__(
        self,
        z: float,
        defx: float = 0.,
        defy: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        defy : float, optional
            deflection kick in slope units to the incoming ray y angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.defy = defy

    @staticmethod
    def deflector_matrix(def_x, def_y):
        '''Single deflector ray transfer matrix

        Parameters
        ----------
        def_x : float
            deflection in x in slope units
        def_y : _type_
            deflection in y in slope units

        Returns
        -------
        ndarray
            Output ray transfer matrix
        '''

        return np.array(
            [[1, 0, 0, 0,     0],
             [0, 1, 0, 0, def_x],
             [0, 0, 1, 0,     0],
             [0, 0, 0, 1, def_y],
             [0, 0, 0, 0,     1]],
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        yield Rays(
            data=np.matmul(
                self.deflector_matrix(self.defx, self.defy),
                rays.data,
            ),
            indices=rays.indices,
            location=self,
            path_length=rays.path_length
        )


class DoubleDeflector(Component):
    def __init__(
        self,
        first: Deflector,
        second: Deflector,
        name: Optional[str] = None,
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self._first = first
        self._second = second
        self._validate_component()

    @classmethod
    def from_params(
        cls,
        z: float,
        distance: float = 0.1,
        name: Optional[str] = None
    ):
        return cls(
            Deflector(
                z - distance / 2.
            ),
            Deflector(
                z + distance / 2.
            ),
            name=name,
        )

    def _validate_component(self):
        if self.first.z >= self.second.z:
            raise InvalidModelError("First deflector must be before second")

    @property
    def length(self) -> float:
        return self._second.z - self._first.z

    @property
    def first(self) -> Deflector:
        return self._first

    @property
    def second(self) -> Deflector:
        return self._second

    @property
    def z(self):
        self._z = (self.first.z + self.second.z) / 2
        return self._z

    def _set_z(self, new_z: float):
        dz = new_z - self.z
        self.first._set_z(self.first.z + dz)
        self.second._set_z(self.second.z + dz)

    @property
    def entrance_z(self) -> float:
        return self.first.z

    @property
    def exit_z(self) -> float:
        return self.second.z

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        for rays in self.first.step(rays):
            rays.location = (self, self.first)
            yield rays
        rays = rays.propagate_to(self.second.entrance_z)
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays

    @staticmethod
    def _send_ray_through_pts_1d(
        in_zp: Tuple[float, float],
        z_out: float,
        pt1_zp: Tuple[float, float],
        pt2_zp: Tuple[float, float],
        in_slope: float = 0.
    ) -> Tuple[float, float]:
        """
        Choose first/second deflector values such that a ray arriving
        at (in_zp) with slope (in_slope), will leave at (z_out, ...) and
        pass through (pt1_zp) then (pt2_zp)
        """
        in_zp = np.asarray(in_zp)
        pt1_zp = np.asarray(pt1_zp)
        pt2_zp = np.asarray(pt2_zp)
        dp = pt1_zp - pt2_zp
        out_zp = np.asarray(
            (
                z_out,
                pt2_zp[1] + dp[1] * (z_out - pt2_zp[0]) / dp[0],
            )
        )
        dd = out_zp - in_zp
        first_def = dd[1] / dd[0]
        first_def += in_slope
        out_slope = dp[1] / dp[0]
        second_def = out_slope - first_def
        return first_def, second_def

    def send_ray_through_points(
        self,
        in_ray: Tuple[float, float],
        pt1: Tuple[float, float, float],
        pt2: Tuple[float, float, float],
        in_slope: Tuple[float, float] = (0., 0.)
    ):
        """
        in_ray is (y, x), z is implicitly the z of the first deflector
        pt1 and pt2 are (z, y, x) after the second deflector
        in_slope is (dy, dx) at the incident point
        """
        self.first.defy, self.second.defy = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[0]),
            self.second.z,
            pt1[:2],
            pt2[:2],
            in_slope=in_slope[0],
        )
        self.first.defx, self.second.defx = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[1]),
            self.second.z,
            (pt1[0], pt1[2]),
            (pt2[0], pt2[2]),
            in_slope=in_slope[1],
        )

    @staticmethod
    def gui_wrapper():
        from .gui import DoubleDeflectorGUI
        return DoubleDeflectorGUI


class Biprism(Component):
    def __init__(
        self,
        z: float,
        x: float = 0.,
        defx: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        x: float
            Central position of biprism in x dimension
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.x = x

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x = rays.x
        x_dist = (pos_x - self.x)
        x_sign = np.sign(x_dist)
        rays.data[1] = rays.data[1] + self.defx*x_sign

        yield Rays(
            data=rays.data,
            indices=rays.indices,
            path_length=rays.path_length + np.abs(self.defx)*x_dist,
            location=self,
        )


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius_inner: float = 0.005,
        radius_outer: float = 0.25,
        x: float = 0.,
        y: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default 'Aperture'
        radius_inner : float, optional
           Inner radius of the aperture, by default 0.005
        radius_outer : float, optional
            Outer radius of the aperture, by default 0.25
        x : int, optional
            X position of the centre of the aperture, by default 0
        y : int, optional
            Y position of the centre of the aperture, by default 0
        '''

        super().__init__(z, name)

        self.x = x
        self.y = y
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x, pos_y = rays.x, rays.y
        distance = np.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        mask = np.logical_and(
            distance >= self.radius_inner,
            distance < self.radius_outer,
        )
        yield Rays(
            data=rays.data[:, mask], indices=rays.indices[mask],
            location=self, path_length=rays.path_length[mask],
        )
