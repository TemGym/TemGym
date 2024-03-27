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
    Radians,
    Degrees,
)
from .rays import Rays
from .utils import (
    P2R, R2P,
    make_beam,
    calculate_phi_0,
    calculate_wavelength
)

from scipy.constants import c, e, m_e

# Defining epsilon constant from page 18 of principles of electron optics 2017, Volume 1.
EPSILON = abs(e)/(2*m_e*c**2)


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
    def __init__(
        self,
        z: float,
        potential: NDArray,
        pixel_size: float,
        shape: Tuple[int, int],
        rotation: Radians = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.phi = potential
        self.pixel_size = pixel_size
        self.shape = shape
        self.rotation = rotation  # degrees
        self.flip_y = flip_y
        self.center = center
        self.dphi_dy, self.dphi_dx = np.gradient(potential)

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

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
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

        flat_icds = np.ravel_multi_index(
            [
                pixel_coords_y[mask],
                pixel_coords_x[mask],
            ],
            self.phi.shape
        )

        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = np.sqrt(1+rays.dx[mask]**2+rays.dy[mask]**2)  # Equation 3.16
        phi_0_plus_phi = (rays.phi_0[mask]+self.phi.ravel()[flat_icds])  # Part of Equation 2.18

        phi_hat = (phi_0_plus_phi)*(1+EPSILON*(phi_0_plus_phi))  # Equation 2.18

        # Between Equation 2.22 & 2.23
        dphi_hat_dx = 1+2*EPSILON*(phi_0_plus_phi)*self.dphi_dx.ravel()[flat_icds]
        dphi_hat_dy = 1+2*EPSILON*(phi_0_plus_phi)*self.dphi_dy.ravel()[flat_icds]

        rays.data[1][mask] += (rho**2)/(2*phi_hat)*dphi_hat_dx  # Equation 3.22
        rays.data[3][mask] += (rho**2)/(2*phi_hat)*dphi_hat_dy  # Equation 3.22

        # Note here we are ignoring the Ez component (dphi/dz) of 3.22,
        # because this has the effect of only
        # slowing the electron down, and since we have modelled the potential of the atom in a plane
        # only, we also won't have an Ez component.

        # Equation 5.16 & 5.17 & 3.16, where ds of 5.16 is replaced by ds/dz * dz,
        # where ds/dz = rho (See 3.16 and a little below it)
        rays.path_length[mask] += rho*np.sqrt(phi_hat/rays.phi_0[mask])

        yield Rays(
            data=rays.data,
            indices=rays.indices,
            path_length=rays.path_length,
            location=self,
            wavelength=rays.wavelength,
            phi_0=rays.phi_0
        )

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
        scan_rotation: Degrees = 0.,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx
        self.scan_rotation = scan_rotation  # converts to radians in setter

    @property
    def scan_rotation(self) -> Degrees:
        return np.rad2deg(self.scan_rotation_rad)

    @scan_rotation.setter
    def scan_rotation(self, val: Degrees):
        self.scan_rotation_rad: Radians = np.deg2rad(val)

    @property
    def scan_rotation_rad(self) -> Radians:
        return self._scan_rotation

    @scan_rotation_rad.setter
    def scan_rotation_rad(self, val: Radians):
        self._scan_rotation = val

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step_yx
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y
        if self.scan_rotation_rad != 0.:
            pos_r, pos_a = R2P(scan_position_x + scan_position_y * 1j)
            pos_c = P2R(pos_r, pos_a + self.scan_rotation_rad)
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
        self, z: float,
        random: bool = False,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
        wavelength: Optional[float] = 0.0,
        phi_0: Optional[float] = 0.0,
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx
        self.random = random

        # Check if both wavelength and acceleration_voltage are provided
        if wavelength is not None and phi_0 is not None:
            raise ValueError("Only one of 'wavelength' or 'phi_0' should be provided.")
        else:
            # Calculate wavelength if acceleration_voltage is provided
            if wavelength is None and phi_0 is not None:
                self.wavelength = calculate_wavelength(phi_0)
            else:
                self.wavelength = wavelength

            # Calculate acceleration_voltage if wavelength is provided
            if phi_0 is None and wavelength is not None:
                self.phi_0 = calculate_phi_0(wavelength)
            else:
                self.phi_0 = phi_0

        @staticmethod
        def gui_wrapper():
            from .gui import STEMSampleGUI
            return STEMSampleGUI

    @abc.abstractmethod
    def get_rays(self, num_rays: int) -> Rays:
        raise NotImplementedError

    def _make_rays(self, r: NDArray, indices: Optional[NDArray] = None) -> Rays:
        if indices is None:
            indices = np.arange(r.shape[1])
        r[1, :] += self.tilt_yx[1]
        r[3, :] += self.tilt_yx[0]
        return Rays(
            data=r,
            indices=indices,
            location=self,
            path_length=np.zeros((r.shape[1],)),
            wavelength=np.ones((r.shape[1],)) * self.wavelength,
            phi_0=np.ones((r.shape[1],)) * self.phi_0,
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
        wavelength: Optional[float] = 0.0,
        phi_0: Optional[float] = 0.0,
        radius: float = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name)
        self.radius = radius

        # Check if both wavelength and acceleration_voltage are provided
        if wavelength is not None and phi_0 is not None:
            raise ValueError("Only one of 'wavelength' or 'phi_0' should be provided.")
        else:
            # Calculate wavelength if acceleration_voltage is provided
            if wavelength is None and phi_0 is not None:
                self.wavelength = calculate_wavelength(phi_0)
            else:
                self.wavelength = wavelength

            # Calculate acceleration_voltage if wavelength is provided
            if phi_0 is None and wavelength is not None:
                self.phi_0 = calculate_phi_0(wavelength)
            else:
                self.phi_0 = phi_0

    def get_rays(self, num_rays: int) -> Rays:
        r, _ = make_beam(num_rays, self.radius, 'circular_beam')
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


class PointBeam(Source):
    def __init__(
        self,
        z: float,
        random: bool,
        wavelength: Optional[float] = None,
        phi_0: Optional[float] = None,
        semi_angle: Optional[float] = 0.,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.semi_angle = semi_angle
        self.tilt_yx = tilt_yx
        self.random = random

        # Check if both wavelength and acceleration_voltage are provided
        if wavelength is not None and phi_0 is not None:
            raise ValueError("Only one of 'wavelength' or 'phi_0' should be provided.")
        else:
            # Calculate wavelength if acceleration_voltage is provided
            if wavelength is None and phi_0 is not None:
                self.wavelength = calculate_wavelength(phi_0)
            else:
                self.wavelength = wavelength

            # Calculate acceleration_voltage if wavelength is provided
            if phi_0 is None and wavelength is not None:
                self.phi_0 = calculate_phi_0(wavelength)
            else:
                self.phi_0 = phi_0

    def get_rays(self, num_rays: int) -> Rays:
        r = np.zeros((5, num_rays))
        if self.random:
            r[1, :] = np.random.uniform(
                -self.semi_angle, self.semi_angle, size=num_rays
            )
            r[3, :] = np.random.uniform(
                -self.semi_angle, self.semi_angle, size=num_rays
            )
        else:
            r, _ = make_beam(num_rays, self.semi_angle, 'point_beam')

        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class XPointBeam(PointBeam):
    def get_rays(self, num_rays: int) -> Rays:
        r = np.zeros((5, num_rays))
        if self.random:
            r[1, :] = np.random.uniform(
                -self.semi_angle, self.semi_angle, size=num_rays
            )
        else:
            r[1, :] = np.linspace(
                -self.semi_angle, self.semi_angle, num=num_rays, endpoint=True
            )
        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class Detector(Component):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        rotation: Degrees = 0.,
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
        self.rotation = rotation  # converts to radians in setter
        self.flip_y = flip_y
        self.center = center

    @property
    def rotation(self) -> Degrees:
        return np.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = np.deg2rad(val)

    @property
    def rotation_rad(self) -> Radians:
        return self._rotation

    @rotation_rad.setter
    def rotation_rad(self, val: Radians):
        self._rotation = val

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
        coord: complex = P2R(mag, angle + self.rotation_rad)
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

    def get_image_intensity(self, rays: Rays) -> NDArray:

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
            dtype=np.complex128,
        )

        # Compute the complex wavefronts for each ray
        wavefronts = np.exp(-1j * 2 * np.pi / rays.wavelength * rays.path_length)

        # Use only the wavefronts for rays that hit the detector
        valid_wavefronts = wavefronts[mask]

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
            valid_wavefronts,
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
        offset: float = 0.,
        rotation: Degrees = 0.,
        deflection: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        offset: float
            Offset distance of biprism line
        rotation: float
            Rotation of biprism in z-plane
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.deflection = deflection
        self.offset = offset
        self.rotation = rotation

    @property
    def rotation(self) -> Degrees:
        return np.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = np.deg2rad(val)

    @property
    def rotation_rad(self) -> Radians:
        return self._rotation

    @rotation_rad.setter
    def rotation_rad(self, val: Radians):
        self._rotation = val

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        deflection = self.deflection
        offset = self.offset
        rot = self.rotation_rad
        pos_x = rays.x
        pos_y = rays.y
        rays_v = np.array([pos_x, pos_y]).T

        biprism_loc_v = np.array([offset*np.cos(rot), offset*np.sin(rot)])

        biprism_v = np.array([-np.sin(rot), np.cos(rot)])
        biprism_v /= np.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        rejection = np.dot(rays_v_centred, biprism_v)
        rejection /= np.dot(biprism_v, biprism_v)
        rejection = rejection[:, np.newaxis]
        rejection = rejection * biprism_v[np.newaxis, :]
        rejection /= np.linalg.norm(rejection, axis=1, keepdims=True)

        # dot_product = np.dot(rays_v_centred, biprism_v) / np.dot(biprism_v, biprism_v)
        # projection = np.outer(dot_product, biprism_v)

        # rejection = projection - rays_v_centred
        # rejection_norm = rejection/np.linalg.norm(rejection, axis=1, keepdims=True)
        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = np.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        rays.data[1] += xdeflection_mag*deflection
        rays.data[3] += ydeflection_mag*deflection

        yield Rays(
            data=rays.data,
            indices=rays.indices,
            path_length=(
                rays.path_length
                + xdeflection_mag * deflection * rays.data[0]
                + ydeflection_mag * deflection * rays.data[2]
            ),
            location=self,
            wavelength=rays.wavelength,
            phi_0=rays.phi_0
        )

    @staticmethod
    def gui_wrapper():
        from .gui import BiprismGUI
        return BiprismGUI


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
