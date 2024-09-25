import abc
from typing import (
    Generator, Tuple, Optional, Type,
    TYPE_CHECKING
)

from .config import use_numpy

if use_numpy:
   import numpy as xp
else:
   import cupy as xp


from numpy.typing import NDArray  # Assuming xp is an alias for numpy
import warnings
import line_profiler

from . import (
    UsageError,
    InvalidModelError,
    PositiveFloat,
    NonNegativeFloat,
    Radians,
    Degrees,
)
from .aber import dopd_dx, dopd_dy, opd
from .gbd import (
    differential_matrix,
    calculate_Qinv,
    calculate_Qpinv,
    propagate_misaligned_gaussian
)
from .rays import Rays
from .utils import (
    P2R, R2P,
    circular_beam,
    fibonacci_beam_gauss_rayset,
    point_beam,
    calculate_direction_cosines,
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
        ...
        raise NotImplementedError

    @staticmethod
    def gui_wrapper() -> Optional[Type['ComponentGUIWrapper']]:
        return None


class Lens(Component):
    def __init__(self, z: float,
                 f: float,
                 name: Optional[str] = None):
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
        return xp.array(
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
        yield rays.new_with(
            data=xp.matmul(self.lens_matrix(self.f), rays.data),
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import LensGUI
        return LensGUI


class PerfectLens(Lens):
    def __init__(self, z: float,
                 f: float,
                 m: Optional[float] = None,
                 z1: Optional[Tuple[float]] = None,
                 z2: Optional[Tuple[float]] = None,
                 name: Optional[str] = None):
        super().__init__(name=name, z=z, f=f)
        self._f = f

        # Initial Numerical Aperture
        self.NA1 = 0.1
        self.NA2 = 0.1

        self._z1, self._z2, self._m = self.initalise_m_and_principal_planes(z1, z2, m, f)

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def ffp(self) -> float:
        return self.z - abs(self.f)

    def update_m_and_principal_planes(self, z1, z2, m):
        f = self._f
        self._z1, self._z2, self._m = self.initalise_m_and_principal_planes(z1, z2, m, f)

    def initalise_m_and_principal_planes(self, z1, z2, m, f):

        # If statements to decide how to define z1, z2 and magnification.
        # We check if magnification is too small or large,
        # and thus a finite-long conjugate approximation is applied
        if ((z1 or z2) is None) and (m is None):
            assert ('Must have either m defined, or both z1 and z2')

        if ((z1 and z2) is None) and (m is not None):
            if xp.abs(m) <= 1e-10:
                z1 = -1e10
                z2 = f
            elif xp.abs(m) > 1e10:
                z1 = -f
                z2 = 1e10
            else:
                z1 = f * (1/m - 1)
                z2 = f * (1 - m)
        elif (m is None) and ((z1 and z2) is not None):
            if xp.abs(z1) > 1e-10:
                m = z2 / z1
            elif z1 <= 1e-10:
                m = 1e10
        elif (
            (m is not None)
            and ((z1 and z2) is not None)
        ):
            warnings.warn("Overspecified magnification (m) and image and object planes (z1 and z2),\
                          the provided magnification is ignored")
            m = z2 / z1

        # If finite long conjugate approximation,
        # we need to set the signal that the numerical aperture is
        # 0.0, which is neccessary for later
        if xp.abs(z1) >= 1e10:
            z1 = 1e10 * (z1 / xp.abs(z1))
            self.NA1 = 0.0  # collimated input

        if xp.abs(z2) >= 1e10:
            z2 = 1e10 * (z2 / xp.abs(z2))
            self.NA2 = 0.0  # collimated input

        m = z2 / z1

        return z1, z2, m

    def get_exit_pupil_coords(self, rays):

        f = self._f
        m = self._m
        z1 = self._z1
        z2 = self._z2
        NA1 = self.NA1
        NA2 = self.NA2

        # Convert slope into direction cosines
        L1 = rays.dx / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
        M1 = rays.dy / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
        N1 = xp.sqrt(1 - L1 ** 2 - M1 ** 2)

        u1 = rays.x
        v1 = rays.y

        # Ray object point coordinates (x1, y1) on front conjugate plane
        if (self.NA1 == 0.0):
            x1 = (L1 / N1) * z1
            y1 = (M1 / N1) * z1
            r1_mag = xp.sqrt((x1 - u1) ** 2 + (y1 - v1) ** 2 + z1 ** 2)

            L1_est = -(x1 - u1) / r1_mag
            M1_est = -(y1 - v1) / r1_mag
        else:
            x1 = (L1 / N1) * z1 + u1
            y1 = (M1 / N1) * z1 + v1
            r1_mag = xp.sqrt((x1 - u1) ** 2 + (y1 - v1) ** 2 + z1 ** 2)

        # Principal Ray directions
        if (self.NA1 == 0.0):
            L1_p = rays.dx / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            M1_p = rays.dy / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            N1_p = 1 / xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)
            p1_mag = xp.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
        else:
            p1_mag = xp.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)

            # Obtain direction cosines of principal ray from second principal plane to image point
            L1_p = (x1 / p1_mag) * z1 / xp.abs(z1)
            M1_p = (y1 / p1_mag) * z1 / xp.abs(z1)
            N1_p = xp.sqrt(1 - L1_p ** 2 - M1_p ** 2)

        # Coordinates in image plane or focal plane
        if xp.abs(m) <= 1.0:
            x2 = z2 * (L1_p / N1_p)
            y2 = z2 * (M1_p / N1_p)

            p2_mag = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)
            L2_p = (x2 / p2_mag) * (z2 / xp.abs(z2))
            M2_p = (y2 / p2_mag) * (z2 / xp.abs(z2))
            N2_p = xp.sqrt(1 - L2_p ** 2 - M2_p ** 2)
        else:
            a = x1 / z1
            b = y1 / z1
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p
            x2 = (L2_p / N2_p) * z2
            y2 = (M2_p / N2_p) * z2
            p2_mag = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

        # Calculation to back propagate to right hand side principal plane
        Cx = m * L2_p - L1_p
        Cy = m * M2_p - M1_p

        if (NA1 == 0.0):
            L2 = (L1_est + Cx) / m
            M2 = (M1_est + Cy) / m
            N2 = xp.sqrt(1 - L2 ** 2 - M2 ** 2)
        else:
            L2 = (L1 + Cx) / m
            M2 = (M1 + Cy) / m
            N2 = xp.sqrt(1 - L2 ** 2 - M2 ** 2)

        # We use a mask to find rays that have gone to the centre,
        # because we are not inputting one ray at a time, but a vector of rays.
        mask = xp.sqrt(u1 ** 2 + v1 ** 2) < 1e-7

        # Initialize the output arrays
        u2 = xp.zeros_like(u1)
        v2 = xp.zeros_like(v1)

        # Handle the case where the mask is true and NA2 = 0.0
        if NA2 == 0.0:
            a = -x1 / f
            b = -y1 / f
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p

            L2[mask] = L2_p[mask]
            M2[mask] = M2_p[mask]
            N2[mask] = N2_p[mask]
            u2[mask] = 0.0
            v2[mask] = 0.0

        # For the case where the mask is false, (rays are not going through the centre of the lens)
        not_mask = ~mask
        u2[not_mask] = -(L2[not_mask] / N2[not_mask]) * z2 + x2[not_mask]
        v2[not_mask] = -(M2[not_mask] / N2[not_mask]) * z2 + y2[not_mask]

        if NA2 == 0:
            a = -x1 / f
            b = -y1 / f
            N2_p = 1 / xp.sqrt(1 + a ** 2 + b ** 2)
            L2_p = a * N2_p
            M2_p = b * N2_p

            L2[not_mask] = L2_p[not_mask]
            M2[not_mask] = M2_p[not_mask]
            N2[not_mask] = N2_p[not_mask]

        # Calculate final distance from image/focal plane to point
        # ray leaves lens for optical path length
        r2_mag = xp.sqrt((x2 - u2) ** 2 + (y2 - v2) ** 2 + z2 ** 2)

        opl1 = r1_mag + r2_mag  # Ray opl
        opl0 = p1_mag + p2_mag  # Principal ray opl

        dopl = (opl0 - opl1)

        return x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        # x1 - object plane x coordinate of ray
        # y1 - object plane y coordinate of ray
        # u2 - exit pupil x coordinate of ray
        # v2 - exit pupil y coordinate of ray
        # x2 - image plane x coordinate of ray
        # y2 - image plane y coordinate of ray
        # L2, M2, N2 - direction cosines of the ray at the exit pupil
        # R - reference sphere radius
        # dopl - optical path length change
        x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl = self.get_exit_pupil_coords(rays)

        rays.x = u2
        rays.y = v2
        rays.dx = L2 / N2
        rays.dy = M2 / N2
        rays.path_length += dopl

        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import PerfectLensGUI
        return PerfectLensGUI


class AberratedLens(PerfectLens):
    def __init__(self, z: float,
                 f: float,
                 m: Optional[float] = None,
                 z1: Optional[Tuple[float]] = None,
                 z2: Optional[Tuple[float]] = None,
                 name: Optional[str] = None,
                 coeffs: Tuple = [0, 0, 0, 0, 0]):  # 5 aberration coefficients (C, K, A, D, F)

        super().__init__(z=z, f=f, m=m, z1=z1, z2=z2, name=name)
        self.coeffs = coeffs

        self._f = f

        # Initial Numerical Aperture
        self.NA1 = 0.1
        self.NA2 = 0.1

        self._z1, self._z2, self._m = self.initalise_m_and_principal_planes(z1, z2, m, f)

    def step(self, rays: Rays) -> Generator[Rays, None, None]:
        # # Call the step function of the parent class
        # yield from super().step(rays)

        z2 = self._z2
        x1, y1, u1, v1, u2, v2, x2, y2, L2, M2, N2, dopl = self.get_exit_pupil_coords(rays)

        coeffs = self.coeffs

        # Reference sphere radius
        R = xp.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

        # Coordinates of reference sphere
        u2_circle = x2 - L2 * R
        v2_circle = y2 - M2 * R
        z2_circle = z2 - N2 * R

        # Height of object point from the optical axis
        h = xp.sqrt(x1 ** 2 + y1 ** 2)

        # Calculate the aberration in x and y (Approximate)
        eps_x = -dopd_dx(u1, v1, h, coeffs) * z2
        eps_y = -dopd_dy(u1, v1, h, coeffs) * z2

        W = opd(u1, v1, h, coeffs)

        # Get aberration direction cosines - remember the aberrated rays must
        # go through the same point on the reference sphere
        # as the perfect rays
        nx, ny, nz = calculate_direction_cosines(x2 + eps_x, y2 + eps_y, z2,
                                                 u2_circle, v2_circle, z2_circle)

        # Calculate the new aberrated ray coordinates in the image plane
        x2_aber = x2 + eps_x
        y2_aber = y2 + eps_y

        # Calculate the new aberrated ray coordinates in the exit pupil plane
        u2_aber = x2_aber - nx / nz * (z2)
        v2_aber = y2_aber - ny / nz * (z2)

        # u2_aber = -nx / nz * (phi_zn) + phi_xn
        # v2_aber = -ny / nz * (phi_zn) + phi_yn

        rays.path_length += W

        rays.x = u2_aber
        rays.y = v2_aber
        rays.dx = nx / nz
        rays.dy = ny / nz

        # Return the modified rays
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import AberratedLensGUI
        return AberratedLensGUI


class Sample(Component):
    def __init__(self, z: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
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
        return xp.rad2deg(self.scan_rotation_rad)

    @scan_rotation.setter
    def scan_rotation(self, val: Degrees):
        self.scan_rotation_rad: Radians = xp.deg2rad(val)

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
        tilt_yx: Tuple[float, float] = (0., 0.),
        voltage: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx
        self.centre_yx: tuple[float, float] = (0., 0.)
        self.phi_0 = voltage

    @property
    def voltage(self):
        return self.phi_0

    @abc.abstractmethod
    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        raise NotImplementedError

    def set_centre(self, centre_yx: tuple[float, float]):
        self.centre_yx = centre_yx

    def _make_rays(self, r: NDArray, wo: Optional[float] = None) -> Rays:
        # Add beam tilt (if any)
        if self.tilt_yx[1] != 0:
            r[1, :] += self.tilt_yx[1]
        if self.tilt_yx[0] != 0:
            r[3, :] += self.tilt_yx[0]

        # Add beam shift (if any)
        if self.centre_yx[1] != 0:
            r[0, :] += self.centre_yx[1]
        if self.centre_yx[0] != 0:
            r[2, :] += self.centre_yx[0]

        wavelength = None
        if self.phi_0 is not None:
            wavelength = calculate_wavelength(self.phi_0)

        return Rays.new(
            data=r,
            location=self,
            wavelength=wavelength,
            wo=wo,
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Source has no effect after get_rays was called
        yield rays.new_with(
            location=self
        )


class ParallelBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float,
        voltage: Optional[float] = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name, voltage=voltage)
        self.radius = radius

    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        r = circular_beam(num_rays, self.radius, random=random)
        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import ParallelBeamGUI
        return ParallelBeamGUI


class XAxialBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        r = xp.zeros((5, num_rays))
        r[0, :] = xp.random.uniform(
            -self.radius, self.radius, size=num_rays
        )
        return self._make_rays(r)


class RadialSpikesBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        xvals = xp.linspace(
            0., self.radius, num=num_rays // 4, endpoint=True
        )
        yvals = xp.zeros_like(xvals)
        origin_c = xvals + yvals * 1j

        orad, oang = R2P(origin_c)
        radius1 = P2R(orad * 0.75, oang + xp.pi * 0.4)
        radius2 = P2R(orad * 0.5, oang + xp.pi * 0.8)
        radius3 = P2R(orad * 0.25, oang + xp.pi * 1.2)
        r_c = xp.concatenate((origin_c, radius1, radius2, radius3))

        r = xp.zeros((5, r_c.size))
        r[0, :] = r_c.real
        r[2, :] = r_c.imag
        return self._make_rays(r)


class PointBeam(Source):
    def __init__(
        self,
        z: float,
        voltage: Optional[float] = None,
        semi_angle: Optional[float] = 0.,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z, voltage=voltage)
        self.semi_angle = semi_angle
        self.tilt_yx = tilt_yx

    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        r = point_beam(num_rays, self.semi_angle, random=random)
        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class XPointBeam(PointBeam):
    def get_rays(self, num_rays: int, random: bool = False) -> Rays:
        r = xp.zeros((5, num_rays))
        r[1, :] = xp.linspace(
            -self.semi_angle, self.semi_angle, num=num_rays, endpoint=True
            )
        return self._make_rays(r)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class GaussBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float,
        wo: float,
        voltage: Optional[float] = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z, voltage=voltage)
        self.wo = wo
        self.radius = radius
        self.tilt_yx = tilt_yx

    def get_rays(self, num_rays: int, beam_type: str = 'fibonacci', random: str = 'False') -> Rays:
        wavelength = calculate_wavelength(self.voltage)

        if beam_type == 'fibonacci':
            r = fibonacci_beam_gauss_rayset(num_rays, self.radius, self.wo,
                                           wavelength)
        else:
            raise NotImplementedError
        return self._make_rays(r, self.wo)

    @staticmethod
    def gui_wrapper():
        from .gui import GaussBeamGUI
        return GaussBeamGUI


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
        interference: Optional[str] = 'gauss'
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
        self.interference = interference

    @property
    def rotation(self) -> Degrees:
        return xp.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = xp.deg2rad(val)

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
        yield rays.new_with(
            location=self
        )

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def get_det_coords_for_gauss_rays(self, xEnd, yEnd):
        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = xp.linspace(-det_size_y / 2, det_size_y / 2, self.shape[0])
        y_det = xp.linspace(-det_size_x / 2, det_size_x / 2, self.shape[1])
        x, y = xp.meshgrid(x_det, y_det)

        r = xp.stack((x, y), axis=-1).reshape(-1, 2)
        endpoints = xp.stack((xEnd, yEnd), axis=-1)
        # r = xp.broadcast_to(r, [n_rays, *r.shape])
        # r = xp.swapaxes(r, 0, 1)
        # has form (n_px, n_gauss, 2:[x, y])]
        # this entire section can be optimised !!!
        return r[:, xp.newaxis, :] - endpoints[xp.newaxis, ...]

    def get_image(
        self,
        rays: Rays,
        interference: str = None,
        out: Optional[NDArray] = None
    ) -> NDArray:

        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = self.on_grid(rays, as_int=True)
        sy, sx = self.shape
        mask = xp.logical_and(
            xp.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            xp.logical_and(
                0 <= pixel_coords_x,
                pixel_coords_x < sx
            )
        )

        if interference == 'ray':
            # If we are doing interference, we add a complex number representing
            # the phase of the ray for now to each pixel.
            # Amplitude is 1.0 for now for each complex ray.
            wavefronts = 1.0 * xp.exp(-1j * (2 * xp.pi / rays.wavelength) * rays.path_length)
            valid_wavefronts = wavefronts[mask]
            image_dtype = valid_wavefronts.dtype
        elif interference == 'gauss':
            image_dtype = xp.complex128
        elif interference is None:
            # If we are not doing interference, we simply add 1 to each pixel that a ray hits
            valid_wavefronts = 1
            image_dtype = type(valid_wavefronts)

        if out is None:
            out = xp.zeros(
                self.shape,
                dtype=image_dtype,
            )
        else:
            assert out.dtype == image_dtype
            assert out.shape == self.shape

        if interference == 'gauss':
            self.get_gauss_image(rays, out)
        else:
            flat_icds = xp.ravel_multi_index(
                    [
                        pixel_coords_y[mask],
                        pixel_coords_x[mask],
                    ],
                    out.shape
                )
            
            # Increment at each pixel for each ray that hits
            if use_numpy:
                xp.add.at(
                    out.ravel(),
                    flat_icds,
                    valid_wavefronts,
                )
            else:
                
                # Split the real and imaginary parts of the out array for cupy compatibility with add.at
                real_out = out.real
                imag_out = out.imag

                # Perform the addition separately for real and imaginary parts
                xp.add.at(real_out.ravel(), flat_icds, valid_wavefronts.real)
                xp.add.at(imag_out.ravel(), flat_icds, valid_wavefronts.imag)

                # Combine the real and imaginary parts back into the out array
                out = real_out + 1j * imag_out

        return out

    @line_profiler.profile
    def get_gauss_image(
        self,
        rays: Rays,
        out: Optional[NDArray] = None
    ) -> NDArray:

        wo = rays.wo
        wavelength = rays.wavelength
        div = rays.wavelength / (xp.pi * wo)
        k = 2 * xp.pi / wavelength
        z_r = xp.pi * wo ** 2 / wavelength

        dPx = wo
        dPy = wo
        dHx = div
        dHy = div

        # rays layout
        # [5, n_rays] where n_rays = 5 * n_gauss
        # so rays.reshape(5, 5, -1)
        #  => [(x, dx, y, dy, 1), (*gauss_beams), n_gauss]

        n_gauss = rays.num // 5

        end_rays = rays.data[0:4, :].T
        path_length = rays.path_length[0::5]

        split_end_rays = xp.split(end_rays, n_gauss, axis=0)

        rayset1 = xp.stack(split_end_rays, axis=-1)

        # rayset1 layout
        # [5g, (x, dx, y, dy), n_gauss]

        # rayset1 = cp.array(rayset1)
        A, B, C, D = differential_matrix(rayset1, dPx, dPy, dHx, dHy)
        # A, B, C, D all have shape (n_gauss, 2, 2)
        Qinv = calculate_Qinv(z_r, n_gauss)
        # matmul, addition and mat inverse inside
        # on operands with form (n_gauss, 2, 2)
        # matmul broadcasts in the last two indices
        # inv can be broadcast with xp.linalg.inv last 2 idcs
        # if all inputs are stacked in the first dim
        Qpinv = calculate_Qpinv(A, B, C, D, Qinv)

        # det_coords = cp.array(det_coords)
        # p2m = cp.array(p2m)
        # path_length = cp.array(path_length)
        # k = cp.array(k)

        phi_x2m = rays.data[1, 0::5]  # slope that central ray arrives at
        phi_y2m = rays.data[3, 0::5]  # slope that central ray arrives at
        p2m = xp.array([phi_x2m, phi_y2m]).T

        xEnd, yEnd = rayset1[0, 0], rayset1[0, 2]
        # central beam final x , y coords
        det_coords = self.get_det_coords_for_gauss_rays(xEnd, yEnd)
        propagate_misaligned_gaussian(
            Qinv, Qpinv, det_coords,
            p2m, k, A, B, path_length, out.ravel()
        )

    @staticmethod
    def gui_wrapper():
        from .gui import DetectorGUI
        return DetectorGUI


class AccumulatingDetector(Detector):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        buffer_length: int,
        rotation: Degrees = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
        interference: Optional[str] = 'gauss'
    ):
        super().__init__(
            z=z,
            pixel_size=pixel_size,
            shape=shape,
            rotation=rotation,
            flip_y=flip_y,
            center=center,
            name=name,
            interference=interference,
        )
        self.buffer = None
        self.buffer_length = buffer_length

    @property
    def buffer_frame_shape(self) -> Optional[Tuple[int, int]]:
        if self.buffer is None:
            return
        return self.buffer.shape[1:]

    def reset_buffer(self, rays: Rays):
        self.buffer = xp.zeros(
            (self.buffer_length, *self.shape),
            dtype=xp.complex128,
        )
        # the next index to write into
        self.buffer_idx = 0

    def get_image(self, rays: Rays) -> NDArray:
        if self.buffer_frame_shape != self.shape:
            self.reset_buffer(rays)
        else:
            self.buffer[self.buffer_idx] = 0.

        super().get_image(
            rays,
            interference=self.interference,
            out=self.buffer[self.buffer_idx],
        )

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_length
        return self.buffer.sum(axis=0)


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

        return xp.array(
            [[1, 0, 0, 0,     0],
             [0, 1, 0, 0, def_x],
             [0, 0, 1, 0,     0],
             [0, 0, 0, 1, def_y],
             [0, 0, 0, 0,     1]],
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        yield rays.new_with(
            data=xp.matmul(
                self.deflector_matrix(self.defx, self.defy),
                rays.data,
            ),
            location=self,
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
            yield rays.new_with(
                location=(self, self.first)
            )
        rays = rays.propagate_to(self.second.entrance_z)
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays.new_with(
                location=(self, self.second)
            )

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
        in_zp = xp.asarray(in_zp)
        pt1_zp = xp.asarray(pt1_zp)
        pt2_zp = xp.asarray(pt2_zp)
        dp = pt1_zp - pt2_zp
        out_zp = xp.asarray(
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
        return xp.rad2deg(self.rotation_rad)

    @rotation.setter
    def rotation(self, val: Degrees):
        self.rotation_rad: Radians = xp.deg2rad(val)

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
        rays_v = xp.array([pos_x, pos_y]).T

        biprism_loc_v = xp.array([offset*xp.cos(rot), offset*xp.sin(rot)])

        biprism_v = xp.array([-xp.sin(rot), xp.cos(rot)])
        biprism_v /= xp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = xp.dot(rays_v_centred, biprism_v) / xp.dot(biprism_v, biprism_v)
        projection = xp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection/xp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = xp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        rays.dx += xdeflection_mag * deflection
        rays.dy += ydeflection_mag * deflection
        rays.path_length += (
            xdeflection_mag * deflection * rays.x
            + ydeflection_mag * deflection * rays.y
        )
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import BiprismGUI
        return BiprismGUI


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius: float,
        x: float = 0.,
        y: float = 0.,
        name: Optional[str] = None,
    ):
        '''
        An Aperture that lets through rays within a radius centered on (x, y)

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default 'Aperture'
        radius : float, optional
           The radius of the aperture
        x : int, optional
            X position of the centre of the aperture, by default 0
        y : int, optional
            Y position of the centre of the aperture, by default 0
        '''

        super().__init__(z, name)

        self.x = x
        self.y = y
        self.radius = radius

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x, pos_y = rays.x, rays.y
        distance = xp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        yield rays.with_mask(
            distance < self.radius,
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import ApertureGUI
        return ApertureGUI
