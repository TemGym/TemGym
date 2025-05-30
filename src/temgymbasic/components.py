import abc
from typing import (
    Generator, Tuple, Optional, Type,
    TYPE_CHECKING, Sequence, Union
)
from dataclasses import dataclass, astuple


import numpy as np
from numpy.typing import NDArray  # Assuming np is an alias for numpy
from scipy.constants import c, e, m_e

from . import (
    UsageError,
    InvalidModelError,
    PositiveFloat,
    NonNegativeFloat,
    Radians,
    Degrees,
    BackendT
)
from .aber import opd, aber_x_aber_y
from .gbd import (
    differential_matrix,
    calculate_Qinv,
    calculate_Qpinv,
    propagate_misaligned_gaussian
)
from .rays import Rays, GaussianRays
from .utils import (
    get_array_module,
    P2R, R2P,
    circular_beam,
    gauss_beam_rayset,
    point_beam,
    calculate_direction_cosines,
    calculate_wavelength,
    get_array_from_device
)

if TYPE_CHECKING:
    from .gui import ComponentGUIWrapper

# Defining epsilon constant from page 18 of principles of electron optics 2017, Volume 1.
EPSILON = abs(e)/(2*m_e*c**2)


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


@dataclass
class LensAberrations:
    spherical: float
    coma: float
    astigmatism: float
    field_curvature: float
    distortion: float

    def __iter__(self):
        return iter(astuple(self))

    def BFCDE(self):
        return (
            self.spherical,
            self.coma,
            self.astigmatism,
            self.field_curvature,
            self.distortion,
        )

    def nonzero(self):
        return any(s != 0 for s in self)


class Lens(Component):
    def __init__(
        self,
        z: float,
        f: Optional[float] = None,
        m: Optional[float] = None,
        z1: Optional[float] = None,
        z2: Optional[float] = None,
        aber_coeffs: Optional[LensAberrations] = None,
        name: Optional[str] = None
    ):
        super().__init__(z=z, name=name)

        self.aber_coeffs = aber_coeffs

        if f is not None and (m is None and z1 is None and z2 is None):
            m = -1.0
        self._z1, self._z2, self._f, self._m = self._calculate_lens_parameters(z1, z2, f, m)

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m: float):
        self._m = m

    @property
    def z1(self) -> float:
        return self._z1

    @z1.setter
    def z1(self, z1: float):
        self._z1 = z1

    @property
    def z2(self) -> float:
        return self._z2

    @z2.setter
    def z2(self, z2: float):
        self._z2 = z2

    @property
    def ffp(self) -> float:
        return self.z - abs(self._f)

    def _calculate_lens_parameters(self, z1, z2, f, m, xp=np):

        if (f is not None and m is not None) and (z1 is None and z2 is None):
            # m <1e-10 means that the object is very far away, the lens focuses the beam to a point.
            if np.abs(m) <= 1e-10:
                z1 = -1e10
                z2 = f
            # m >1e-10 means that the image is formed very far away, the lens collimates the beam.
            elif np.abs(m) > 1e10:
                z1 = -f
                z2 = 1e10
            else:
                z1 = f * (1/m - 1)
                z2 = f * (1 - m)
        elif (z1 is not None and z2 is not None) and (f is None and m is None):
            if np.abs(z1) < 1e-10:
                z2 = 1e10
                f = -1e10
                m = 1e10
            elif np.abs(z2) < 1e-10:
                z1 = 1e10
                f = 1e10
                m = 0.0
            else:
                f = (1 / z2 - 1 / z1) ** -1
                m = z2 / z1
        elif (f is not None and z1 is not None) and (z2 is None and m is None):
            if np.abs(z1) < 1e-10:
                z2 = 1e10
                m = 1e10
            elif np.abs(f) < 1e-10:
                z2 = 1e10
                m = 1e10
            else:
                z2 = (1 / f + 1 / z1) ** -1
                m = z2 / z1
        else:
            raise InvalidModelError("Lens must have defined: f and m, or z1 and z2, or f and z1")

        return z1, z2, f, m

    @staticmethod
    def lens_matrix(f, xp=np):
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

        xp = rays.xp

        # Rays at lens plane
        u1 = rays.x_central
        v1 = rays.y_central

        du1 = rays.dx_central
        dv1 = rays.dy_central

        rays.data = xp.matmul(self.lens_matrix(xp.float64(self.f), xp=xp), rays.data)
        rays.path_length -= (rays.x ** 2 + rays.y ** 2) / (2 * xp.float64(self.f))

        if self.aber_coeffs is not None and self.aber_coeffs.nonzero():

            coeffs = self.aber_coeffs

            M = self._m
            z2 = self._z2

            # Rays at object plane
            x1 = u1 + du1 * self._z1
            y1 = v1 + dv1 * self._z1

            psi_a = np.arctan2(v1, u1)
            psi_o = np.arctan2(y1, x1)
            psi = psi_a - psi_o

            # Calculate the aberration in x and y
            # (Approximate R' as the reference sphere radius at image side)
            eps_x, eps_y = aber_x_aber_y(u1, v1, x1, y1, coeffs, z2, M, xp=xp)

            x2 = u1 + rays.dx_central * z2
            y2 = v1 + rays.dy_central * z2

            x2_aber = x2 + eps_x
            y2_aber = y2 + eps_y

            nx, ny, nz = calculate_direction_cosines(x2, y2, z2, u1, v1, 0.0, xp=xp)
            nx_aber, ny_aber, nz_aber = calculate_direction_cosines(x2_aber, y2_aber, z2,
                                                                    u1, v1, 0.0, xp=xp)
            W = opd(u1, v1, x1, y1, psi, coeffs, z2, M, xp=xp)

            dx_slope = nx_aber / nz_aber - nx / nz
            dy_slope = ny_aber / nz_aber - ny / nz

            if isinstance(rays, GaussianRays):
                rays.dx += xp.repeat(dx_slope, 5)
                rays.dy += xp.repeat(dy_slope, 5)
                rays.path_length += xp.repeat(W, 5)
            else:
                rays.dx += dx_slope
                rays.dy += dy_slope
                rays.path_length += W

        # Just straightforward matrix multiplication
        yield rays.new_with(
            data=rays.data,
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import BasicLensGUI
        return BasicLensGUI


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
        self,
        z: float,
        tilt_yx: Tuple[float, float] = (0., 0.),
        centre_yx: Tuple[float, float] = (0., 0.),
        voltage: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx
        self.centre_yx = centre_yx
        self.phi_0 = voltage
        self.random = False

    @property
    def voltage(self):
        return self.phi_0

    @abc.abstractmethod
    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        raise NotImplementedError

    def set_centre(self, centre_yx: tuple[float, float]):
        self.centre_yx = centre_yx

    def _rays_args(self, r: NDArray, backend: BackendT = 'cpu'):
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

        r = get_array_module(backend).asarray(r)

        return dict(
            data=r,
            location=self,
            wavelength=wavelength,
        )

    def _make_rays(self, r: NDArray, backend: BackendT = 'cpu') -> Rays:
        return Rays.new(
            **self._rays_args(r, backend=backend),
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
        centre_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name, voltage=voltage, centre_yx=centre_yx)
        self.radius = radius

    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        r = circular_beam(num_rays, self.radius,
                          random=random if random is not None else self.random)
        return self._make_rays(r, backend=backend)

    @staticmethod
    def gui_wrapper():
        from .gui import ParallelBeamGUI
        return ParallelBeamGUI


class XAxialBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
        r = np.zeros((5, num_rays))
        r[0, :] = np.random.uniform(
            -self.radius, self.radius, size=num_rays
        )
        return self._make_rays(r)


class RadialSpikesBeam(ParallelBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
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
        voltage: Optional[float] = None,
        semi_angle: Optional[float] = 0.,
        tilt_yx: Tuple[float, float] = (0., 0.),
        centre_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z, voltage=voltage, centre_yx=centre_yx)
        self.semi_angle = semi_angle
        self.tilt_yx = tilt_yx
        self.centre_yx = centre_yx

    def get_rays(self, num_rays: int, random: Optional[bool] = None, backend='cpu') -> Rays:
        r = point_beam(num_rays, self.semi_angle,
                       random=random if random is not None else self.random)
        return self._make_rays(r, backend=backend)

    @staticmethod
    def gui_wrapper():
        from .gui import PointBeamGUI
        return PointBeamGUI


class XPointBeam(PointBeam):
    def get_rays(self, num_rays: int, random: bool = False, backend='cpu') -> Rays:
        r = np.zeros((5, num_rays))
        r[1, :] = np.linspace(
            -self.semi_angle, self.semi_angle, num=num_rays, endpoint=True
            )
        return self._make_rays(r, backend=backend)

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
        centre_yx: Tuple[float, float] = (0., 0.),
        semi_angle: Optional[float] = 0.,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z, voltage=voltage)
        self.wo = wo
        self.radius = radius
        self.tilt_yx = tilt_yx
        self.semi_angle = semi_angle
        self.centre_yx = centre_yx

    def get_rays(
        self,
        num_rays: int,
        random: Optional[bool] = None,
        backend='cpu',
    ) -> Rays:
        wavelength = calculate_wavelength(self.voltage)

        # if random:
        #     raise NotImplementedError
        # else:
        xp = get_array_module(backend)
        r = gauss_beam_rayset(
            num_rays,
            self.radius,
            self.semi_angle,
            self.wo,
            wavelength,
            random=random if random is not None else self.random,
            xp=xp,
            centre_yx=self.centre_yx,
        )

        return self._make_rays(r, backend=backend)

    def _make_rays(self, r: NDArray, backend: BackendT = 'cpu') -> Rays:
        return GaussianRays.new(
            **self._rays_args(r, backend=backend),
            wo=self.wo,
        )

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
        self.buffer = None

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

    def get_det_coords_for_gauss_rays(self, xEnd, yEnd, xp=np):
        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = xp.linspace(-det_size_y / 2, det_size_y / 2, self.shape[0], dtype=xEnd.dtype)
        y_det = xp.linspace(-det_size_x / 2, det_size_x / 2, self.shape[1], dtype=yEnd.dtype)
        x, y = xp.meshgrid(x_det, y_det)

        r = xp.stack((x, y), axis=-1).reshape(-1, 2)

        return r

    def image_dtype(
        self,
        interfere: bool
    ):
        if interfere:
            return np.complex128
            return np.complex64
        else:
            return np.int32

    def get_image(
        self,
        rays: Union[Rays, Sequence[Rays]],
        interfere: bool = True,
        out: Optional[NDArray] = None,
    ) -> NDArray:

        if not isinstance(rays, Sequence):
            rays = [rays]
        assert len(rays) > 0

        xp = rays[0].xp

        image_dtype = self.image_dtype(interfere)

        if out is None:
            out = xp.zeros(
                self.shape,
                dtype=image_dtype,
            )
        else:
            assert out.dtype == image_dtype
            assert out.shape == self.shape

        if interfere and isinstance(rays[0], GaussianRays):
            if len(rays) < 2:
                raise UsageError(
                    "GaussianRays must have two sets of rays to calculate interference"
                )

            self._gaussian_beam_summation(rays, out=out)

        else:
            self._basic_beam_summation(rays, interfere, out=out)
        return get_array_from_device(out)

    def _basic_beam_summation(
        self,
        rays: tuple[Rays],
        interfere: bool,
        out: Optional[NDArray] = None,
    ) -> NDArray:

        rays = rays[-1]
        xp = rays.xp

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

        # Add rays as complex numbers if interference is enabled,
        # or just add rays as counts otherwise.
        if interfere:
            # If we are doing interference, we add a complex number representing
            # the phase of the ray for now to each pixel.
            # Amplitude is 1.0 for now for each complex ray.
            # Need to implement triangulation of wavefront
            # to properly track amplitude of each ray.
            wavefronts = 1.0 * xp.exp(-1j * (2 * xp.pi / rays.wavelength) * rays.path_length)
            valid_wavefronts = wavefronts[mask]
        else:
            # If we are not doing interference, we simply add 1 to each pixel that a ray hits
            valid_wavefronts = 1

        # Get a flattened list pixel indices where rays have hit
        flat_icds = xp.ravel_multi_index(
                [
                    pixel_coords_y[mask],
                    pixel_coords_x[mask],
                ],
                out.shape
            )

        self.sum_rays_on_detector(out, flat_icds, valid_wavefronts, xp=xp)

    def _gaussian_beam_summation(
        self,
        rays: tuple[GaussianRays],
        out: Optional[NDArray] = None,
    ) -> NDArray:

        rays_start = rays[0]
        rays_end = rays[-1]

        xp = rays_end.xp
        float_dtype = out.real.dtype.type

        wo = rays_start.wo
        wavelength = rays_start.wavelength

        div = rays_start.wavelength / (xp.pi * wo)
        k = float_dtype(2 * xp.pi / wavelength)

        z0 = float_dtype(rays_start.z)
        z_r = float_dtype(xp.pi * wo ** 2 / wavelength)

        dPx = float_dtype(wo)
        dPy = float_dtype(wo)
        dHx = float_dtype(div)
        dHy = float_dtype(div)

        # rays layout
        # [5, n_rays] where n_rays = 5 * n_gauss
        # so rays.reshape(5, 5, -1)
        #  => [(x, dx, y, dy, 1), (*gauss_beams), n_gauss]

        n_gauss = rays_end.num // 5
        path_length = rays_end.path_length[0::5].astype(float_dtype)

        rayset1 = xp.moveaxis(
            rays_end.data[0:4, :].reshape(4, n_gauss, 5),
            -1,
            0,
        )
        rayset1 = rayset1.astype(float_dtype)

        # rayset1 layout
        # [5g, (x, dx, y, dy), n_gauss]

        A, B, C, D = differential_matrix(rayset1, dPx, dPy, dHx, dHy, xp=xp)

        # A, B, C, D all have shape (n_gauss, 2, 2)
        Qinv = calculate_Qinv(z0, z_r, wo, wavelength, n_gauss, xp=xp)
        # matmul, addition and mat inverse inside
        # on operands with form (n_gauss, 2, 2)
        # matmul broadcasts in the last two indices
        # inv can be broadcast with xp.linalg.inv last 2 idcs
        # if all inputs are stacked in the first dim
        Qpinv = calculate_Qpinv(A, B, C, D, Qinv, xp=xp)

        px1m = rays_start.data[0, 0::5]  # x that central ray leaves at
        py1m = rays_start.data[2, 0::5]  # y that central ray leaves at
        thetax1m = rays_start.data[1, 0::5]  # slope that central ray arrives at
        thetay1m = rays_start.data[3, 0::5]  # slope that central ray arrives at

        p1m = xp.array([px1m, py1m]).T.astype(float_dtype)
        theta1m = xp.array([thetax1m, thetay1m]).T.astype(float_dtype)

        xEnd, yEnd = rayset1[0, 0], rayset1[0, 2]

        # central beam final x , y coords
        det_coords = self.get_det_coords_for_gauss_rays(xEnd, yEnd, xp=xp)

        propagate_misaligned_gaussian(
            Qinv, Qpinv, det_coords, p1m,
            theta1m, k, A, B, path_length, out.ravel(), xp=xp
        )

    def sum_rays_on_detector(self,
                             out: NDArray,
                             flat_icds: NDArray,
                             valid_wavefronts: NDArray,
                             xp=np):

        # Increment at each pixel for each ray that hits
        if xp == np:
            np.add.at(
                out.ravel(),
                flat_icds,
                valid_wavefronts,
            )
        else:
            if xp.iscomplexobj(out):
                # Separate the real and imaginary parts
                real_out = out.real
                imag_out = out.imag

                # Perform the addition separately for real and imaginary parts
                xp.add.at(real_out.reshape(-1), flat_icds, valid_wavefronts.real)
                xp.add.at(imag_out.reshape(-1), flat_icds, valid_wavefronts.imag)

                # Combine the real and imaginary parts back into the out array
                out = real_out + 1j * imag_out
            else:
                # Perform the addition directly for non-complex arrays
                xp.add.at(out.reshape(-1), flat_icds, valid_wavefronts)

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
    ):
        super().__init__(
            z=z,
            pixel_size=pixel_size,
            shape=shape,
            rotation=rotation,
            flip_y=flip_y,
            center=center,
            name=name,
        )
        self.buffer = None
        self.buffer_length = buffer_length

    @property
    def buffer_frame_shape(self) -> Optional[Tuple[int, int]]:
        if self.buffer is None:
            return
        return self.buffer.shape[1:]

    def delete_buffer(self):
        self.buffer = None

    def reset_buffer(self, rays: Rays):
        xp = rays.xp
        self.buffer = xp.zeros(
            (self.buffer_length, *self.shape),
            dtype=self.image_dtype(rays),
        )
        # the next index to write into
        self.buffer_idx = 0

    def get_image(self, rays: Rays) -> NDArray:
        if self.buffer is None or self.buffer_frame_shape != self.shape:
            if len(rays) > 1:
                self.reset_buffer(rays[0])
            else:
                self.reset_buffer(rays)
        else:
            self.buffer[self.buffer_idx] = 0.

        super().get_image(
            rays,
            out=self.buffer[self.buffer_idx],
        )

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_length
        # Convert always to array on cpu device.
        return get_array_from_device(self.buffer.sum(axis=0))

    @staticmethod
    def gui_wrapper():
        from .gui import AccumulatingDetectorGUI
        return AccumulatingDetectorGUI


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
    def deflector_matrix(def_x, def_y, xp=np):
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
        xp = rays.xp
        yield rays.new_with(
            data=xp.matmul(
                self.deflector_matrix(xp.float64(self.defx), xp.float64(self.defy), xp=xp),
                rays.data,
            ),
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import DeflectorGUI
        return DeflectorGUI


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

        xp = rays.xp
        deflection = xp.array(self.deflection)
        offset = xp.array(self.offset)
        rot = xp.array(self.rotation_rad)
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
        xp = rays.xp
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


class PotentialSample(Sample):
    def __init__(
        self,
        z: float,
        potential,
        Ex,
        Ey,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)

        # We're renaming here some terms to be closer to the math in Hawkes
        # Not sure if this is recommended or breaks any convetions
        self.phi = potential
        self.dphi_dx = Ex
        self.dphi_dy = Ey

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:

        xp = rays.xp
        # See Chapter 2 & 3 of principles of electron optics 2017 Vol 1 for more info
        rho = xp.sqrt(1 + rays.dx ** 2 + rays.dy ** 2)  # Equation 3.16
        phi_0_plus_phi = (rays.phi_0 + self.phi((rays.x, rays.y)))  # Part of Equation 2.18

        phi_hat = (phi_0_plus_phi) * (1 + EPSILON * (phi_0_plus_phi))  # Equation 2.18

        # Between Equation 2.22 & 2.23
        dphi_hat_dx = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dx((rays.x, rays.y))
        dphi_hat_dy = (1 + 2 * EPSILON * (phi_0_plus_phi)) * self.dphi_dy((rays.x, rays.y))

        # Perform deflection to ray in slope coordinates
        rays.dx += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dx  # Equation 3.22
        rays.dy += ((rho ** 2) / (2 * phi_hat)) * dphi_hat_dy  # Equation 3.22

        # Note here we are ignoring the Ez component (dphi/dz) of 3.22,
        # since we have modelled the potential of the atom in a plane
        # only, we won't have an Ez component (At least I don't think this is the case?
        # I could be completely wrong here though - it might actually have an effect.
        # But I'm not sure I can get an Ez from an infinitely thin slice.

        # Equation 5.16 & 5.17 & 3.16, where ds of 5.16 is replaced by ds/dz * dz,
        # where ds/dz = rho (See 3.16 and a little below it)
        rays.path_length += rho * xp.sqrt(phi_hat / rays.phi_0)

        # Currently the modifications are all inplace so we only need
        # to change the location, this should be made clearer
        yield rays.new_with(
            location=self,
        )

    @staticmethod
    def gui_wrapper():
        from .gui import SampleGUI
        return SampleGUI


class ProjectorLensSystem(Component):
    def __init__(
        self,
        first: Lens,
        second: Lens,
        magnification: float = -1.,
        name: Optional[str] = None,
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self.magnification = magnification

        self._first = first
        self._second = second

        self._validate_component()

        self.adjust_z2_and_z3_from_magnification(self.magnification)

    @classmethod
    def from_params(
        cls,
        z: float,
        z1: float,
        z2: float,
        z3: float,
        z4: float,
        distance: float = 0.1,
        name: Optional[str] = None
    ):
        return cls(
            Lens(
                z=z - distance / 2.,
                z1=z1,
                z2=z2,
            ),
            Lens(
                z=z + distance / 2.,
                z3=z3,
                z4=z4,
            ),
            name=name,
        )

    def _validate_component(self):
        if self.first.z >= self.second.z:
            raise InvalidModelError("First Projector Lens must be before second")

    @property
    def distance(self) -> float:
        return self._second.z - self._first.z

    @property
    def first(self) -> Lens:
        return self._first

    @property
    def second(self) -> Lens:
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

    def adjust_z2_and_z3_from_magnification(self, magnification):

        z1 = self.first.z1
        dz = self.distance
        z4 = self.second.z2
        z2 = (magnification * z1 * dz) / (magnification * z1 + z4)
        z3 = z2-dz

        self.first.z2 = z2
        self.second.z1 = z3
        self.first.f = 1/(1/z2 - 1/z1)
        self.second.f = 1/(1/z4 - 1/z3)

    @staticmethod
    def gui_wrapper():
        from .gui import ProjectorLensSystemGUI
        return ProjectorLensSystemGUI
