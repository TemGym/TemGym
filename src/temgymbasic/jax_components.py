import jax_dataclasses as jdc
import jax.numpy as jnp
from jax.numpy import ndarray as NDArray
from typing import (
    Tuple, Optional,  Sequence, Union
)

from .jax_ray import Ray, propagate, GaussianRay, ray_on_grid
from . import UsageError
from .jax_utils import R2P, P2R
from . import (
    Degrees,
)
from typing_extensions import TypeAlias

Radians: TypeAlias = jnp.float64  # type: ignore


@jdc.pytree_dataclass
class Lens:
    z: float
    focal_length: float

    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)

        new_matrix = jnp.array([x, y, new_dx, new_dy, jnp.ones_like(x)]).T

        return Ray(
            z=ray.z,
            matrix=new_matrix,
            amplitude=ray.amplitude,
            pathlength=pathlength,
            wavelength=ray.wavelength
        )


@jdc.pytree_dataclass
class Deflector:
    z: float
    def_x: float
    def_y: float

    def step(self, ray: Ray):

        x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        new_matrix = jnp.array([x, y, new_dx, new_dy, jnp.ones_like(x)]).T

        return Ray(
            z=ray.z,
            matrix=new_matrix,
            amplitude=ray.amplitude,
            pathlength=pathlength,
            wavelength=ray.wavelength
        )


@jdc.pytree_dataclass
class DoubleDeflector:
    z: float
    first: Deflector
    second: Deflector

    def step(self, ray: Ray):
        ray = self.first.step(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step, ray)
        ray = self.second.step(ray)

        return ray


@jdc.pytree_dataclass
class Aperture:
    z: float
    radius: float
    x: float = 0.
    y: float = 0.

    def step(self, ray: Ray):

        pos_x, pos_y, _, _ = ray.x, ray.y, ray.dx, ray.dy
        distance = jnp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        # This code evaluates to 1 if the ray is blocked already,
        # even if the new ray is inside the aperture,
        # evaluates to 1 if the ray was not blocked before and is now,
        # and evaluates to 0 if the ray was not blocked before and is NOT now.
        blocked = jnp.where(distance > self.radius, 1, ray.blocked)
        return Ray(
            z=ray.z,
            matrix=ray.matrix,
            amplitude=ray.amplitude,
            pathlength=ray.pathlength,
            wavelength=ray.wavelength,
            blocked=blocked
        )


@jdc.pytree_dataclass
class Biprism:
    z: float
    offset: float = 0.
    rotation: Degrees = 0.
    deflection: float = 0.

    def step(
        self, ray: Ray,
    ) -> Ray:

        pos_x, pos_y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

        deflection = self.deflection
        offset = self.offset
        rot = jnp.deg2rad(self.rotation)

        rays_v = jnp.array([pos_x, pos_y]).T

        biprism_loc_v = jnp.array([offset*jnp.cos(rot), offset*jnp.sin(rot)])

        biprism_v = jnp.array([-jnp.sin(rot), jnp.cos(rot)])
        biprism_v /= jnp.linalg.norm(biprism_v)

        rays_v_centred = rays_v - biprism_loc_v

        dot_product = jnp.dot(rays_v_centred, biprism_v) / jnp.dot(biprism_v, biprism_v)
        projection = jnp.outer(dot_product, biprism_v)

        rejection = rays_v_centred - projection
        rejection = rejection/jnp.linalg.norm(rejection, axis=1, keepdims=True)

        # If the ray position is located at [zero, zero], rejection_norm returns a nan,
        # so we convert it to a zero, zero.
        rejection = jnp.nan_to_num(rejection)

        xdeflection_mag = rejection[:, 0]
        ydeflection_mag = rejection[:, 1]

        # Using .squeeze is a crappy hack for now because the
        # dimensions wont match with jax without it.
        new_dx = (dx + xdeflection_mag * deflection).squeeze()
        new_dy = (dy + ydeflection_mag * deflection).squeeze()

        pathlength = ray.pathlength + (
            xdeflection_mag * deflection * pos_x + ydeflection_mag * deflection * pos_y
        )

        new_matrix = jnp.array([pos_x, pos_y, new_dx, new_dy, jnp.ones_like(pos_x)]).T

        return Ray(
            z=ray.z,
            matrix=new_matrix,
            amplitude=ray.amplitude,
            pathlength=pathlength,
            wavelength=ray.wavelength
        )


@jdc.pytree_dataclass
class Detector:
    z: jdc.Static[float]
    pixel_size: jdc.Static[float]
    shape: jdc.Static[Tuple[int, int]]
    rotation: jdc.Static[Degrees] = 0.
    flip_y: jdc.Static[bool] = False
    center: jdc.Static[Tuple[float, float]] = (0., 0.)

    def step(self, ray: Ray):
        return ray

    @property
    def rotation_deg(self) -> Degrees:  # type: ignore
        return jnp.rad2deg(self.rotation_rad)

    @rotation_deg.setter
    def rotation_deg(self, val: Degrees):
        self.rotation_rad: Radians = jnp.deg2rad(val)

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

    def on_grid(self, ray: Ray, as_int: bool = True) -> NDArray:
        return ray_on_grid(
            ray,
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def get_det_coords_for_gauss_rays(self, xEnd, yEnd):
        det_size_y = self.shape[0] * self.pixel_size
        det_size_x = self.shape[1] * self.pixel_size

        x_det = jnp.linspace(-det_size_y / 2,
                             det_size_y / 2,
                             self.shape[0],
                             dtype=xEnd.dtype) + self.center[0]

        y_det = jnp.linspace(-det_size_x / 2,
                             det_size_x / 2,
                             self.shape[1],
                             dtype=yEnd.dtype) + self.center[1]

        x, y = jnp.meshgrid(x_det, y_det)

        r = jnp.stack((x, y), axis=-1).reshape(-1, 2)

        return r

    def image_dtype(
        self,
        interfere: bool
    ):
        if interfere:
            return jnp.complex128
            return jnp.complex64
        else:
            return jnp.int32

    def get_image(
        self,
        ray: Union[Ray, Sequence[Ray]],
        interfere: bool = True,
        out: Optional[NDArray] = None,
    ) -> NDArray:

        if not isinstance(ray, Sequence):
            ray = [ray]
        assert len(ray) > 0

        image_dtype = self.image_dtype(interfere)

        if out is None:
            out = jnp.zeros(
                self.shape,
                dtype=image_dtype,
            )
        else:
            assert out.dtype == image_dtype
            assert out.shape == self.shape

        if interfere and isinstance(ray[0], GaussianRay):
            if len(ray) < 2:
                raise UsageError(
                    "GaussianRays must have two sets of rays to calculate interference"
                )

            self._gaussian_beam_summation(ray, out=out)

        else:
            out = self._basic_beam_summation(ray, interfere, out=out)

        return out.reshape(self.shape)

    def _basic_beam_summation(
        self,
        ray: tuple[Ray],
        interfere: bool,
        out: Optional[NDArray] = None,
    ) -> NDArray:

        ray = ray[-1]

        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = self.on_grid(ray, as_int=True)
        sy, sx = self.shape
        mask = jnp.logical_and(
            jnp.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            jnp.logical_and(
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
            wavefronts = 1.0 * jnp.exp(-1j * (2 * jnp.pi / ray.wavelength) * ray.pathlength)
            valid_wavefronts = wavefronts[mask]
        else:
            # If we are not doing interference, we simply add 1 to each pixel that a ray hits
            valid_wavefronts = 1

        # Get a flattened list pixel indices where rays have hit
        flat_icds = jnp.ravel_multi_index(
                [
                    pixel_coords_y[mask],
                    pixel_coords_x[mask],
                ],
                out.shape
            )
        
        out_flat = out.flatten()
        out_flat = self.sum_rays_on_detector(out_flat, flat_icds, valid_wavefronts)

        return out_flat

    def sum_rays_on_detector(self,
                             out: NDArray,
                             flat_icds: NDArray,
                             valid_wavefronts: NDArray):

        if jnp.iscomplexobj(out):
            # Separate the real and imaginary parts
            real_out = out.real
            imag_out = out.imag

            # Perform the addition separately for real and imaginary parts
            real_out = real_out.at[flat_icds].add(valid_wavefronts.real)
            imag_out = imag_out.at[flat_icds].add(valid_wavefronts.imag)

            # Combine the real and imaginary parts back into the out array
            out = real_out + 1j * imag_out
        else:
            # Perform the addition directly for non-complex arrays
            out = out.at[flat_icds].add(valid_wavefronts)

        return out

    # def _gaussian_beam_summation(
    #     self,
    #     ray: tuple[GaussianRay],
    #     ABCD: NDArray,
    #     out: Optional[NDArray] = None,
    # ) -> NDArray:

    #     ray_start = ray[0]
    #     ray_end = ray[-1]

    #     float_dtype = out.real.dtype.type

    #     wo = ray_start.wo
    #     wavelength = ray_start.wavelength

    #     div = ray_start.wavelength / (jnp.pi * wo)
    #     k = float_dtype(2 * jnp.pi / wavelength)

    #     z0 = float_dtype(ray_start.z)
    #     z_r = float_dtype(jnp.pi * wo ** 2 / wavelength)


    #     pathlength = ray_end.path_length[0::5].astype(float_dtype)


    #     # A, B, C, D all have shape (n_gauss, 2, 2)
    #     Qinv = calculate_Qinv(z0, z_r, wo, wavelength, n_gauss, xp=xp)

    #     # matmul, addition and mat inverse inside
    #     # on operands with form (n_gauss, 2, 2)
    #     # matmul broadcasts in the last two indices
    #     # inv can be broadcast with jnp.linalg.inv last 2 idcs
    #     # if all inputs are stacked in the first dim
    #     Qpinv = calculate_Qpinv(A, B, C, D, Qinv, xp=xp)

    #     px1m = rays_start.data[0, 0::5]  # x that central ray leaves at
    #     py1m = rays_start.data[2, 0::5]  # y that central ray leaves at
    #     thetax1m = rays_start.data[1, 0::5]  # slope that central ray arrives at
    #     thetay1m = rays_start.data[3, 0::5]  # slope that central ray arrives at

    #     p1m = jnp.array([px1m, py1m]).T.astype(float_dtype)
    #     theta1m = jnp.array([thetax1m, thetay1m]).T.astype(float_dtype)

    #     xEnd, yEnd = rayset1[0, 0], rayset1[0, 2]

    #     # central beam final x , y coords
    #     det_coords = get_det_coords_for_gauss_rays(xEnd, yEnd, xp=xp)

    #     propagate_misaligned_gaussian(
    #         Qinv, Qpinv, det_coords, p1m,
    #         theta1m, k, A, B, path_length, out.ravel(), xp=xp
    #     )
