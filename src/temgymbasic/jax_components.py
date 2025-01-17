import jax_dataclasses as jdc
import jax.numpy as jnp
from typing import Tuple
from .jax_ray import Ray, propagate
from . import (
    Degrees,
)


@jdc.pytree_dataclass
class Lens:
    z: float
    focal_length: float

    @jdc.jit
    def step(self, ray: Ray):
        f = self.focal_length

        x, y, dx, dy = ray.matrix[:4]

        new_dx = -x / f + dx
        new_dy = -y / f + dy

        pathlength = ray.pathlength - (x ** 2 + y ** 2) / (2 * f)

        new_matrix = jnp.array([x, y, new_dx, new_dy, 1.])

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

    @jdc.jit
    def step(self, ray: Ray):

        x, y, dx, dy = ray.matrix[:4]
        new_dx = dx + self.def_x
        new_dy = dy + self.def_y

        pathlength = ray.pathlength + dx * x + dy * y

        new_matrix = jnp.array([x, y, new_dx, new_dy, 1.])

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

    @jdc.jit
    def step(self, ray: Ray):
        ray = self.first.step(ray)
        z_step = self.second.z - self.first.z
        ray = propagate(z_step)
        ray = self.second.step(ray)

        return ray


@jdc.pytree_dataclass
class Aperture:
    z: float
    radius: float
    x: float
    y: float

    def step(self, ray: Ray):

        pos_x, pos_y, dx, dy = ray.matrix[:4]
        distance = jnp.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        mask = jnp.where(distance < self.radius, 1, ray.mask)
        return Ray(
            z=ray.z,
            matrix=ray.matrix,
            amplitude=ray.amplitude,
            pathlength=ray.pathlength,
            wavelength=ray.wavelength,
            mask=mask
        )


@jdc.pytree_dataclass
class Biprism:
    z: float
    offset: float = 0.
    rotation: Degrees = 0.
    deflection: float = 0.

    @jdc.jit
    def step(
        self, ray: Ray,
    ) -> Ray:

        pos_x, pos_y, dx, dy = ray.matrix[:4]

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

        new_dx = dx + xdeflection_mag * deflection
        new_dy = dy + ydeflection_mag * deflection

        pathlength = ray.pathlength + (
            xdeflection_mag * deflection * pos_x + ydeflection_mag * deflection * pos_y
        )

        new_matrix = jnp.array([pos_x, pos_y, new_dx, new_dy, 1.])

        return Ray(
            z=ray.z,
            matrix=new_matrix,
            amplitude=ray.amplitude,
            pathlength=pathlength,
            wavelength=ray.wavelength
        )


@jdc.pytree_dataclass
class Detector:
    z: float
    pixel_size: float
    shape: jdc.Static[Tuple[int, int]]
    rotation: Degrees = 0.
    flip_y: jdc.Static[bool] = False
    center: jdc.Static[Tuple[float, float]] = (0., 0.)

    @jdc.jit
    def step(self, ray: Ray):
        return ray
