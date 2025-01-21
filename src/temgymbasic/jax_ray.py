import jax_dataclasses as jdc
import jax.numpy as jnp
from typing import Optional
from .jax_utils import get_pixel_coords
from numpy.typing import NDArray
from typing import Tuple
from . import (
    PositiveFloat,
    Degrees,
)


@jdc.pytree_dataclass
class Ray:
    matrix: jnp.ndarray  # Shape (5,) vector [x, y, dx, dy, 1], or shape (N, 5)
    z: float
    amplitude: jdc.Static[float] = 1.0
    pathlength: jdc.Static[float] = 0.0
    wavelength: jdc.Static[float] = 1.0
    blocked: jdc.Static[Optional[jnp.ndarray]] = 0

    @property
    def x(self):
        return self.matrix[..., 0]

    @property
    def y(self):
        return self.matrix[..., 1]

    @property
    def dx(self):
        return self.matrix[..., 2]

    @property
    def dy(self):
        return self.matrix[..., 3]


@jdc.pytree_dataclass
class GaussianRay(Ray):
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


def propagate(distance, ray: Ray):
    x, y, dx, dy = ray.x, ray.y, ray.dx, ray.dy

    new_x = x + dx * distance
    new_y = y + dy * distance

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)

    Ray = ray_matrix(new_x, new_y, dx, dy,
                    ray.z + distance, ray.amplitude,
                    pathlength, ray.wavelength,
                    ray.blocked)
    return Ray


def ray_on_grid(
    ray,
    shape: Tuple[int, int],
    pixel_size: 'PositiveFloat',
    flip_y: bool = False,
    rotation: Degrees = 0.,
    as_int: bool = True
) -> Tuple[NDArray, NDArray]:
    """Returns in yy, xx!"""
    xx, yy = get_pixel_coords(
        rays_x=ray.x,
        rays_y=ray.y,
        shape=shape,
        pixel_size=pixel_size,
        flip_y=flip_y,
        scan_rotation=rotation,
    )

    if as_int:
        xx = jnp.round(xx).astype(jnp.int32)
        yy = jnp.round(yy).astype(jnp.int32)

    return yy, xx


def ray_matrix(x, y, dx, dy,
               z, amplitude,
               pathlength, wavelength,
               blocked):

    # new_matrix = jnp.stack([x, y, dx, dy, jnp.ones_like(x)], axis=1) # Doesnt work if all values have 0 shape
    new_matrix = jnp.array([x, y, dx, dy, jnp.ones_like(x)]).T  # Doesnt work if all values have 0 shape
    return Ray(
        matrix=new_matrix,
        z=z,
        amplitude=amplitude,
        pathlength=pathlength,
        wavelength=wavelength,
        blocked=blocked
    )
