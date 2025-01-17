import jax_dataclasses as jdc
import jax.numpy as jnp
from typing import Optional
from utils import get_pixel_coords
from numpy.typing import NDArray
from typing import Tuple
from . import (
    PositiveFloat,
    Degrees,
)


@jdc.pytree_dataclass
class Ray:
    matrix: jnp.ndarray  # Shape (5,) vector [x, y, dx, dy, 1]
    z: float
    amplitude: float
    pathlength: float
    wavelength: float
    blocked: jdc.Static[Optional[jnp.ndarray]] = 0


@jdc.pytree_dataclass
class GaussianRay(Ray):
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


def propagate(distance, ray: Ray):
    x, y, dx, dy = ray.matrix[:4]

    new_x = x + dx * distance
    new_y = y + dy * distance

    pathlength = ray.pathlength + distance * jnp.sqrt(1 + dx ** 2 + dy ** 2)
    new_matrix = jnp.array([new_x, new_y, dx, dy, 1.])

    return Ray(
        z=ray.z + distance,
        matrix=new_matrix,
        amplitude=ray.amplitude,
        pathlength=pathlength,
        wavelength=ray.wavelength,
        blocked=ray.blocked
    )


def on_grid(
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
