import jax
import jax_dataclasses as jdc
import jax.numpy as jnp
from typing import Optional


@jdc.pytree_dataclass
class Ray:
    matrix: jnp.ndarray  # Shape (5,) vector [x, y, dx, dy, 1]
    z: float
    amplitude: float
    pathlength: float
    wavelength: float
    mask: jdc.Static[Optional[jnp.ndarray]] = None
    blocked: jdc.Static[Optional[jnp.ndarray]] = None


@jdc.pytree_dataclass
class GaussianRay(Ray):
    w0x: float = 1.0
    Rx: float = 0.0
    w0y: float = 1.0
    Ry: float = 0.0


@jax.jit
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
        wavelength=ray.wavelength
    )
