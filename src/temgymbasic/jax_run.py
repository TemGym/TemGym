
from .jax_ray import propagate
import jax
# import jax.numpy as jnp


def run_to_end(ray, components):
    for component in components:

        distance = (component.z - ray.z).squeeze()
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray
