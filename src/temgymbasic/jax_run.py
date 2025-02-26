
from .jax_ray import propagate

def run_to_end(ray, components):
    for component in components:
        distance = (component.z - ray.z)
        ray = propagate(distance, ray)
        ray = component.step(ray)

    return ray

def run_to_component(ray, component):
    distance = (component.z - ray.z)
    ray = propagate(distance, ray)
    ray = component.step(ray)
    return ray