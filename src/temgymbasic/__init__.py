from typing_extensions import TypeAlias, Literal


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float

BackendT = Literal['cpu', 'gpu']


class UsageError(Exception):
    ...


class InvalidModelError(Exception):
    ...


def get_cupy():
    try:
        import cupy as cp
    except ImportError:
        cp = None
    return cp
