from typing_extensions import TypeAlias


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float


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
