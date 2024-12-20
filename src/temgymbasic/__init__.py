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
