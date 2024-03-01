from typing_extensions import TypeAlias

PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float
Radians: TypeAlias = float
Degrees: TypeAlias = float


class UsageError(Exception):
    ...


class InvalidModelError(Exception):
    ...
