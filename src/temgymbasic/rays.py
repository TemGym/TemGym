from typing import (
    Tuple, Optional, Union, TYPE_CHECKING
)
from typing_extensions import Self
from dataclasses import dataclass, asdict

from numpy.typing import NDArray

from . import (
    PositiveFloat,
    Degrees,
)
from .utils import (
    get_pixel_coords,
    calculate_phi_0
)

from .backend import xp

if TYPE_CHECKING:
    from .components import Component


LocationT = Union[float, 'Component', Tuple['Component', ...]]


@dataclass
class Rays:
    data: NDArray
    location: LocationT
    path_length: NDArray
    wavelength: Optional[float] = None
    mask: Optional[NDArray] = None
    blocked: Optional[NDArray] = None
    wo: Optional[NDArray] = None

    def __eq__(self, other: 'Rays') -> bool:
        return self.num == other.num and (self.data == other.data).all()

    @classmethod
    def new(
        cls,
        data: NDArray,
        location: LocationT,
        wavelength: Optional[float] = None,
        path_length: Union[float, NDArray] = 0.,
        wo: Optional[NDArray] = None,
    ):
        num_rays = data.shape[1]
        if xp.isscalar(path_length):
            path_length = xp.full((num_rays,), path_length)
        return cls(
            data=data,
            location=location,
            path_length=path_length,
            wavelength=wavelength,
            wo=wo
        )

    @property
    def z(self) -> float:
        try:
            return self.component.z
        except AttributeError:
            return self.location

    @property
    def component(self) -> Optional['Component']:
        try:
            return self.location[-1]
        except TypeError:
            pass
        try:
            _ = self.location.z
            return self.location
        except AttributeError:
            pass
        return None

    @property
    def num(self):
        return self.data.shape[1]

    @property
    def x(self):
        return self.data[0, :]

    @x.setter
    def x(self, xpos):
        self.data[0, :] = xpos

    @property
    def y(self):
        return self.data[2, :]

    @y.setter
    def y(self, ypos):
        self.data[2, :] = ypos

    @property
    def yx(self):
        return self.data[[2, 0], :]

    @property
    def dx(self):
        return self.data[1, :]

    @dx.setter
    def dx(self, xslope):
        self.data[1, :] = xslope

    @property
    def dy(self):
        return self.data[3, :]

    @dy.setter
    def dy(self, yslope):
        self.data[3, :] = yslope

    @property
    def phi_0(self):
        if self.wavelength is not None:
            return calculate_phi_0(self.wavelength)
        return self.wavelength

    @staticmethod
    def propagation_matrix(z):
        '''
        Propagation matrix

        Parameters
        ----------
        z : float
            Distance to propagate rays

        Returns
        -------
        ndarray
            Propagation matrix
        '''
        return xp.array(
            [[1, z, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, z, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]
        )

    def propagate(self, distance: float) -> Self:
        degree_x = xp.rad2deg(xp.arctan(self.dx))
        degree_y = xp.rad2deg(xp.arctan(self.dy))

        if xp.any(degree_x > 20):
            raise ValueError(f"dx is too large for parabasal representation: {xp.max(degree_x)}")
        elif xp.any(degree_y > 20):
            raise ValueError(f"dy is too large for parabasal representation: {xp.max(degree_y)}")

        return self.new_with(
            data=xp.matmul(
                self.propagation_matrix(distance),
                self.data,
            ),
            location=self.z + distance,
            path_length=(
                self.path_length
                + 1.0 * distance * (1 + self.dx ** 2 + self.dy ** 2) ** 0.5
            ),
        )

    def propagate_to(self, z: float) -> Self:
        return self.propagate(z - self.z)

    def on_grid(
        self,
        shape: Tuple[int, int],
        pixel_size: 'PositiveFloat',
        flip_y: bool = False,
        rotation: Degrees = 0.,
        as_int: bool = True
    ) -> Tuple[NDArray, NDArray]:
        """Returns in yy, xx!"""
        xx, yy = get_pixel_coords(
            rays_x=self.x,
            rays_y=self.y,
            shape=shape,
            pixel_size=pixel_size,
            flip_y=flip_y,
            scan_rotation=rotation,
        )
        if as_int:
            return xp.round((yy, xx)).astype(int)
        return yy, xx

    def get_image(
        self,
        shape: Tuple[int, int],
        pixel_size: PositiveFloat,
        flip_y: bool = False,
        rotation: Degrees = 0.
    ):
        from .components import Detector
        det = Detector(
            z=self.z,
            pixel_size=pixel_size,
            shape=shape,
            rotation=rotation,
            flip_y=flip_y,
        )
        return det.get_image(self)

    def new_with(self, **kwargs):
        kwargs = {
            **asdict(self),
            **kwargs,
        }
        kwargs.pop('mask', None)
        kwargs.pop('blocked', None)
        return type(self)(**kwargs)

    def with_mask(
        self, mask: NDArray, **kwargs,
    ):
        return type(self)(
            data=self.data[:, mask],
            path_length=self.path_length[mask],
            mask=mask,
            blocked=self.data[:, ~mask],
            **kwargs,
        )

    def blocked_rays(self) -> Optional['Rays']:
        if self.blocked is None or self.mask is None:
            return None
        return Rays(
            data=self.blocked,
            location=self.z,
            path_length=None,  # this is invalidates the interface
            wavelength=self.wavelength,
            mask=~self.mask,
        )
