import abc
from typing import Generator, Iterable, Tuple, Optional, Union, Type, TypeAlias, Self
from itertools import pairwise
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from .utils import (
    get_pixel_coords,
    P2R, R2P,
    circular_beam,
)


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float


class UsageError(Exception):
    ...


class InvalidModelError(Exception):
    ...


@dataclass
class Rays:
    data: np.ndarray
    indices: np.ndarray
    location: Union[float, 'Component', Tuple['Component', ...]]
    path_length: np.ndarray

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

    @property
    def y(self):
        return self.data[2, :]

    @property
    def yx(self):
        return self.data[[2, 0], :]

    @property
    def dx(self):
        return self.data[1, :]

    @property
    def dy(self):
        return self.data[3, :]

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
        return np.array(
            [[1, z, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, z, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]]
        )

    def propagate(self, distance: float) -> Self:
        return Rays(
            data=np.matmul(
                self.propagation_matrix(distance),
                self.data,
            ),
            indices=self.indices,
            location=self.z + distance,
            path_length=(
                self.path_length
                + 1.0 * distance * (1 + self.dx**2 + self.dy**2)**0.5
            )
        )

    def propagate_to(self, z: float) -> Self:
        return self.propagate(z - self.z)

    def on_grid(
        self,
        shape: Tuple[int, int],
        pixel_size: PositiveFloat,
        flip_y: bool = False,
        rotation: float = 0.,
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
            return np.round((yy, xx)).astype(int)
        return yy, xx

    def get_image(
        self,
        shape: Tuple[int, int],
        pixel_size: PositiveFloat,
        flip_y: bool = False,
        rotation: float = 0.
    ):
        det = Detector(
            z=self.z,
            pixel_size=pixel_size,
            shape=shape,
            rotation=rotation,
            flip_y=flip_y,
        )
        return det.get_image(self)


class Component(abc.ABC):
    def __init__(self, z: float, name: Optional[str] = None):
        if name is None:
            name = type(self).__name__
        self._name = name
        self._z = z

    def _validate_component(self):
        pass

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, new_z: float):
        raise UsageError("Do not set z on a component directly, use Model methods")

    def _set_z(self, new_z: float):
        self._z = new_z

    @property
    def entrance_z(self) -> float:
        return self.z

    @property
    def exit_z(self) -> float:
        return self.z

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}: {self._name} @ z = {self.z}'

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        raise NotImplementedError

    @staticmethod
    def gui_wrapper() -> Type['ComponentGUIWrapper']:
        return ComponentGUIWrapper


class ComponentGUIWrapper:
    def __init__(self, component: Component):
        self.component = component


class Lens(Component):
    def __init__(self, z: float, f: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)
        self._f = f

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

    @property
    def ffp(self) -> float:
        return self.z - abs(self.f)

    @staticmethod
    def lens_matrix(f):
        '''
        Lens ray transfer matrix

        Parameters
        ----------
        f : float
            Focal length of lens

        Returns
        -------
        ndarray
            Output Ray Transfer Matrix
        '''

        return np.array(
            [[1,      0, 0,      0, 0],
             [-1 / f, 1, 0,      0, 0],
             [0,      0, 1,      0, 0],
             [0,      0, -1 / f, 1, 0],
             [0,      0, 0,      0, 1]]
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Just straightforward matrix multiplication
        yield Rays(
            data=np.matmul(self.lens_matrix(self.f), rays.data),
            indices=rays.indices,
            location=self,
            path_length=rays.path_length
        )


class Sample(Component):
    def __init__(self, z: float, name: Optional[str] = None):
        super().__init__(name=name, z=z)

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Sample has no effect, yet
        # Could implement ray intensity / attenuation ??
        rays.location = self
        yield rays


class STEMSample(Sample):
    def __init__(
        self,
        z: float,
        overfocus: NonNegativeFloat = 0.,
        semiconv_angle: PositiveFloat = 0.01,
        scan_shape: Tuple[int, int] = (8, 8),
        scan_step_yx: Tuple[float, float] = (0.01, 0.01),
        scan_rotation: float = 0.,
        name: Optional[str] = None,
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx
        self.scan_rotation = scan_rotation

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step_yx
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y
        if self.scan_rotation != 0.:
            pos_r, pos_a = R2P(scan_position_x + scan_position_y * 1j)
            pos_c = P2R(pos_r, pos_a + self.scan_rotation)
            scan_position_y, scan_position_x = pos_c.imag, pos_c.real
        return (scan_position_y, scan_position_x)

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.scan_shape,
            # This needs to be refactored...
            pixel_size=self.scan_step_yx[0],
            rotation=self.scan_rotation,
            as_int=as_int,
        )


class Source(Component):
    def __init__(
        self, z: float, tilt_yx: Tuple[float, float] = (0., 0.), name: Optional[str] = None
    ):
        super().__init__(z=z, name=name)
        self.tilt_yx = tilt_yx

    @abc.abstractmethod
    def get_rays(self, num_rays: int) -> Rays:
        raise NotImplementedError

    def _make_rays(self, r: NDArray, indices: Optional[NDArray] = None) -> Rays:
        if indices is None:
            indices = np.arange(r.shape[1])
        r[1, :] += self.tilt_yx[1]
        r[3, :] += self.tilt_yx[0]
        return Rays(
            data=r, indices=indices, location=self, path_length=np.zeros((r.shape[1],))
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Source has no effect after get_rays was called
        rays.location = self
        yield rays


class ParallelBeam(Source):
    def __init__(
        self,
        z: float,
        radius: float = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name)
        self.radius = radius

    def get_rays(self, num_rays: int) -> Rays:
        r, _ = circular_beam(num_rays, self.radius)
        return self._make_rays(r)


class XAxialBeam(ParallelBeam):
    def get_rays(self, num_rays: int) -> Rays:
        r = np.zeros((5, num_rays))
        r[0, :] = np.linspace(
            -self.radius, self.radius, num=num_rays, endpoint=True
        )
        return self._make_rays(r)


class RadialSpikesBeam(ParallelBeam):
    def get_rays(self, num_rays: int) -> Rays:
        xvals = np.linspace(
            0., self.radius, num=num_rays // 4, endpoint=True
        )
        yvals = np.zeros_like(xvals)
        origin_c = xvals + yvals * 1j

        orad, oang = R2P(origin_c)
        radius1 = P2R(orad * 0.75, oang + np.pi * 0.4)
        radius2 = P2R(orad * 0.5, oang + np.pi * 0.8)
        radius3 = P2R(orad * 0.25, oang + np.pi * 1.2)
        r_c = np.concatenate((origin_c, radius1, radius2, radius3))

        r = np.zeros((5, r_c.size))
        r[0, :] = r_c.real
        r[2, :] = r_c.imag
        return self._make_rays(r)


# class PointSource(Source):
#     def __init__(
#         self,
#         z: float,
#         semi_angle: Optional[float] = 0.,
#         tilt_yx: Tuple[float, float] = (0., 0.),
#         name: Optional[str] = None,
#     ):
#         super().__init__(name=name, z=z)
#         self.semi_angle = semi_angle
#         self.tilt_yx = tilt_yx

#     def get_rays(self, num_rays: int) -> Rays:
#         r, _ = point_beam(num_rays, self.semi_angle)
#         return self._make_rays(r)


class Detector(Component):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        rotation: float = 0.,
        flip_y: bool = False,
        center: Tuple[float, float] = (0., 0.),
        name: Optional[str] = None,
    ):
        """
        The intention of rotation is to rotate the detector
        realative to the common y/x coordinate system of the optics.
        A positive rotation would rotate the detector clockwise
        looking down a ray , and the image will appear
        to rotate anti-clockwise.

        In STEMModel the scan grid is aligned with the optics
        y/x coordinate system default, but can also
        be rotated using the "scan_rotation" parameter.

        The detector flip_y acts only at the image generation step,
        the scan grid itself can be flipped by setting negative
        scan step values
        """
        super().__init__(name=name, z=z)
        self.pixel_size = pixel_size
        self.shape = shape
        self.rotation = rotation  # degrees
        self.flip_y = flip_y
        self.center = center

    def set_center_px(self, center_px: Tuple[int, int]):
        """
        For the desired image center in pixels (after any flip / rotation)
        set the image center in the physical coordinates of the microscope

        The continuous coordinate can be set directly on detector.center
        """
        iy, ix = center_px
        sy, sx = self.shape
        cont_y = (iy - sy // 2) * self.pixel_size
        cont_x = (ix - sx // 2) * self.pixel_size
        if self.flip_y:
            cont_y = -1 * cont_y
        mag, angle = R2P(cont_x + 1j * cont_y)
        coord: complex = P2R(mag, angle + np.deg2rad(self.rotation))
        self.center = coord.imag, coord.real

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Detector has no effect on rays
        rays.location = self
        yield rays

    def on_grid(self, rays: Rays, as_int: bool = True) -> NDArray:
        return rays.on_grid(
            shape=self.shape,
            pixel_size=self.pixel_size,
            flip_y=self.flip_y,
            rotation=self.rotation,
            as_int=as_int,
        )

    def get_image(self, rays: Rays) -> NDArray:
        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = self.on_grid(rays, as_int=True)
        sy, sx = self.shape
        mask = np.logical_and(
            np.logical_and(
                0 <= pixel_coords_y,
                pixel_coords_y < sy
            ),
            np.logical_and(
                0 <= pixel_coords_x,
                pixel_coords_x < sx
            )
        )
        image = np.zeros(
            self.shape,
            dtype=int,
        )
        flat_icds = np.ravel_multi_index(
            [
                pixel_coords_y[mask],
                pixel_coords_x[mask],
            ],
            image.shape
        )
        # Increment at each pixel for each ray that hits
        np.add.at(
            image.ravel(),
            flat_icds,
            1,
        )
        return image


class Deflector(Component):
    '''Creates a single deflector component and handles calls to GUI creation, updates to GUI
        and stores the component matrix. See Double Deflector component for a more useful version
    '''
    def __init__(
        self,
        z: float,
        defx: float = 0.,
        defy: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        defy : float, optional
            deflection kick in slope units to the incoming ray y angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.defy = defy

    @staticmethod
    def deflector_matrix(def_x, def_y):
        '''Single deflector ray transfer matrix

        Parameters
        ----------
        def_x : float
            deflection in x in slope units
        def_y : _type_
            deflection in y in slope units

        Returns
        -------
        ndarray
            Output ray transfer matrix
        '''

        return np.array(
            [[1, 0, 0, 0,     0],
             [0, 1, 0, 0, def_x],
             [0, 0, 1, 0,     0],
             [0, 0, 0, 1, def_y],
             [0, 0, 0, 0,     1]],
        )

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        yield Rays(
            data=np.matmul(
                self.deflector_matrix(self.defx, self.defy),
                rays.data,
            ),
            indices=rays.indices,
            location=self,
            path_length=rays.path_length
        )


class DoubleDeflector(Component):
    def __init__(
        self,
        first: Deflector,
        second: Deflector,
        name: Optional[str] = None,
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self._first = first
        self._second = second
        self._validate_component()

    def _validate_component(self):
        if self.first.z >= self.second.z:
            raise InvalidModelError("First deflector must be before second")

    @property
    def length(self) -> float:
        return self._second.z - self._first.z

    @property
    def first(self) -> Deflector:
        return self._first

    @property
    def second(self) -> Deflector:
        return self._second

    @property
    def z(self):
        self._z = (self.first.z + self.second.z) / 2
        return self._z

    def _set_z(self, new_z: float):
        dz = new_z - self.z
        self.first._set_z(self.first.z + dz)
        self.second._set_z(self.second.z + dz)

    @property
    def entrance_z(self) -> float:
        return self.first.z

    @property
    def exit_z(self) -> float:
        return self.second.z

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        for rays in self.first.step(rays):
            rays.location = (self, self.first)
            yield rays
        rays = rays.propagate_to(self.second.entrance_z)
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays

    @staticmethod
    def _send_ray_through_pts_1d(
        in_zp: Tuple[float, float],
        z_out: float,
        pt1_zp: Tuple[float, float],
        pt2_zp: Tuple[float, float],
        in_slope: float = 0.
    ) -> Tuple[float, float]:
        """
        Choose first/second deflector values such that a ray arriving
        at (in_zp) with slope (in_slope), will leave at (z_out, ...) and
        pass through (pt1_zp) then (pt2_zp)
        """
        in_zp = np.asarray(in_zp)
        pt1_zp = np.asarray(pt1_zp)
        pt2_zp = np.asarray(pt2_zp)
        dp = pt1_zp - pt2_zp
        out_zp = np.asarray(
            (
                z_out,
                pt2_zp[1] + dp[1] * (z_out - pt2_zp[0]) / dp[0],
            )
        )
        dd = out_zp - in_zp
        first_def = dd[1] / dd[0]
        first_def += in_slope
        out_slope = dp[1] / dp[0]
        second_def = out_slope - first_def
        return first_def, second_def

    def send_ray_through_points(
        self,
        in_ray: Tuple[float, float],
        pt1: Tuple[float, float, float],
        pt2: Tuple[float, float, float],
        in_slope: Tuple[float, float] = (0., 0.)
    ):
        """
        in_ray is (y, x), z is implicitly the z of the first deflector
        pt1 and pt2 are (z, y, x) after the second deflector
        in_slope is (dy, dx) at the incident point
        """
        self.first.defy, self.second.defy = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[0]),
            self.second.z,
            pt1[:2],
            pt2[:2],
            in_slope=in_slope[0],
        )
        self.first.defx, self.second.defx = self._send_ray_through_pts_1d(
            (self.first.z, in_ray[1]),
            self.second.z,
            (pt1[0], pt1[2]),
            (pt2[0], pt2[2]),
            in_slope=in_slope[1],
        )


class Biprism(Component):
    def __init__(
        self,
        z: float,
        x: float = 0.,
        defx: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        x: float
            Central position of biprism in x dimension
        name : str, optional
            Name of this component which will be displayed by GUI, by default ''
        defx : float, optional
            deflection kick in slope units to the incoming ray x angle, by default 0.5
        '''
        super().__init__(z=z, name=name)
        self.defx = defx
        self.x = x

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x = rays.x
        x_dist = (pos_x - self.x)
        x_sign = np.sign(x_dist)
        rays.data[1] = rays.data[1] + self.defx*x_sign

        yield Rays(
            data=rays.data,
            indices=rays.indices,
            path_length=rays.path_length + np.abs(self.defx)*x_dist,
            location=self,
        )


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius_inner: float = 0.005,
        radius_outer: float = 0.25,
        x: float = 0.,
        y: float = 0.,
        name: Optional[str] = None,
    ):
        '''

        Parameters
        ----------
        z : float
            Position of component in optic axis
        name : str, optional
            Name of this component which will be displayed by GUI, by default 'Aperture'
        radius_inner : float, optional
           Inner radius of the aperture, by default 0.005
        radius_outer : float, optional
            Outer radius of the aperture, by default 0.25
        x : int, optional
            X position of the centre of the aperture, by default 0
        y : int, optional
            Y position of the centre of the aperture, by default 0
        '''

        super().__init__(z, name)

        self.x = x
        self.y = y
        self.radius_inner = radius_inner
        self.radius_outer = radius_outer

    def step(
        self, rays: Rays,
    ) -> Generator[Rays, None, None]:
        pos_x, pos_y = rays.x, rays.y
        distance = np.sqrt(
            (pos_x - self.x) ** 2 + (pos_y - self.y) ** 2
        )
        mask = np.logical_and(
            distance >= self.radius_inner,
            distance < self.radius_outer,
        )
        yield Rays(
            data=rays.data[:, mask], indices=rays.indices[mask],
            location=self, path_length=rays.path_length[mask],
        )


class Model:
    def __init__(self, components: Iterable[Component]):
        self._components = components
        self._sort_components()
        self._validate_components()

    def _validate_components(self):
        if len(self._components) <= 1:
            raise InvalidModelError("Must have at least one component")
        if not isinstance(self.source, Source):
            raise InvalidModelError("First component must always be a Source")
        if any(
            next_c.entrance_z <= this_c.exit_z
            for this_c, next_c
            in pairwise(self._components)
        ):
            raise InvalidModelError(
                "Components must be sorted in increasing in z position with no overlap"
            )
        for c in self._components:
            c._validate_component()

    @property
    def components(self) -> Iterable[Component]:
        return self._components

    def __repr__(self):
        repr_string = f'[{self.__class__.__name__}]:'
        for component in self.components:
            repr_string = repr_string + f'\n - {repr(component)}'
        return repr_string

    @property
    def source(self) -> Source:
        return self.components[0]

    @property
    def detector(self) -> Optional[Detector]:
        if isinstance(self.last, Detector):
            return self.last
        return None

    @property
    def last(self) -> Detector:
        return self.components[-1]

    def move_component(
        self,
        component: Component,
        z: float,
    ):
        old_z = component.z
        component._set_z(z)
        self._sort_components()
        try:
            self._validate_components()
        except InvalidModelError as err:
            # Unwind the change but reraise
            self.move_component(component, old_z)
            raise err from None

    def add_component(self, component: Component):
        original_components = tuple(self._components)
        self._components = tuple(self._components) + (component,)
        self._sort_components()
        try:
            self._validate_components()
        except InvalidModelError as err:
            # Unwind the change but reraise
            self._components = original_components
            raise err from None

    def remove_component(self, component: Component):
        original_components = tuple(self._components)
        self._components = tuple(
            c for c in self._components
            if c is not component
        )
        if len(self._components) == len(original_components):
            raise ValueError("Component not found in model, cannot remove")
        try:
            self._validate_components()
        except InvalidModelError as err:
            # Unwind the change but reraise
            self._components = original_components
            raise err from None

    def _sort_components(self):
        self._components = tuple(
            sorted(self._components, key=lambda c: c.z)
        )

    def run_iter(
        self, num_rays: int
    ) -> Generator[Rays, None, None]:
        source: Source = self.components[0]
        rays = source.get_rays(num_rays)
        for component in self.components:
            rays = rays.propagate_to(component.entrance_z)
            for rays in component.step(rays):
                # Could use generator with return value here...
                yield rays

    def run_to_z(self, num_rays: int, z: float) -> Optional[Rays]:
        """
        Get the rays at a point z

        If z is the position of a component, this returns the rays before the
        component affects the rays. To get the rays just after the component
        use 'run_to_component' instead.
        """
        last_rays = None
        for rays in self.run_iter(num_rays):
            if last_rays is not None and (last_rays.z < z <= rays.z):
                return last_rays.propagate_to(z)
            last_rays = rays
        return None

    def run_to_end(self, num_rays: int) -> Rays:
        for rays in self.run_iter(num_rays):
            pass
        return rays

    def run_to_component(
        self,
        component: Component,
        num_rays: int,
    ) -> Optional[Rays]:
        for rays in self.run_iter(num_rays):
            if rays.component is component:
                return rays
        return None


class STEMModel(Model):
    def __init__(self):
        # Note a flip_y or flip_x can be achieved by setting
        # either of scan_step_yx to negative values
        self._scan_pixel_yx = (0, 0)  # Maybe should live on STEMSample
        super().__init__(self.default_components())
        self.set_stem_params()

    def _validate_components(self):
        super()._validate_components()
        if not isinstance(self.source, ParallelBeam):
            raise InvalidModelError("Must have a ParallelBeam for STEMModel")
        # Called here because even if all components are valid from
        # the perspective of z the current overfocus/semiconv
        # could forbid the new state, so we change the state during
        # validation such that it could be unwound if necessary
        self.set_stem_params()

    def _sort_components(self):
        """Component order fixed in STEMModel"""
        pass

    def add_component(self, component: Component):
        raise UsageError("Cannot add components to STEMModel")

    def remove_component(self, component: Component):
        raise UsageError("Cannot remove components from STEMModel")

    @staticmethod
    def default_components():
        """
        Just an initial valid state for the model
        """
        return (
            ParallelBeam(
                z=0.0,
                radius=0.01,
            ),
            DoubleDeflector(
                first=Deflector(z=0.1),
                second=Deflector(z=0.15),
            ),
            Lens(
                z=0.3,
                f=0.1,
            ),
            STEMSample(
                z=0.5,
            ),
            DoubleDeflector(
                first=Deflector(z=0.6),
                second=Deflector(z=0.625),
            ),
            Detector(
                z=1.,
                pixel_size=0.01,
                shape=(128, 128),
            ),
        )

    @property
    def source(self) -> ParallelBeam:
        return self.components[0]

    @property
    def scan_coils(self) -> DoubleDeflector:
        return self.components[1]

    @property
    def objective(self) -> Lens:
        return self.components[2]

    @property
    def sample(self) -> STEMSample:
        return self.components[3]

    @property
    def descan_coils(self) -> DoubleDeflector:
        return self.components[4]

    @property
    def scan_coord(self) -> Tuple[int, int]:
        return self._scan_pixel_yx

    @scan_coord.setter
    def scan_coord(self, scan_pixel_yx: Tuple[int, int]):
        self._scan_pixel_yx = scan_pixel_yx
        self.move_to(self.scan_coord)

    def set_stem_params(
        self,
        overfocus: Optional[float] = None,
        semiconv_angle: Optional[PositiveFloat] = None,
        scan_step_yx: Optional[Tuple[PositiveFloat, PositiveFloat]] = None,
        scan_shape: Optional[Tuple[int, int]] = None,
        scan_rotation: Optional[float] = None,
        camera_length: Optional[float] = None,
    ) -> Self:
        """
        Change one-or-more STEM params

        This must be the endpoint that is used to keep the
        model in a valid state, as all the potential input parameters
        eventually affect the coil deflection values

        The other method is `move_to` to set the current scan position.
        Neither method requires that the current scan coordinate is actually
        within the scan grid.
        """
        if overfocus is not None:
            self.sample.overfocus = overfocus
        if semiconv_angle is not None:
            self.sample.semiconv_angle = semiconv_angle
        self.set_obj_lens_f_from_overfocus()
        self.set_beam_radius_from_semiconv()
        if scan_step_yx is not None:
            self.sample.scan_step_yx = scan_step_yx
        if scan_shape is not None:
            self.sample.scan_shape = scan_shape
        if scan_rotation is not None:
            self.sample.scan_rotation = scan_rotation
        if camera_length is not None:
            self.move_component(
                self.detector,
                self.sample.z + camera_length
            )
        self.move_to(self.scan_coord)
        return self

    def set_obj_lens_f_from_overfocus(self):
        if self.sample.overfocus > (self.sample.z - self.objective.z):
            raise InvalidModelError("Overfocus point is before lens")
        self.objective.f = self.sample.z - (self.objective.z + self.sample.overfocus)

    def set_beam_radius_from_semiconv(self):
        self.source.radius = abs(self.objective.f) * np.tan(abs(self.sample.semiconv_angle))

    def move_to(self, scan_pixel_yx: Tuple[int, int]):
        self._scan_pixel_yx = scan_pixel_yx
        scan_position = self.sample.scan_position(scan_pixel_yx)
        centerline = (0., 0.)
        exit_axis = self.detector.center

        self.scan_coils.send_ray_through_points(
            centerline,
            (self.objective.ffp, *centerline),
            (self.objective.z, *scan_position),
        )
        self.descan_coils.send_ray_through_points(
            scan_position,
            (self.descan_coils.second.z + 0.01, *exit_axis),
            (self.descan_coils.second.z + 0.02, *exit_axis),
        )

    def scan_point(self, num_rays: int, yx: Tuple[int, int]) -> Rays:
        self.move_to(yx)
        return self.run_to_end(num_rays)

    def scan_point_iter(
        self, num_rays: int, yx: Tuple[int, int]
    ) -> Generator[Rays, None, None]:
        self.move_to(yx)
        yield from self.run_iter(num_rays)

    def scan(
        self, num_rays: int
    ) -> Generator[Tuple[Tuple[int, int], Rays], None, None]:
        sy, sx = self.sample.scan_shape
        for y in range(sy):
            for x in range(sx):
                pos = (y, x)
                yield pos, self.scan_point(num_rays, pos)
