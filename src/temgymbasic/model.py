from typing import (
    Generator, Iterable, Tuple, Optional
)
from typing_extensions import Self
import numpy as np

from . import (
    PositiveFloat,
    UsageError,
    InvalidModelError,
)
from . import components as comp
from .rays import Rays
from .utils import pairwise


class Model:
    def __init__(self, components: Iterable[comp.Component]):
        self._components = components
        self._sort_components()
        self._validate_components()

    def _validate_components(self):
        if len(self._components) <= 1:
            raise InvalidModelError("Must have at least one component")
        if not isinstance(self.source, comp.Source):
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
    def components(self) -> Iterable[comp.Component]:
        return self._components

    def __repr__(self):
        repr_string = f'[{self.__class__.__name__}]:'
        for component in self.components:
            repr_string = repr_string + f'\n - {repr(component)}'
        return repr_string

    @property
    def source(self) -> comp.Source:
        return self.components[0]

    @property
    def detector(self) -> Optional[comp.Detector]:
        if isinstance(self.last, comp.Detector):
            return self.last
        return None

    @property
    def last(self) -> comp.Detector:
        return self.components[-1]

    def move_component(
        self,
        component: comp.Component,
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

    def add_component(self, component: comp.Component):
        original_components = tuple(self._components)
        self._components = tuple(self._components) + (component,)
        self._sort_components()
        try:
            self._validate_components()
        except InvalidModelError as err:
            # Unwind the change but reraise
            self._components = original_components
            raise err from None

    def remove_component(self, component: comp.Component):
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
        source: comp.Source = self.components[0]
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
        component: comp.Component,
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
        if not isinstance(self.source, comp.ParallelBeam):
            raise InvalidModelError("Must have a ParallelBeam for STEMModel")
        # Called here because even if all components are valid from
        # the perspective of z the current overfocus/semiconv
        # could forbid the new state, so we change the state during
        # validation such that it could be unwound if necessary
        self.set_stem_params()

    def _sort_components(self):
        """Component order fixed in STEMModel"""
        pass

    def add_component(self, component: comp.Component):
        raise UsageError("Cannot add components to STEMModel")

    def remove_component(self, component: comp.Component):
        raise UsageError("Cannot remove components from STEMModel")

    @staticmethod
    def default_components():
        """
        Just an initial valid state for the model
        """
        return (
            comp.ParallelBeam(
                z=0.0,
                radius=0.01,
            ),
            comp.DoubleDeflector(
                first=comp.Deflector(z=0.1, name='Upper'),
                second=comp.Deflector(z=0.15, name='Lower'),
                name='Scan Coil',
            ),
            comp.Lens(
                z=0.3,
                f=0.1,
            ),
            comp.STEMSample(
                z=0.5,
                name='STEM Sample'
            ),
            comp.DoubleDeflector(
                first=comp.Deflector(z=0.6, name='Upper'),
                second=comp.Deflector(z=0.625, name='Lower'),
                name='Descan Coil',
            ),
            comp.Detector(
                z=1.,
                pixel_size=0.01,
                shape=(128, 128),
            ),
        )

    @property
    def source(self) -> comp.ParallelBeam:
        return self.components[0]

    @property
    def scan_coils(self) -> comp.DoubleDeflector:
        return self.components[1]

    @property
    def objective(self) -> comp.Lens:
        return self.components[2]

    @property
    def sample(self) -> comp.STEMSample:
        return self.components[3]

    @property
    def descan_coils(self) -> comp.DoubleDeflector:
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
        *,
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
