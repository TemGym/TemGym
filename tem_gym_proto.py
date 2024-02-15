import abc
from typing import Generator, Iterable, Tuple, Optional, Union, Type, TypeAlias, Self
from itertools import pairwise
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


from temgymbasic.functions import (
    circular_beam,
    point_beam,
    get_pixel_coords,
)


PositiveFloat: TypeAlias = float
NonNegativeFloat: TypeAlias = float


class UsageError(Exception):
    ...


@dataclass
class Rays:
    data: np.ndarray
    indices: np.ndarray
    location: Union[float, 'Component', Tuple['Component', ...]]

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
    def dx(self):
        return self.data[1, :]

    @property
    def dy(self):
        return self.data[3, :]

    @staticmethod
    def propagation_matix(z):
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
                self.propagation_matix(distance),
                self.data,
            ),
            indices=self.indices,
            location=self.z + distance,
        )

    def get_image(
        self,
        shape: Tuple[int, int],
        pixel_size: PositiveFloat,
        flip_y: bool = False,
        scan_rotation: float = 0.
    ):
        det = Detector(
            z=self.z,
            pixel_size=pixel_size,
            shape=shape,
            scan_rotation=scan_rotation,
            flip_y=flip_y,
        )
        return det.get_image(self)


class Component(abc.ABC):
    def __init__(self, z: float, name: str):
        self._name = name
        self._z = z

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
        return f'{self.__class__.__name__}: {self._name} @ z = {self._z}'

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
    def __init__(self, z: float, f: float, name: str = "Lens"):
        super().__init__(name=name, z=z)
        self._f = f

    @property
    def f(self) -> float:
        return self._f

    @f.setter
    def f(self, f: float):
        self._f = f

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
        )


class Sample(Component):
    def __init__(self, z: float, name: str = "Sample"):
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
        scan_step_yx: Tuple[PositiveFloat, PositiveFloat] = (0.01, 0.01),
        name: str = "STEMSample"
    ):
        super().__init__(name=name, z=z)
        self.overfocus = overfocus
        self.semiconv_angle = semiconv_angle
        self.scan_shape = scan_shape
        self.scan_step_yx = scan_step_yx

    def scan_position(self, yx: Tuple[int, int]) -> Tuple[float, float]:
        y, x = yx
        # Get the scan position in physical units
        scan_step_y, scan_step_x = self.scan_step_yx
        sy, sx = self.scan_shape
        scan_position_x = (x - sx / 2.) * scan_step_x
        scan_position_y = (y - sy / 2.) * scan_step_y
        return (scan_position_y, scan_position_x)


class Source(Component):
    def __init__(
        self, z: float, tilt_yx: Tuple[float, float] = (0., 0.), name: str = "Source"
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
        return Rays(data=r, indices=indices, location=self)

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
        name: str = "ParallelBeam",
    ):
        super().__init__(z=z, tilt_yx=tilt_yx, name=name)
        self.radius = radius

    def get_rays(self, num_rays: int) -> Rays:
        r, _ = circular_beam(num_rays, self.radius)
        return self._make_rays(r)


class PointSource(Component):
    def __init__(
        self,
        z: float,
        semi_angle: Optional[float] = None,
        tilt_yx: Tuple[float, float] = (0., 0.),
        name: str = "PointSource",
    ):
        super().__init__(name=name, z=z)
        self.semi_angle = semi_angle
        self.tilt_yx = tilt_yx

    def get_rays(self, num_rays: int) -> Rays:
        r, _ = point_beam(num_rays, self.semi_angle)
        return self._make_rays(r)


class Detector(Component):
    def __init__(
        self,
        z: float,
        pixel_size: float,
        shape: Tuple[int, int],
        scan_rotation: float = 0.,
        flip_y: bool = False,
        name: str = "Detector",
    ):
        super().__init__(name=name, z=z)
        self.pixel_size = pixel_size
        self.shape = shape
        self.scan_rotation = scan_rotation
        self.flip_y = flip_y

    def step(
        self, rays: Rays
    ) -> Generator[Rays, None, None]:
        # Detector has no effect on rays
        rays.location = self
        yield rays

    def get_image(self, rays: Rays) -> NDArray:
        # Convert rays from detector positions to pixel positions
        pixel_coords_y, pixel_coords_x = np.round(
            get_pixel_coords(
                rays_x=rays.x,
                rays_y=rays.y,
                shape=self.shape,
                pixel_size=self.pixel_size,
                flip_y=self.flip_y,
                scan_rotation=self.scan_rotation,
            )
        ).astype(int)
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
        name: str = "Deflector",
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
        )


class DoubleDeflector(Component):
    def __init__(
        self,
        first: Deflector,
        second: Deflector,
        name: str = "DoubleDeflector",
    ):
        super().__init__(
            z=(first.z + second.z) / 2,
            name=name,
        )
        self._first = first
        self._second = second
        assert self.first.z < self.second.z

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
        return (self.first.z + self.second.z) / 2

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
        rays = rays.propagate(
            self.second.entrance_z - self.first.exit_z,
        )
        for rays in self.second.step(rays):
            rays.location = (self, self.second)
            yield rays


class Aperture(Component):
    def __init__(
        self,
        z: float,
        radius_inner: float = 0.005,
        radius_outer: float = 0.25,
        x: float = 0.,
        y: float = 0.,
        name: str = 'Aperture',
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
            data=rays.data[:, mask], indices=rays.indices[mask], location=self
        )


class Model:
    def __init__(self, components: Iterable[Component]):
        self._components = components
        assert len(self._components) >= 2
        assert isinstance(self.source, Source)
        assert isinstance(self.detector, Detector)
        assert all(
            next_c.entrance_z > this_c.exit_z
            for this_c, next_c
            in pairwise(self._components)
        ), "Components must be sorted in increasing in z position with no overlap"

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
    def detector(self) -> Detector:
        return self.components[-1]

    def run_iter(
        self, num_rays: int
    ) -> Generator[Rays, None, None]:
        rays = None
        for this_component, next_component in pairwise(self.components):
            if rays is None:
                # At the gun
                this_component: Source
                rays = this_component.get_rays(num_rays)
            for rays in this_component.step(rays):
                yield rays
            rays = rays.propagate(
                next_component.entrance_z - this_component.exit_z,
            )
        # At detector plane
        next_component: Detector
        yield from next_component.step(rays)

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
    def __init__(
        self,
        overfocus: Optional[NonNegativeFloat] = None,
        semiconv_angle: Optional[PositiveFloat] = None,
        scan_step_yx: Optional[Tuple[PositiveFloat, PositiveFloat]] = None,
        scan_shape: Optional[Tuple[int, int]] = None,
    ):
        super().__init__(self.default_components())
        assert isinstance(self.source, ParallelBeam)
        self.set_stem_params(
            overfocus=overfocus,
            semiconv_angle=semiconv_angle,
            scan_step_yx=scan_step_yx,
            scan_shape=scan_shape,
        )

    @staticmethod
    def default_components():
        return (
            ParallelBeam(
                z=0.,
                radius=0.01,
            ),
            DoubleDeflector(
                first=Deflector(z=0.225),
                second=Deflector(z=0.275),
            ),
            Lens(
                z=0.5,
                f=-0.2,
            ),
            STEMSample(
                z=0.6,
            ),
            DoubleDeflector(
                first=Deflector(z=0.725),
                second=Deflector(z=0.775),
            ),
            Detector(
                z=1.,
                pixel_size=0.05,
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

    def set_stem_params(
        self,
        overfocus: Optional[NonNegativeFloat] = None,
        semiconv_angle: Optional[PositiveFloat] = None,
        scan_step_yx: Optional[Tuple[PositiveFloat, PositiveFloat]] = None,
        scan_shape: Optional[Tuple[int, int]] = None,
    ):
        if overfocus is not None:
            self.sample.overfocus = overfocus
        if semiconv_angle is not None:
            self.sample.semiconv_angle = semiconv_angle
        self.set_obj_lens_f_from_overfocus()
        self.set_beam_radius_from_semiconv()
        # These could be set externally, no invariants to hold
        if scan_step_yx is not None:
            self.sample.scan_step_yx = scan_step_yx
        if scan_shape is not None:
            self.sample.scan_shape = scan_shape

    def set_obj_lens_f_from_overfocus(self):
        if self.sample.overfocus < 0.:
            raise NotImplementedError(
                'Only support positive overfocus values (crossover above sample)'
            )
        # Lens f is always a negative number, and overfocus for now is always a positive number
        self.objective.f = -1 * (self.objective.z - self.sample.z - self.sample.overfocus)

    def set_beam_radius_from_semiconv(self):
        self.source.radius = abs(self.objective.f) * np.tan(self.sample.semiconv_angle)

    def update_scan_coil_ratios(self, scan_pixel_yx: Tuple[int, int]):
        # Distance to front focal plane from bottom deflector
        dist_to_lens = self.objective.z - self.scan_coils.exit_z
        dist_to_ffp = dist_to_lens - abs(self.objective.f)

        scan_position_y, scan_position_x = self.sample.scan_position(scan_pixel_yx)

        # Scan coil setting
        sc_length = self.scan_coils.length
        sc_defratio = -1 * (1 + sc_length / dist_to_ffp)

        self.scan_coils.first.defx = (
            scan_position_x / (sc_length + dist_to_lens * (1 + sc_defratio))
        )
        self.scan_coils.second.defx = sc_defratio * self.scan_coils.first.defx

        self.scan_coils.first.defy = (
            scan_position_y / (sc_length + dist_to_lens * (1 + sc_defratio))
        )
        self.scan_coils.second.defy = sc_defratio * self.scan_coils.first.defy

        # Descan coil setting
        desc_length = self.descan_coils.length
        # desc_defratio = -1 * (1 + desc_length / dist_to_ffp)

        self.descan_coils.first.defx = (
            -self.scan_coils.first.defx
            * (sc_length + dist_to_lens * (1 + sc_defratio)) / desc_length
        )
        self.descan_coils.second.defx = -self.descan_coils.first.defx

        self.descan_coils.first.defy = (
            -self.scan_coils.first.defy
            * (sc_length + dist_to_lens * (1 + sc_defratio)) / desc_length
        )
        self.descan_coils.second.defy = -self.descan_coils.first.defy

    def scan_point(self, num_rays: int, yx: Tuple[int, int]) -> Rays:
        self.update_scan_coil_ratios(yx)
        return self.run_to_end(num_rays)

    def scan_point_iter(
        self, num_rays: int, yx: Tuple[int, int]
    ) -> Generator[Rays, None, None]:
        self.update_scan_coil_ratios(yx)
        yield from self.run_iter(num_rays)

    def scan(
        self, num_rays: int
    ) -> Generator[Tuple[Tuple[int, int], Rays], None, None]:
        sy, sx = self.sample.scan_shape
        for y in range(sy):
            for x in range(sx):
                pos = (y, x)
                yield pos, self.scan_point(num_rays, pos)


class GUIModel:
    def __init__(self, model: Model):
        self._model = model
        self._gui_components = tuple(
            c.gui_wrapper()(c) for c in self._model._components
        )


if __name__ == '__main__':
    model = STEMModel()
    model.set_stem_params(semiconv_angle=0.01)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Variables to store the previous component's ray positions
    # prev_x_positions = None
    # prev_y_positions = None
    # prev_z = None

    # Iterate over components and their ray positions
    num_rays = 5
    yx = (5, 5)
    all_rays = tuple(model.scan_point_iter(num_rays=num_rays, yx=yx))

    xvals = np.stack(tuple(r.x for r in all_rays), axis=0)
    yvals = np.stack(tuple(r.x for r in all_rays), axis=0)
    zvals = np.asarray(tuple(r.z for r in all_rays))
    ax.plot(xvals, zvals)

    # Optional: Mark the component positions
    extent = 1.5 * np.abs(xvals).max()
    for component in model.components:
        if isinstance(component, DoubleDeflector):
            ax.hlines(
                component.first.z, -extent, extent, linestyle='--'
            )
            ax.text(-extent, component.first.z, repr(component.first), va='bottom')
            ax.hlines(
                component.second.z, -extent, extent, linestyle='--'
            )
            ax.text(-extent, component.second.z, repr(component.second), va='bottom')
        else:
            ax.hlines(component.z, -extent, extent, label=repr(component))
            ax.text(-extent, component.z, repr(component), va='bottom')

    ax.hlines(
        model.objective.z - model.objective.f, -extent, extent, linestyle=':'
    )

    ax.axvline(color='black', linestyle=":", alpha=0.3)
    _, scan_pos_x = model.sample.scan_position(yx)
    ax.plot([scan_pos_x], [model.sample.z], 'ko')

    ax.set_xlabel('x position')
    ax.set_ylabel('z position')
    ax.invert_yaxis()
    ax.set_title(f'Ray paths for {num_rays} rays at position {yx}')
    plt.show()

    # model.detector.shape = (512, 512)
    # model.detector.pixel_size = 0.002
    # image = model.detector.get_image(rays)

    # image = rays.get_image((512, 512), 0.002)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(rays.x, rays.y, 'o')
    # br = (np.asarray(model.detector.shape) // 2) * model.detector.pixel_size
    # b, r = br  # noqa
    # tl = -1 * br
    # t, l = tl  # noqa
    # tr = [t, r]
    # bl = [b, l]
    # points = np.stack((tl, tr, br, bl), axis=0)
    # ax.fill(points[:, 1], points[:, 0], facecolor='none', edgecolor='black')
    # ax.invert_yaxis()
    # ax.axis('equal')

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # plt.show()

    exit(0)

    components = (
        Gun(
            z=1.,
            beam_type="parallel",
            beam_radius=0.03,
            beam_semi_angle=0.001,
            beam_tilt_yx=(0., 0.),
        ),
        Lens(
            z=0.5,
            f=-0.05,
        ),
        DoubleDeflector(
            first=Deflector(
                z=0.275,
                defy=0.05,
                defx=0.05,
            ),
            second=Deflector(
                z=0.225,
                defy=-0.025,
                defx=0.,
            ),
        ),
        Sample(
            z=0.2
        ),
        Aperture(
            0.15,
            radius_inner=0.,
            radius_outer=0.0075,
        ),
        Detector(
            z=0.,
            pixel_size=0.01,
            shape=(128, 128),
        ),
    )
    model = Model(components)
    print(model)

    rays = model.run_to_component(model.detector, 512)
    image = model.detector.get_image(rays)

    import matplotlib.pyplot as plt
    plt.imshow(image)

    fig, ax = plt.subplots()
    ax.plot(rays.x, rays.y, 'o')
    br = (np.asarray(model.detector.shape) // 2) * model.detector.pixel_size
    b, r = br  # noqa
    tl = -1 * br
    t, l = tl  # noqa
    tr = [t, r]
    bl = [b, l]
    points = np.stack((tl, tr, br, bl), axis=0)
    ax.fill(points[:, 1], points[:, 0], facecolor='none', edgecolor='black')
    ax.invert_yaxis()
    ax.axis('equal')

    plt.show()

    # rays_y = det_rays[2]
    # rays_x = det_rays[0]
    # plt.plot(rays_x, rays_y, 'o')
    # plt.show()
