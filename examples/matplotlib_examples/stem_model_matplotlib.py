from typing import Tuple, TypedDict
from temgymbasic.model import STEMModel
from temgymbasic.plotting import plot_model
import matplotlib.pyplot as plt


class OverfocusParams(TypedDict):
    overfocus: float  # m
    scan_pixel_size: float  # m
    camera_length: float  # m
    detector_pixel_size: float  # m
    semiconv: float  # rad
    cy: float
    cx: float
    scan_rotation: float
    scan_shape: Tuple[int, int]
    flip_y: bool


def make_model_proto(params: OverfocusParams):
    model = STEMModel()
    model.move_component(model.scan_coils.first, 0.1)
    model.move_component(model.scan_coils.second, 0.15)
    model.move_component(model.objective, 0.2)
    model.move_component(model.sample, 0.225)
    model.move_component(model.descan_coils.first, 0.25)
    model.move_component(model.descan_coils.second, 0.3)
    
    return model.set_stem_params(
        camera_length=params['camera_length'],
        semiconv_angle=params['semiconv'],
        scan_step_yx=(
            params['scan_pixel_size'],
            params['scan_pixel_size'],
        ),
        scan_shape=params['scan_shape'],
        overfocus=params['overfocus'],
    )


dataset_shape = [100, 100]
overfocus_params = OverfocusParams(
    overfocus=0.01,  # m
    scan_pixel_size=0.01,  # m
    camera_length=0.15,  # m
    detector_pixel_size=0.050,  # m
    semiconv=5,  # rad
    scan_rotation=0,
    flip_y=False,
    scan_shape=tuple(dataset_shape),
    # Offset to avoid subchip gap
    cy=100,
    cx=100,
)

model = make_model_proto(overfocus_params)

# Step the rays through the model to get the ray positions throughout the column
model.move_to((0, 72))
fig, ax = plot_model(model)

plt.show()
