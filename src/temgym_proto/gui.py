from . import Model, Component


class ComponentGUIWrapper:
    def __init__(self, component: Component):
        self.component = component


class GUIModel:
    def __init__(self, model: Model):
        self._model = model
        self._gui_components = tuple(
            c.gui_wrapper()(c) for c in self._model._components
        )
