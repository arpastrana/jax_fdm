from compas_viewer.components import Treeform
from compas_viewer.components.objectsetting import ObjectSetting

from compas.geometry import length_vector
from jax_fdm.visualization.viewers.scene_objects import FDEdgeObject
from jax_fdm.visualization.viewers.scene_objects import FDLoadObject
from jax_fdm.visualization.viewers.scene_objects import FDObject
from jax_fdm.visualization.viewers.scene_objects import FDPointObject
from jax_fdm.visualization.viewers.scene_objects import FDReactionObject

__all__ = ["FDObjectSetting"]


class FDObjectSetting(ObjectSetting):
    """
    The object settings tab, extended with a force density readout.

    Selecting a per-element scene object (an edge, point, load or reaction of
    a force density datastructure) shows its attributes read live from the
    datastructure, plus a one-line summary in the status bar. The readout is
    read-only: editing an attribute like the force density would call for a
    re-solve, which is beyond the viewer. Any other object populates the
    native settings widgets.
    """

    def populate(self, obj):
        if not isinstance(obj, FDObject):
            return super().populate(obj)

        self.reset()
        title, attributes = self._element_attributes(obj)

        form = Treeform(show_headers=False)
        form.update_from_dict({title: attributes})
        form.widget.expandAll()
        self.add(form)

        statusbar = getattr(self.viewer.ui, "statusbar", None)
        if statusbar is not None:
            summary = " | ".join([title] + [f"{name}: {value}" for name, value in attributes.items()])
            statusbar.widget.showMessage(summary, 5000)

    @staticmethod
    def _element_attributes(obj):
        """
        The title and attribute values of one force density element.
        """
        parent = obj.fdparent
        datastructure = parent.datastructure
        key = obj.key

        if isinstance(obj, FDEdgeObject):
            return f"Edge {key}", {"q": f"{datastructure.edge_forcedensity(key):.4g}",
                                   "force": f"{datastructure.edge_force(key):.4g}",
                                   "length": f"{datastructure.edge_length(key):.4g}"}

        if isinstance(obj, FDPointObject):
            x, y, z = parent.point_coordinates(key)
            return f"{parent.point_name} {key}", {"xyz": f"({x:.4g}, {y:.4g}, {z:.4g})",
                                                  "support": f"{parent.point_is_support(key)}",
                                                  "load": _vector_value(parent.point_load(key)),
                                                  "reaction": _vector_value(parent.point_reaction(key))}

        if isinstance(obj, FDLoadObject):
            return f"Load ({key})", {"vector": _vector_value(parent.point_load(key))}

        if isinstance(obj, FDReactionObject):
            return f"Reaction ({key})", {"vector": _vector_value(parent.point_reaction(key))}

        return obj.name, {}


def _vector_value(vector):
    x, y, z = vector
    return f"({x:.4g}, {y:.4g}, {z:.4g}), norm: {length_vector(vector):.4g}"
