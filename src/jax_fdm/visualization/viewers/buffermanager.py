import numpy as np
from compas_viewer.gl import update_vertex_buffer
from compas_viewer.scene.buffermanager import BufferManager

from compas.colors import Color

__all__ = ["FastBufferManager"]


class FastBufferManager(BufferManager):
    """
    A buffer manager with a vectorized in-place update.

    ``BufferManager.update_object_data`` (compas_viewer 2.0.2) locates an
    object's slice in the combined render buffers with a pure-Python linear
    scan over the per-vertex object indices, which makes every in-place update
    O(total scene vertices). This subclass mirrors that method but finds the
    offset with a numpy lookup, so per-frame updates of large batched buffers
    (or of a mesh living late in the combined buffer) stay cheap.
    """

    def update_object_data(self, obj):
        """
        Update the position and color buffers for a single object.
        """
        if obj not in self.objects:
            return

        index = self.objects[obj]

        data_types = ["_points_data", "_lines_data", "_frontfaces_data", "_backfaces_data"]
        for data_type in data_types:
            if hasattr(obj, data_type) and getattr(obj, data_type):
                setattr(obj, data_type, getattr(obj, f"_read{data_type}")())
                data = getattr(obj, data_type)

                if not self.buffer_ids[data_type]:
                    continue

                # Elements are not updated: topology is fixed at add time.
                positions, colors, _ = data

                if len(colors) > len(positions):
                    colors = colors[: len(positions)]
                elif len(colors) < len(positions):
                    colors = colors + [colors[-1]] * (len(positions) - len(colors))

                pos_array = np.array(positions, dtype=np.float32).flatten()
                col_array = np.array([c.rgba for c in colors] if len(colors) > 0 and isinstance(colors[0], Color) else colors, dtype=np.float32).flatten()

                # numpy offset lookup instead of the upstream per-entry Python scan
                matches = np.flatnonzero(self.object_indices[data_type] == index)
                start_idx = int(matches[0]) if len(matches) else 0

                pos_byte_offset = start_idx * 3 * 4  # 3 floats per vertex * 4 bytes per float
                update_vertex_buffer(pos_array, self.buffer_ids[data_type]["positions"], offset=pos_byte_offset)

                col_byte_offset = start_idx * 4 * 4  # 4 floats per color * 4 bytes per float
                update_vertex_buffer(col_array, self.buffer_ids[data_type]["colors"], offset=col_byte_offset)
