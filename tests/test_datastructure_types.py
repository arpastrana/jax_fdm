"""
Regression tests for the runtime neutrality of the typing-only mixins.

The mixins in `datastructures/types.py` must leave no trace at runtime: COMPAS's
JSON serializer walks the MRO of a datastructure calling `__clstype__()` on
every base above `Datastructure`, so a foreign class in the bases crashes
`to_json`. The mixin names are PEP 560 placeholders that erase themselves from
the bases; these tests pin that erasure and the serialization it protects.
"""

from compas.datastructures import Datastructure
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork


def test_typing_mixins_absent_from_runtime_mro():
    """
    The typing mixin classes must not appear in the runtime MRO.
    """
    for cls in (FDMesh, FDNetwork):
        names = [base.__name__ for base in cls.__mro__]
        assert not any(name.endswith("Type") for name in names), names


def test_typing_mixins_leave_compas_bases_first():
    """
    With the mixins erased, the COMPAS base is the first proper base.
    """
    assert issubclass(FDMesh, Datastructure)
    assert issubclass(FDNetwork, Datastructure)
    assert FDMesh.__mro__[1].__name__ == "Mesh"
    assert FDNetwork.__mro__[1].__name__ == "Graph"


def test_network_json_roundtrip(arch_network, tmp_path):
    """
    An FD network serializes to JSON and loads back as an FD network.
    """
    filepath = tmp_path / "network.json"
    arch_network.to_json(filepath)
    network = FDNetwork.from_json(filepath)

    assert type(network) is FDNetwork
    assert network.number_of_nodes() == arch_network.number_of_nodes()
    assert network.number_of_edges() == arch_network.number_of_edges()


def test_mesh_json_roundtrip(meshgrid_mesh, tmp_path):
    """
    An FD mesh serializes to JSON and loads back as an FD mesh.
    """
    filepath = tmp_path / "mesh.json"
    meshgrid_mesh.to_json(filepath)
    mesh = FDMesh.from_json(filepath)

    assert type(mesh) is FDMesh
    assert mesh.number_of_vertices() == meshgrid_mesh.number_of_vertices()
    assert mesh.number_of_faces() == meshgrid_mesh.number_of_faces()


def test_constructors_return_fd_subclass():
    """
    The stubbed COMPAS constructors build the FD subclass through `cls`.
    """
    mesh = FDMesh.from_meshgrid(2.0, 2)
    assert type(mesh) is FDMesh
    assert type(mesh.copy()) is FDMesh

    network = FDNetwork.from_lines([([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])])
    assert type(network) is FDNetwork
    assert type(network.copy()) is FDNetwork
