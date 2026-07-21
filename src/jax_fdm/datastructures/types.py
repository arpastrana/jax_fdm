"""Typing-only mixins for the force density datastructures."""

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Self
from typing import TypeVar
from typing import overload

if TYPE_CHECKING:
    from compas.data import Data
    from compas.geometry import Transformation

__all__ = [
    "FDDatastructureType",
    "FDMeshType",
    "FDNetworkType",
]

# The `copy(cls=...)` mode returns an instance of the passed class, which need
# not be the FD subclass, so its overload cannot return `Self`.
CopyTarget = TypeVar("CopyTarget", bound="Data")

MixinClass = TypeVar("MixinClass")


class _TypingMixinPlaceholder:
    """
    A stand-in that removes itself from the bases of any class using it.
    """

    def __mro_entries__(self, bases: tuple[type, ...]) -> tuple[type, ...]:
        return ()


def _typing_only(mixin: MixinClass) -> MixinClass:
    """
    Erase a typing-only mixin class from the runtime bases of its subclasses.

    Static checkers see the decorated class unchanged. At runtime the name is
    rebound to a placeholder that PEP 560 ``__mro_entries__`` removes from the
    bases of any class listing it, so the runtime MRO is exactly as if the
    mixin were never there. A merely-empty runtime base would not do: COMPAS's
    JSON serializer walks the MRO calling ``__clstype__()`` on every base,
    which a foreign class breaks.
    """
    if TYPE_CHECKING:
        return mixin
    return _TypingMixinPlaceholder()


# The mixins below exist because COMPAS types its constructors with python-2
# comments as `-> Mesh` / `-> Data`, which hides every FD-specific method from
# a static checker downstream of e.g. `FDMesh.from_meshgrid(...)`, and because
# its getter/setter dual accessors return broad inferred unions. The stub
# bodies never execute; at runtime the COMPAS methods construct through `cls`
# and already return the FD subclass, so the `Self` signatures are truthful.
# Non-overload stubs raise NotImplementedError instead of `...` because
# pylint's inference reads a `...` body as returning None and flags call
# sites with E1111/E1133; a raising body is opaque to it. Overload variants
# keep `...` — they resolve through the trailing declaration.
# Keep parameter names and defaults in sync with COMPAS 2.x.


@_typing_only
class FDDatastructureType:
    """
    Typing-only declarations of the COMPAS accessors `FDDatastructure` calls.

    Notes
    -----
    The declared methods exist on the concrete subclasses' COMPAS bases
    (`Mesh` and `Graph`), which the `FDDatastructure` mixin deliberately
    does not inherit. Widths are chosen so that both bases' inferred
    signatures remain assignable, and the first parameter of the per-edge
    accessors is positional-only because `Mesh` names it ``edge`` while
    `Graph` names it ``key``.
    """

    def edges(self, data: bool = False) -> Iterator[Any]:
        raise NotImplementedError

    def edge_attribute(
        self,
        key: tuple[int, int],
        /,
        name: str,
        value: Any = None,
    ) -> Any:
        raise NotImplementedError

    def edge_attributes(
        self,
        key: tuple[int, int],
        /,
        names: Iterable[str] | None = None,
        values: Iterable[Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    def edges_attribute(
        self,
        name: str,
        value: Any = None,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> Any:
        raise NotImplementedError

    def edges_attributes(
        self,
        names: Iterable[str] | None = None,
        values: Iterable[Any] | None = None,
        keys: Iterable[tuple[int, int]] | None = None,
    ) -> Any:
        raise NotImplementedError


@_typing_only
class FDMeshType:
    """
    Typing-only declarations of the COMPAS API surface `FDMesh` re-narrows.
    """

    @classmethod
    def from_meshgrid(
        cls,
        dx: float,
        nx: int,
        dy: float | None = None,
        ny: int | None = None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def from_obj(
        cls,
        filepath: str | PathLike[str],
        precision: str | None = None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def from_json(cls, filepath: str | PathLike[str]) -> Self:
        raise NotImplementedError

    @classmethod
    def from_vertices_and_faces(
        cls,
        vertices: Sequence[Sequence[float]] | dict[int, Sequence[float]],
        faces: Sequence[Sequence[int]] | dict[int, Sequence[int]],
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def from_polyhedron(cls, f: int) -> Self:
        raise NotImplementedError

    # copy() without cls duplicates the FD datastructure; copy(cls=Other)
    # constructs and returns an Other, so only the first mode is Self
    @overload
    def copy(self, cls: None = None, copy_guid: bool = False) -> Self: ...
    @overload
    def copy(
        self,
        cls: type[CopyTarget],
        copy_guid: bool = False,
    ) -> CopyTarget: ...
    def copy(
        self,
        cls: type[Any] | None = None,
        copy_guid: bool = False,
    ) -> Any:
        raise NotImplementedError

    def transformed(self, transformation: "Transformation") -> Self:
        raise NotImplementedError

    def subdivided(self, scheme: str = "catmullclark", **options: Any) -> Self:
        raise NotImplementedError

    # The accessors below are getter/setter duals in COMPAS, whose single
    # implementation returns a broad inferred union. The overloads split
    # the call modes; attribute values stay Any since they are
    # heterogeneous. Each overload group closes with a non-overloaded
    # declaration standing in for the inherited implementation.

    @overload
    def vertices(self, data: Literal[False] = False) -> Iterator[int]: ...
    @overload
    def vertices(
        self,
        data: Literal[True],
    ) -> Iterator[tuple[int, dict[str, Any]]]: ...
    def vertices(self, data: bool = False) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def vertices_where(
        self,
        conditions: dict[str, Any] | None = None,
        data: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[int]: ...
    @overload
    def vertices_where(
        self,
        conditions: dict[str, Any] | None = None,
        *,
        data: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, dict[str, Any]]]: ...
    def vertices_where(
        self,
        conditions: dict[str, Any] | None = None,
        data: bool = False,
        **kwargs: Any,
    ) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def vertex_attribute(self, key: int, name: str) -> Any: ...
    @overload
    def vertex_attribute(self, key: int, name: str, value: Any) -> None: ...
    def vertex_attribute(self, key: int, name: str, value: Any = None) -> Any:
        raise NotImplementedError

    def vertex_attributes(
        self,
        key: int,
        names: Iterable[str] | None = None,
        values: Iterable[Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    def vertex_coordinates(self, key: int, axes: str = "xyz") -> list[float]:
        raise NotImplementedError

    def index_vertex(self) -> dict[int, int]:
        raise NotImplementedError

    @overload
    def edges(self, data: Literal[False] = False) -> Iterator[tuple[int, int]]: ...
    @overload
    def edges(
        self,
        data: Literal[True],
    ) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]: ...
    def edges(self, data: bool = False) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def edge_attribute(self, edge: tuple[int, int], name: str) -> Any: ...
    @overload
    def edge_attribute(
        self,
        edge: tuple[int, int],
        name: str,
        value: Any,
    ) -> None: ...
    def edge_attribute(
        self,
        edge: tuple[int, int],
        name: str,
        value: Any = None,
    ) -> Any:
        raise NotImplementedError


@_typing_only
class FDNetworkType:
    """
    Typing-only declarations of the COMPAS API surface `FDNetwork` re-narrows.
    """

    # loose lines param: point pairs, COMPAS Line objects, or Polyline
    # segments all unpack as two points at runtime
    @classmethod
    def from_lines(
        cls,
        lines: Iterable[Any],
        precision: int | None = None,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def from_json(cls, filepath: str | PathLike[str]) -> Self:
        raise NotImplementedError

    # loose params: the node and edge containers routinely come straight
    # from untyped COMPAS iterators (mesh.vertices(), mesh.edges()), whose
    # inferred unions a precise signature would reject
    @classmethod
    def from_nodes_and_edges(cls, nodes: Any, edges: Any) -> Self:
        raise NotImplementedError

    # copy() without cls duplicates the FD datastructure; copy(cls=Other)
    # constructs and returns an Other, so only the first mode is Self
    @overload
    def copy(self, cls: None = None, copy_guid: bool = False) -> Self: ...
    @overload
    def copy(
        self,
        cls: type[CopyTarget],
        copy_guid: bool = False,
    ) -> CopyTarget: ...
    def copy(
        self,
        cls: type[Any] | None = None,
        copy_guid: bool = False,
    ) -> Any:
        raise NotImplementedError

    def transformed(self, transformation: "Transformation") -> Self:
        raise NotImplementedError

    # Getter/setter dual accessors, split as on FDMeshType but in the
    # graph vocabulary (`Graph` names the per-element key `key`).

    @overload
    def nodes(self, data: Literal[False] = False) -> Iterator[int]: ...
    @overload
    def nodes(
        self,
        data: Literal[True],
    ) -> Iterator[tuple[int, dict[str, Any]]]: ...
    def nodes(self, data: bool = False) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def nodes_where(
        self,
        conditions: dict[str, Any] | None = None,
        data: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[int]: ...
    @overload
    def nodes_where(
        self,
        conditions: dict[str, Any] | None = None,
        *,
        data: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, dict[str, Any]]]: ...
    def nodes_where(
        self,
        conditions: dict[str, Any] | None = None,
        data: bool = False,
        **kwargs: Any,
    ) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def node_attribute(self, key: int, name: str) -> Any: ...
    @overload
    def node_attribute(self, key: int, name: str, value: Any) -> None: ...
    def node_attribute(self, key: int, name: str, value: Any = None) -> Any:
        raise NotImplementedError

    def node_attributes(
        self,
        key: int,
        names: Iterable[str] | None = None,
        values: Iterable[Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    def node_coordinates(self, key: int, axes: str = "xyz") -> list[float]:
        raise NotImplementedError

    def index_node(self) -> dict[int, int]:
        raise NotImplementedError

    def index_edge(self) -> dict[int, tuple[int, int]]:
        raise NotImplementedError

    @overload
    def edges(self, data: Literal[False] = False) -> Iterator[tuple[int, int]]: ...
    @overload
    def edges(
        self,
        data: Literal[True],
    ) -> Iterator[tuple[tuple[int, int], dict[str, Any]]]: ...
    def edges(self, data: bool = False) -> Iterator[Any]:
        raise NotImplementedError

    @overload
    def edge_attribute(self, key: tuple[int, int], name: str) -> Any: ...
    @overload
    def edge_attribute(
        self,
        key: tuple[int, int],
        name: str,
        value: Any,
    ) -> None: ...
    def edge_attribute(
        self,
        key: tuple[int, int],
        name: str,
        value: Any = None,
    ) -> Any:
        raise NotImplementedError
