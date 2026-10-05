"""
Read-only source indexing for ``ValidationError`` instances.

A ``SourceIndex`` maps each ``ValidationError`` back to the actual schema
node in a ``referencing.Registry`` snapshot that produced it.  It associates
the error's ``absolute_schema_path`` and the reference URIs that were
followed with concrete nodes, preserving cross-document identity: a
``$ref`` into another document resolves to the node *in that document*, not
a copy.

The index is bound to a specific registry snapshot.  Once created, using it
with a different registry raises ``SnapshotMismatch``; the original errors
and schemas are never modified, and repeated references to the same node
share a single cached localization.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from attrs import frozen
import referencing.exceptions

from jsonschema import _utils

if TYPE_CHECKING:
    from jsonschema import exceptions

_unset = _utils.Unset()


def _escape_pointer(part: str) -> str:
    """Escape a single JSON Pointer reference token (RFC 6901)."""
    return part.replace("~", "~0").replace("/", "~1")


def _find_pointer(
    contents: Any,
    target: Any,
    pointer: str = "",
) -> str | None:
    """
    Return the JSON Pointer of ``target`` within ``contents`` by identity.

    The comparison is done with ``is`` so that the actual node object (not an
    equal copy) is located.  Returns ``None`` when the target is not present.
    """
    if contents is target:
        return pointer
    if isinstance(contents, Mapping):
        for key, value in contents.items():
            result = _find_pointer(
                value,
                target,
                pointer + "/" + _escape_pointer(str(key)),
            )
            if result is not None:
                return result
    elif isinstance(contents, (list, tuple)):
        for index, value in enumerate(contents):
            result = _find_pointer(value, target, pointer + "/" + str(index))
            if result is not None:
                return result
    return None


@frozen
class SourceLocation:
    """
    The source schema node that caused a validation error.

    Attributes:

        node:
            The actual schema node object from the registry.

        uri:
            The URI of the resource containing the node.

        pointer:
            The JSON Pointer within the resource that locates the node.

        keyword:
            The schema keyword that failed (e.g. ``"type"``), when known.

        ref_chain:
            The reference URIs followed to reach the node, when known.

    """

    node: Mapping[str, Any] | bool
    uri: str
    pointer: str
    keyword: str | None
    ref_chain: tuple[str, ...] = ()


@frozen
class ReferenceFailure:
    """
    A reference that could not be resolved.

    Attributes:

        ref:
            The unresolvable reference URI.

        message:
            A human-readable description of the failure.

    """

    ref: str
    message: str = ""


class SnapshotMismatch(Exception):
    """
    A `SourceIndex` was used with a registry that is not its bound snapshot.
    """


class SourceIndex:
    """
    Locate ``ValidationError`` instances to their source schema nodes.

    A read-only index that maps each ``ValidationError`` back to the actual
    schema node in a ``referencing.Registry`` snapshot that produced it.  It
    associates the error's ``absolute_schema_path`` and the reference URIs
    that were followed with concrete nodes, preserving cross-document
    identity: a ``$ref`` into another document resolves to the node *in that
    document*, not a copy.

    The index is bound to a specific registry snapshot.  Once created, using
    it with a different registry raises ``SnapshotMismatch``; the original
    errors and schemas are never modified, and repeated references to the
    same node share a single cached localization.

    Arguments:

        registry:
            The `referencing.Registry` snapshot to bind the index to.

    """

    def __init__(self, registry: referencing.Registry) -> None:
        self._registry = registry
        self._cache: dict[int, SourceLocation] = {}

    @property
    def registry(self) -> referencing.Registry:
        """The registry snapshot this index is bound to."""
        return self._registry

    def locate(
        self,
        error: exceptions.ValidationError,
        *,
        registry: referencing.Registry | None = None,
    ) -> SourceLocation | None:
        """
        Locate the source schema node for a `ValidationError`.

        Returns ``None`` when the error's schema node cannot be found in the
        bound registry.

        Arguments:

            error:
                The `ValidationError` to locate.

            registry:
                An optional registry snapshot that must match the bound
                snapshot.

        Raises:

            SnapshotMismatch:
                if ``registry`` is provided and is not the bound snapshot.

        """
        self._check_snapshot(registry)
        node = error.schema
        if node is _unset or node is None:
            return None

        key = id(node)
        if key in self._cache:
            return self._cache[key]

        located = self._find_node(node)
        if located is None:
            return None

        uri, pointer = located
        keyword = error.validator if isinstance(error.validator, str) else None
        location = SourceLocation(
            node=node,
            uri=uri,
            pointer=pointer,
            keyword=keyword,
            ref_chain=(),
        )
        self._cache[key] = location
        return location

    def locate_ref(
        self,
        ref: str,
        *,
        base_uri: str = "",
        registry: referencing.Registry | None = None,
    ) -> SourceLocation | ReferenceFailure:
        """
        Resolve a reference URI, returning its location or a failure.

        Arguments:

            ref:
                The reference URI to resolve.

            base_uri:
                The base URI to resolve the reference against.

            registry:
                An optional registry snapshot that must match the bound
                snapshot.

        Raises:

            SnapshotMismatch:
                if ``registry`` is provided and is not the bound snapshot.

        """
        self._check_snapshot(registry)
        resolver = self._registry.resolver(base_uri=base_uri)
        try:
            resolved = resolver.lookup(ref)
        except referencing.exceptions.Unresolvable as exc:
            return ReferenceFailure(ref=ref, message=str(exc))

        node = resolved.contents
        located = self._find_node(node)
        if located is None:
            return ReferenceFailure(
                ref=ref,
                message="resolved node was not found in the registry snapshot",
            )

        uri, pointer = located
        return SourceLocation(
            node=node,
            uri=uri,
            pointer=pointer,
            keyword=None,
            ref_chain=(ref,),
        )

    def locate_failure(
        self,
        failure: BaseException,
        *,
        registry: referencing.Registry | None = None,
    ) -> ReferenceFailure:
        """
        Classify a reference resolution failure as a `ReferenceFailure`.

        Arguments:

            failure:
                The `referencing.exceptions.Unresolvable` (or a subclass)
                raised during validation.

            registry:
                An optional registry snapshot that must match the bound
                snapshot.

        Raises:

            SnapshotMismatch:
                if ``registry`` is provided and is not the bound snapshot.

        """
        self._check_snapshot(registry)
        ref = getattr(failure, "ref", None)
        if ref is None:
            return ReferenceFailure(ref="", message=str(failure))
        return ReferenceFailure(ref=ref, message=str(failure))

    def _check_snapshot(
        self,
        registry: referencing.Registry | None,
    ) -> None:
        if registry is not None and registry is not self._registry:
            raise SnapshotMismatch(
                "SourceIndex is bound to a different registry snapshot; "
                f"expected {self._registry!r}, got {registry!r}",
            )

    def _find_node(self, node: Any) -> tuple[str, str] | None:
        """
        Find the resource and JSON Pointer of a schema node by identity.
        """
        for uri, resource in self._registry.items():
            contents = resource.contents
            pointer = _find_pointer(contents, node)
            if pointer is not None:
                return uri, pointer
        return None
