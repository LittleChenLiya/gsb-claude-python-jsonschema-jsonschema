"""
Read-only source indexing for `jsonschema.ValidationError` instances.

A `SourceIndex` correlates a validation error's ``absolute_schema_path``
and the ``$ref`` URIs which led to it with the *actual* schema nodes in
the documents which were validated against, preserving the cross-document
identity of ``$ref`` targets.

The index is bound to an immutable snapshot of a `referencing.Registry`
(see `RegistrySnapshot` / `SnapshotStore`).  It never mutates the errors,
the schemas, or the registry it is bound to, and locating the same node
repeatedly (e.g. via several ``$ref`` references pointing at it) shares a
single location result.

Two failure domains are reported separately:

* validation failures locate successfully to a `SchemaLocation`, while
* reference failures (e.g. a ``$ref`` whose target, such as a ``$defs``
  entry, does not exist) raise `ReferenceFailure` instead of being
  reported as ordinary validation errors.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urldefrag, urljoin

from referencing.exceptions import Unresolvable
import referencing.jsonschema

from jsonschema.exceptions import ValidationError

if TYPE_CHECKING:
    from referencing.typing import URI

__all__ = [
    "ReferenceFailure",
    "RegistrySnapshot",
    "SchemaLocation",
    "SchemaNodeNotFound",
    "SnapshotStore",
    "SourceIndex",
    "SourceIndexError",
    "StaleSnapshotError",
]


class SourceIndexError(Exception):
    """Base class for errors raised by source indexing."""


class StaleSnapshotError(SourceIndexError):
    """
    The index is bound to a registry snapshot which has been swapped out.

    Build a fresh index from the store's current snapshot instead of
    reusing this one.
    """


class ReferenceFailure(SourceIndexError):
    """
    A reference (e.g. a ``$ref`` URI) could not be resolved.

    This is reported separately from validation failures: it means the
    schema's references are broken against the bound registry snapshot,
    not that some instance failed validation.
    """

    def __init__(self, ref: str, base_uri: URI = ""):
        super().__init__(
            f"Cannot resolve reference {ref!r} (base URI {base_uri!r}) "
            "against this registry snapshot.",
        )
        self.ref = ref
        self.base_uri = base_uri


class SchemaNodeNotFound(SourceIndexError):
    """
    The error's schema node is not part of the bound registry snapshot.

    The error presumably originates from a different registry or schema
    object than the ones this index was built from.
    """


def _json_pointer_escape(segment: str | int) -> str:
    return str(segment).replace("~", "~0").replace("/", "~1")


def _pointer_from_fragment(fragment: str) -> tuple[str | int, ...] | None:
    """
    Parse a URI fragment into JSON pointer segments.

    Returns `None` for plain-name (anchor) fragments, whose position
    within their resource is not derivable from the fragment alone.
    """
    if not fragment:
        return ()
    if not fragment.startswith("/"):
        return None
    return tuple(
        unquote(segment).replace("~1", "/").replace("~0", "~")
        for segment in fragment[1:].split("/")
    )


@dataclass(frozen=True)
class SchemaLocation:
    """
    The read-only location of a schema node within a registry snapshot.

    Attributes:

        uri: the retrieval URI of the document (resource) containing the
            node -- this is what preserves ``$ref`` cross-document
            identity

        pointer: JSON pointer segments to the node within that document,
            or `None` if the position is not known (e.g. the node was
            reached via an anchor in a document outside the snapshot)

        node: the actual schema node object, shared (never copied or
            mutated) with the schema documents

    """

    uri: URI
    pointer: tuple[str | int, ...] | None
    node: Any

    @property
    def json_pointer(self) -> str | None:
        """The location as a JSON pointer string, if known."""
        if self.pointer is None:
            return None
        return "".join("/" + _json_pointer_escape(s) for s in self.pointer)

    @property
    def canonical_uri(self) -> str:
        """The location as a full URI (with fragment, if known)."""
        pointer = self.json_pointer
        if pointer is None:
            return self.uri
        return f"{self.uri}#{pointer}"


@dataclass(frozen=True)
class RegistrySnapshot:
    """
    An immutable view of a `referencing.Registry` plus a root schema.

    Each snapshot carries a unique identity `token`; `SourceIndex`
    instances bind to one specific snapshot and refuse to answer once
    their `SnapshotStore` has swapped it for another one.
    """

    registry: Any  # referencing.Registry, typed loosely to stay generic
    root_schema: Any = None
    root_uri: URI = ""
    token: object = field(default_factory=object, repr=False, compare=False)


class SnapshotStore:
    """
    Holds the *current* registry snapshot.

    Swapping the snapshot (e.g. because schemas were reloaded) retires
    every `SourceIndex` created earlier: they raise `StaleSnapshotError`
    rather than silently answering against stale documents.
    """

    def __init__(
        self,
        registry,
        root_schema: Any = None,
        root_uri: URI = "",
    ):
        self._current = RegistrySnapshot(
            registry=registry,
            root_schema=root_schema,
            root_uri=root_uri,
        )

    @property
    def current(self) -> RegistrySnapshot:
        """The snapshot indexes are currently built from."""
        return self._current

    def swap(
        self,
        registry,
        root_schema: Any = None,
        root_uri: URI = "",
    ) -> RegistrySnapshot:
        """
        Replace the current snapshot, retiring pre-existing indexes.
        """
        self._current = RegistrySnapshot(
            registry=registry,
            root_schema=root_schema,
            root_uri=root_uri,
        )
        return self._current

    def index(self) -> SourceIndex:
        """Create a new `SourceIndex` bound to the current snapshot."""
        return SourceIndex(self)


class SourceIndex:
    """
    A read-only index mapping validation errors to schema source nodes.

    The index is bound to one `RegistrySnapshot` of a `SnapshotStore`;
    once the store swaps snapshots, this index refuses further queries.

    Locations are computed by object identity against the documents in
    the snapshot, so repeated references (``$ref``) to the same node
    share a single location result, and nodes keep the URI of the
    document they really live in.
    """

    def __init__(self, store: SnapshotStore):
        self._store = store
        self._snapshot = store.current
        self._nodes: dict[int, SchemaLocation] = {}
        self._reference_cache: dict[tuple[URI, str], SchemaLocation] = {}
        self._registry = store.current.registry
        self._default_base = store.current.root_uri
        self._built = False

    @classmethod
    def bind(
        cls,
        registry,
        root_schema: Any = None,
        root_uri: URI = "",
    ) -> SourceIndex:
        """
        Create an index over a fixed snapshot of the given registry.

        For registries which change over time, prefer keeping a
        `SnapshotStore` and calling `SnapshotStore.index` instead.
        """
        return SnapshotStore(
            registry,
            root_schema=root_schema,
            root_uri=root_uri,
        ).index()

    @property
    def snapshot(self) -> RegistrySnapshot:
        """The snapshot this index is bound to."""
        return self._snapshot

    def locate(self, error: ValidationError) -> SchemaLocation:
        """
        Locate the schema node a validation error originated from.

        The error's ``schema`` is found by identity within the snapshot
        (so ``$ref`` crossings, which ``absolute_schema_path`` omits, do
        not need to be guessed), then the walkable tail of
        ``absolute_schema_path`` refines the location to the failing
        keyword.  Neither the error nor any schema is modified.

        Raises:

            `StaleSnapshotError`: the bound snapshot was swapped out

            `ReferenceFailure`: the error *is* a reference resolution
                failure (e.g. an unresolvable ``$ref`` such as a missing
                ``$defs`` entry) rather than a validation failure

            `SchemaNodeNotFound`: the error's schema does not belong to
                the bound snapshot

        """
        self._check_fresh()
        if isinstance(error, Unresolvable):
            raise ReferenceFailure(
                ref=getattr(error, "ref", "<unknown>"),
            ) from error
        if not isinstance(error, ValidationError):
            raise TypeError(
                f"Cannot locate {type(error).__name__}; "
                "expected a ValidationError.",
            )

        self._ensure_built()
        base = self._nodes.get(id(error.schema))
        if base is None:
            raise SchemaNodeNotFound(
                "The error's schema node is not part of this index's "
                "registry snapshot; the error likely originates from "
                "different schema documents.",
            )

        node, pointer = base.node, base.pointer
        path = list(error.absolute_schema_path)
        for start in range(len(path)):
            walked = _walk_from(base.node, path[start:])
            if walked is not _NOT_FOUND:
                node = walked
                if pointer is not None:
                    pointer = pointer + tuple(path[start:])
                break
        return SchemaLocation(uri=base.uri, pointer=pointer, node=node)

    def locate_node(self, node: Any) -> SchemaLocation:
        """
        Locate an arbitrary schema node (by identity) in the snapshot.
        """
        self._check_fresh()
        self._ensure_built()
        location = self._nodes.get(id(node))
        if location is None:
            raise SchemaNodeNotFound(
                "The given node is not part of this index's registry "
                "snapshot.",
            )
        return location

    def resolve_reference(
        self,
        ref: str,
        base_uri: URI | None = None,
    ) -> SchemaLocation:
        """
        Resolve a ``$ref`` URI to the location of its target node.

        Repeated resolutions of the same reference share one cached
        location result.

        Raises:

            `StaleSnapshotError`: the bound snapshot was swapped out

            `ReferenceFailure`: the reference cannot be resolved against
                the bound snapshot (e.g. a missing ``$defs`` entry)

        """
        self._check_fresh()
        self._ensure_built()
        base = self._default_base if base_uri is None else base_uri
        key = (base, ref)
        location = self._reference_cache.get(key)
        if location is not None:
            return location

        resolver = self._registry.resolver(base)
        try:
            resolved = resolver.lookup(ref)
        except Unresolvable as error:
            raise ReferenceFailure(ref=ref, base_uri=base) from error

        location = self._nodes.get(id(resolved.contents))
        if location is None:
            uri, fragment = urldefrag(urljoin(base, ref))
            location = SchemaLocation(
                uri=uri,
                pointer=_pointer_from_fragment(fragment),
                node=resolved.contents,
            )
        self._reference_cache[key] = location
        return location

    def _check_fresh(self):
        if self._store.current is not self._snapshot:
            raise StaleSnapshotError(
                "This index is bound to a registry snapshot which has "
                "been swapped out; create a new index from the store's "
                "current snapshot.",
            )

    def _ensure_built(self):
        if self._built:
            return
        snapshot = self._snapshot
        root = snapshot.root_schema
        if root is None and snapshot.root_uri:
            resource = snapshot.registry.get(snapshot.root_uri)
            if resource is not None:
                root = resource.contents

        # The effective registry pairs the root schema with its URI, so
        # that root-relative references (e.g. "#/$defs/...") resolve the
        # same way they did during validation.
        registry = snapshot.registry
        self._default_base = snapshot.root_uri
        if root is not None:
            resource = _resource_for(root)
            uri = resource.id() or snapshot.root_uri
            registry = registry.with_resource(uri, resource)
            self._default_base = uri
            self._walk(root, uri, ())
        self._registry = registry

        for uri, resource in registry.crawl().items():
            self._walk(resource.contents, uri, ())
        self._built = True

    def _walk(self, node, uri, pointer):
        if isinstance(node, Mapping):
            if id(node) in self._nodes:
                return
            self._nodes[id(node)] = SchemaLocation(uri, pointer, node)
            for key, value in node.items():
                self._walk(value, uri, (*pointer, key))
        elif isinstance(node, list):
            if id(node) in self._nodes:
                return
            self._nodes[id(node)] = SchemaLocation(uri, pointer, node)
            for index, value in enumerate(node):
                self._walk(value, uri, (*pointer, index))


_NOT_FOUND = object()


def _resource_for(schema):
    """
    Create a `referencing.Resource` for a root schema.

    The dialect is detected the same way `jsonschema` detects it for
    validation, so references resolve identically.
    """
    # Deferred to keep importing this module cheap and cycle-free.
    from jsonschema.validators import validator_for

    dialect = validator_for(schema).META_SCHEMA
    id_of = getattr(dialect, "get", lambda key, default=None: default)
    specification = referencing.jsonschema.specification_with(
        dialect_id=id_of("$id") or "urn:unknown-dialect",
        default=referencing.Specification.OPAQUE,
    )
    return specification.create_resource(schema)


def _walk_from(node, segments):
    """
    Walk *segments* down from *node*, returning the node found there.

    Returns the `_NOT_FOUND` sentinel if any segment does not exist.
    """
    for segment in segments:
        if (isinstance(node, Mapping) and segment in node) or (
            isinstance(node, list)
            and isinstance(segment, int)
            and 0 <= segment < len(node)
        ):
            node = node[segment]
        else:
            return _NOT_FOUND
    return node
