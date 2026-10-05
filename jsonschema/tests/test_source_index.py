"""
Tests for `jsonschema._source_index`, the read-only source index which
correlates validation errors (and ``$ref`` URIs) with the schema nodes
they originate from.

Run with:

    python -m pytest jsonschema/tests/test_source_index.py -v
"""
from copy import deepcopy

from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012
import pytest

from jsonschema import Draft202012Validator
from jsonschema._source_index import (
    ReferenceFailure,
    SchemaNodeNotFound,
    SnapshotStore,
    SourceIndex,
    StaleSnapshotError,
)


def first_error(schema, instance, **kwargs):
    validator = Draft202012Validator(schema, **kwargs)
    return next(validator.iter_errors(instance))


class TestLocateValidationErrors:
    def test_root_ref_error_locates_into_defs(self):
        """
        A root ``$ref`` into ``$defs``: the error locates to the node in
        ``$defs``, not to the (ref-less) root path.
        """
        schema = {
            "$defs": {"positive": {"type": "integer", "minimum": 1}},
            "$ref": "#/$defs/positive",
        }
        index = SourceIndex.bind(Registry(), root_schema=schema)

        error = first_error(schema, "nope")
        assert list(error.absolute_schema_path) == ["type"]

        location = index.locate(error)
        assert location.uri == ""
        assert location.pointer == ("$defs", "positive", "type")
        assert location.json_pointer == "/$defs/positive/type"
        assert location.node == "integer"
        assert location.canonical_uri == "#/$defs/positive/type"

    def test_local_error_locates_from_root(self):
        schema = {"properties": {"a": {"type": "string"}}}
        index = SourceIndex.bind(Registry(), root_schema=schema)

        location = index.locate(first_error(schema, {"a": 1}))
        assert location.pointer == ("properties", "a", "type")
        assert location.node == "string"

    def test_error_and_schema_are_not_modified(self):
        schema = {
            "$defs": {"x": {"type": "string"}},
            "properties": {"a": {"$ref": "#/$defs/x"}},
        }
        pristine = deepcopy(schema)
        error = first_error(schema, {"a": 1})
        contents = error._contents()

        index = SourceIndex.bind(Registry(), root_schema=schema)
        index.locate(error)
        index.resolve_reference("#/$defs/x")

        assert schema == pristine
        assert error._contents() == contents

    def test_error_from_other_documents_is_not_found(self):
        index = SourceIndex.bind(Registry(), root_schema={"type": "string"})
        error = first_error({"type": "integer"}, "nope")
        with pytest.raises(SchemaNodeNotFound):
            index.locate(error)


class TestReferenceFailures:
    def test_missing_defs_is_a_reference_failure_not_a_type_error(self):
        """
        A ``$ref`` to a missing ``$defs`` entry fails validation with an
        unresolvable-reference error (not a ``ValidationError``), and the
        index reports it as a `ReferenceFailure`.
        """
        schema = {"$ref": "#/$defs/missing"}
        index = SourceIndex.bind(Registry(), root_schema=schema)

        with pytest.raises(Unresolvable) as exc_info:
            Draft202012Validator(schema).validate(1)

        with pytest.raises(ReferenceFailure):
            index.locate(exc_info.value)

        with pytest.raises(ReferenceFailure) as failure:
            index.resolve_reference("#/$defs/missing")
        assert failure.value.ref == "#/$defs/missing"

    def test_reference_failures_are_distinct_from_validation_errors(self):
        index = SourceIndex.bind(Registry(), root_schema={"$ref": "#/nope"})
        with pytest.raises(ReferenceFailure):
            index.resolve_reference("#/nope")


class TestTwoDocuments:
    """
    A two-document example: ``root.json`` references a ``$defs`` entry in
    a second document, and errors locate across the document boundary.
    """

    ROOT_URI = "https://example.com/root.json"
    OTHER_URI = "https://example.com/other.json"

    def make(self):
        root = {"$id": self.ROOT_URI, "$ref": "other.json#/$defs/name"}
        other = {
            "$id": self.OTHER_URI,
            "$defs": {"name": {"type": "string"}},
        }
        registry = Registry().with_resource(
            self.OTHER_URI,
            DRAFT202012.create_resource(other),
        )
        return root, other, registry

    def test_error_locates_to_the_other_document(self):
        root, _, registry = self.make()
        index = SourceIndex.bind(registry, root_schema=root)

        error = first_error(root, 123, registry=registry)
        location = index.locate(error)

        # the cross-document identity of the $ref target is preserved
        assert location.uri == self.OTHER_URI
        assert location.pointer == ("$defs", "name", "type")
        assert location.node == "string"
        assert (
            location.canonical_uri
            == "https://example.com/other.json#/$defs/name/type"
        )

    def test_reference_uri_resolves_to_the_other_document(self):
        root, other, registry = self.make()
        index = SourceIndex.bind(registry, root_schema=root)

        location = index.resolve_reference("other.json#/$defs/name")
        assert location.uri == self.OTHER_URI
        assert location.pointer == ("$defs", "name")
        assert location.node is other["$defs"]["name"]

    def test_missing_defs_in_the_other_document_is_reference_failure(self):
        root, other, registry = self.make()
        del other["$defs"]
        index = SourceIndex.bind(registry, root_schema=root)

        with pytest.raises(ReferenceFailure):
            index.resolve_reference("other.json#/$defs/name")


class TestSharedLocations:
    def test_repeated_references_to_one_node_share_locations(self):
        schema = {
            "$defs": {"x": {"type": "string"}},
            "properties": {
                "a": {"$ref": "#/$defs/x"},
                "b": {"$ref": "#/$defs/x"},
            },
        }
        index = SourceIndex.bind(Registry(), root_schema=schema)

        validator = Draft202012Validator(schema)
        first, second = validator.iter_errors({"a": 1, "b": 2})
        assert first.schema is second.schema

        one, two = index.locate(first), index.locate(second)
        assert one == two
        assert one.node is two.node
        assert one.pointer == ("$defs", "x", "type")

    def test_repeated_resolution_shares_one_cached_location(self):
        schema = {"$defs": {"x": {"type": "string"}}}
        index = SourceIndex.bind(Registry(), root_schema=schema)

        one = index.resolve_reference("#/$defs/x")
        two = index.resolve_reference("#/$defs/x")
        assert one is two
        assert one.node is schema["$defs"]["x"]

    def test_locate_node_shares_snapshot_locations(self):
        schema = {"$defs": {"x": {"type": "string"}}}
        index = SourceIndex.bind(Registry(), root_schema=schema)

        one = index.locate_node(schema["$defs"]["x"])
        two = index.locate_node(schema["$defs"]["x"])
        assert one is two
        assert one.pointer == ("$defs", "x")


class TestSnapshotBinding:
    def test_swapping_the_snapshot_retires_old_indexes(self):
        store = SnapshotStore(
            Registry(),
            root_schema={"$defs": {"x": {"type": "string"}}},
        )
        old_index = store.index()

        schema = {
            "$defs": {"x": {"type": "integer"}},
            "$ref": "#/$defs/x",
        }
        store.swap(Registry(), root_schema=schema)

        error = first_error(schema, "nope")
        with pytest.raises(StaleSnapshotError):
            old_index.locate(error)
        with pytest.raises(StaleSnapshotError):
            old_index.resolve_reference("#/$defs/x")
        with pytest.raises(StaleSnapshotError):
            old_index.locate_node(schema["$defs"]["x"])

        fresh_index = store.index()
        location = fresh_index.locate(error)
        assert location.pointer == ("$defs", "x", "type")
        assert location.node == "integer"

    def test_indexes_bind_to_distinct_snapshots(self):
        store = SnapshotStore(Registry(), root_schema={"type": "string"})
        first, second = store.index(), store.index()
        assert first.snapshot is second.snapshot

        store.swap(Registry(), root_schema={"type": "integer"})
        third = store.index()
        assert third.snapshot is not first.snapshot
