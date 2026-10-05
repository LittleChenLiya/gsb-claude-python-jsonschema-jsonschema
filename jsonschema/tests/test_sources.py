"""
Tests for the read-only `SourceIndex` for `ValidationError`\\ s.
"""
from __future__ import annotations

import copy

import pytest
import referencing
import referencing.exceptions
import referencing.jsonschema

from jsonschema import (
    Draft202012Validator,
    ReferenceFailure,
    SnapshotMismatch,
    SourceIndex,
    SourceLocation,
    ValidationError,
)


def _registry(*pairs):
    return referencing.Registry().with_contents(
        pairs,
        default_specification=referencing.jsonschema.DRAFT202012,
    )


def _errors(schema, instance, registry=None):
    validator = Draft202012Validator(schema, registry=registry)
    return list(validator.iter_errors(instance))


class TestSourceIndex:
    def test_locate_ref_in_defs(self):
        schema = {
            "$defs": {"foo": {"type": "integer"}},
            "$ref": "#/$defs/foo",
        }
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        errors = _errors(schema, "hello", registry=registry)
        assert len(errors) == 1

        location = index.locate(errors[0])
        assert isinstance(location, SourceLocation)
        assert location.node is schema["$defs"]["foo"]
        assert location.uri == "http://example.com"
        assert location.pointer == "/$defs/foo"
        assert location.keyword == "type"

    def test_missing_defs_returns_reference_failure(self):
        schema = {"$ref": "#/$defs/missing"}
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        with pytest.raises(referencing.exceptions.Unresolvable) as exc_info:
            _errors(schema, "hello", registry=registry)

        failure = index.locate_failure(exc_info.value)
        assert isinstance(failure, ReferenceFailure)
        # The referencing library reports the pointer fragment for a
        # PointerToNowhere, so the ref is the fragment without the leading '#'.
        assert failure.ref == "/$defs/missing"

    def test_snapshot_mismatch_rejects(self):
        schema = {
            "$defs": {"foo": {"type": "integer"}},
            "$ref": "#/$defs/foo",
        }
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        errors = _errors(schema, "hello", registry=registry)
        assert len(errors) == 1

        swapped = registry.with_contents(
            [("http://example.com/other", {"type": "string"})],
            default_specification=referencing.jsonschema.DRAFT202012,
        )
        with pytest.raises(SnapshotMismatch):
            index.locate(errors[0], registry=swapped)

    def test_two_documents_preserves_cross_document_identity(self):
        doc_a = {
            "$id": "http://example.com/a",
            "properties": {
                "f": {"$ref": "http://example.com/b#/$defs/bar"},
            },
        }
        doc_b = {
            "$id": "http://example.com/b",
            "$defs": {"bar": {"type": "integer"}},
        }
        registry = _registry(
            ("http://example.com/a", doc_a),
            ("http://example.com/b", doc_b),
        )
        index = SourceIndex(registry)

        errors = _errors(doc_a, {"f": "hello"}, registry=registry)
        assert len(errors) == 1

        location = index.locate(errors[0])
        assert isinstance(location, SourceLocation)
        assert location.node is doc_b["$defs"]["bar"]
        assert location.uri == "http://example.com/b"
        assert location.pointer == "/$defs/bar"
        assert location.keyword == "type"

    def test_repeated_refs_share_location(self):
        schema = {
            "$defs": {"foo": {"type": "integer"}},
            "properties": {
                "a": {"$ref": "#/$defs/foo"},
                "b": {"$ref": "#/$defs/foo"},
            },
        }
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        errors = _errors(
            schema,
            {"a": "hello", "b": "world"},
            registry=registry,
        )
        assert len(errors) == 2

        first = index.locate(errors[0])
        second = index.locate(errors[1])
        assert first is second
        assert first.node is schema["$defs"]["foo"]
        assert first.uri == "http://example.com"
        assert first.pointer == "/$defs/foo"

    def test_locate_ref_resolves_to_node(self):
        schema = {"$defs": {"foo": {"type": "integer"}}}
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        location = index.locate_ref(
            "#/$defs/foo",
            base_uri="http://example.com",
        )
        assert isinstance(location, SourceLocation)
        assert location.node is schema["$defs"]["foo"]
        assert location.uri == "http://example.com"
        assert location.pointer == "/$defs/foo"
        assert location.ref_chain == ("#/$defs/foo",)

    def test_locate_ref_failure(self):
        schema = {}
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        failure = index.locate_ref(
            "#/$defs/missing",
            base_uri="http://example.com",
        )
        assert isinstance(failure, ReferenceFailure)
        assert failure.ref == "#/$defs/missing"

    def test_original_error_and_schema_unchanged(self):
        schema = {
            "$defs": {"foo": {"type": "integer"}},
            "$ref": "#/$defs/foo",
        }
        original = copy.deepcopy(schema)
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        errors = _errors(schema, "hello", registry=registry)
        assert len(errors) == 1
        index.locate(errors[0])

        assert schema == original
        assert errors[0].schema is schema["$defs"]["foo"]

    def test_locate_returns_none_for_unknown_node(self):
        schema = {"type": "integer"}
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        error = ValidationError(
            "boom",
            validator="type",
            validator_value="integer",
            instance="hello",
            schema={"type": "integer"},
        )
        assert index.locate(error) is None

    def test_locate_without_registry_uses_snapshot(self):
        schema = {
            "$defs": {"foo": {"type": "integer"}},
            "$ref": "#/$defs/foo",
        }
        registry = _registry(("http://example.com", schema))
        index = SourceIndex(registry)

        errors = _errors(schema, "hello", registry=registry)
        location = index.locate(errors[0])
        assert isinstance(location, SourceLocation)
        assert location.node is schema["$defs"]["foo"]
