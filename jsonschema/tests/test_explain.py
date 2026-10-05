"""
Tests for `jsonschema.explain`, the read-only anyOf/oneOf interpreter.
"""

from jsonschema import Draft202012Validator, ValidationError, validators
from jsonschema.explain import (
    ALL_BRANCHES_FAILED,
    MULTIPLE_BRANCHES_MATCHED,
    explain,
)


def only_error(schema, instance, cls=Draft202012Validator):
    errors = list(cls(schema).iter_errors(instance))
    assert len(errors) == 1
    return errors[0]


class TestAnyOf:
    def test_branch_missing_fewer_fields_is_shown_first(self):
        """
        A branch missing 1 field ranks ahead of one missing 3 fields.
        """
        schema = {
            "anyOf": [
                {"allOf": [{"required": ["b"]}]},
                {"allOf": [
                    {"required": ["b"]},
                    {"required": ["c"]},
                    {"required": ["d"]},
                ]},
            ],
        }
        error = only_error(schema, {"a": 1})

        explanation = explain(error)

        assert explanation.kind == ALL_BRANCHES_FAILED
        assert [branch.index for branch in explanation.branches] == [0, 1]
        best = explanation.best
        assert best.index == 0
        assert best.error_count == 1
        assert explanation.branches[1].error_count == 3
        rendered = explanation.render()
        assert rendered.index("branch #0") < rendered.index("branch #1")
        assert "<-- closest" in rendered.splitlines()[3]

    def test_fewer_distinct_failing_paths_ranks_first(self):
        schema = {
            "anyOf": [
                {"properties": {
                    "a": {"const": 1},
                    "b": {"const": 2},
                }},
                {"properties": {"a": {"const": 1}}},
            ],
        }
        error = only_error(schema, {"a": 0, "b": 0})

        explanation = explain(error)

        # Branch 1 fails on 1 path, branch 0 on 2 paths.
        assert [branch.index for branch in explanation.branches] == [1, 0]
        assert explanation.best.index == 1
        assert [len(b.failing_paths) for b in explanation.branches] == [1, 2]

    def test_ties_keep_original_schema_order(self):
        """
        The display sort is stable: equally-ranked branches keep their
        original order within the anyOf array.
        """
        schema = {"anyOf": [{"required": ["a"]}, {"required": ["b"]}]}
        error = only_error(schema, {})

        explanation = explain(error)

        assert [branch.index for branch in explanation.branches] == [0, 1]

    def test_original_error_references_are_preserved(self):
        """
        Every context error appears in the report, as the same object.
        """
        schema = {
            "anyOf": [
                {"allOf": [{"required": ["b"]}, {"required": ["c"]}]},
                {"type": "string"},
            ],
        }
        error = only_error(schema, {"a": 1})

        explanation = explain(error)

        reported = [
            each for branch in explanation.branches for each in branch.errors
        ]
        assert len(reported) == len(error.context)
        assert all(
            any(reported_error is original for original in error.context)
            for reported_error in reported
        )
        # ... and the parent error itself is the original object too.
        assert explanation.error is error

    def test_nested_context_is_explained_recursively(self):
        schema = {
            "anyOf": [
                {"anyOf": [{"required": ["x"]}, {"required": ["y"]}]},
                {"required": ["z"]},
            ],
        }
        error = only_error(schema, {})

        explanation = explain(error)

        by_index = {b.index: b for b in explanation.branches}
        nested = by_index[0].nested
        assert len(nested) == 1
        assert nested[0].kind == ALL_BRANCHES_FAILED
        assert len(nested[0].branches) == 2
        # The nested explanation is also rendered, indented.
        rendered = explanation.render()
        assert "    anyOf explanation: all branches failed" in rendered

    def test_ref_keeps_instance_location_and_schema_source(self):
        """
        Errors which passed through $ref report both the instance
        location and the schema path at the referring site.
        """
        schema = {
            "$defs": {"name": {"type": "string", "minLength": 3}},
            "anyOf": [
                {"$ref": "#/$defs/name"},
                {"type": "integer"},
            ],
        }
        error = only_error(schema, "ab")

        explanation = explain(error)

        by_index = {b.index: b for b in explanation.branches}
        ref_error = by_index[0].errors[0]
        assert list(ref_error.absolute_path) == []
        assert ref_error.json_path == "$"
        assert list(ref_error.absolute_schema_path) == [
            "anyOf", 0, "minLength",
        ]
        rendered = explanation.render()
        assert "instance: $" in rendered
        assert "schema: /anyOf/0/minLength" in rendered


class TestOneOf:
    def test_all_branches_failed(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "boolean"}]}
        error = only_error(schema, 12)

        explanation = explain(error)

        assert explanation.kind == ALL_BRANCHES_FAILED
        assert len(explanation.branches) == 2
        assert explanation.best is not None

    def test_multiple_matches_is_a_success_conflict(self):
        """
        A oneOf failed by matching two branches is reported as a
        success conflict, not as field errors.
        """
        schema = {"oneOf": [{"type": "integer"}, {"minimum": 2}]}
        error = only_error(schema, 3)

        explanation = explain(error)

        assert explanation.kind == MULTIPLE_BRANCHES_MATCHED
        assert explanation.branches == ()
        assert explanation.best is None
        rendered = explanation.render()
        assert "multiple branches matched" in rendered
        assert "no field errors" in rendered
        assert "branch #" not in rendered


class TestExplainIsReadOnly:
    def test_explaining_does_not_revalidate(self):
        """
        Building and rendering an explanation runs no validation: the
        schema's keywords are evaluated exactly once, by iter_errors.
        """
        calls = []

        def counted(validator, value, instance, schema):
            calls.append(value)
            yield ValidationError("counted failure")

        CountingValidator = validators.extend(
            Draft202012Validator, validators={"counted": counted},
        )
        schema = {"anyOf": [{"counted": 1}, {"counted": 2}]}
        error = only_error(schema, {}, cls=CountingValidator)
        assert calls == [1, 2]  # one validation pass over both branches

        explanation = explain(error)
        explanation.render()
        _ = explanation.best
        _ = [branch.nested for branch in explanation.branches]

        assert calls == [1, 2]  # explaining re-ran nothing

    def test_non_branch_keyword_is_not_explainable(self):
        error = only_error({"type": "string"}, 12)
        assert explain(error) is None
