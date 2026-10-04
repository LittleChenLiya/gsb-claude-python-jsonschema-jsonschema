"""
Tests for the read-only anyOf/oneOf branch interpreter.
"""
from collections.abc import Sequence
from unittest import TestCase, mock

from jsonschema import Draft202012Validator
from jsonschema.exceptions import (
    BranchInterpreter,
    CandidateBranch,
    ValidationError,
    branch_report,
    format_branch_report,
)


def _errors(instance, schema):
    return list(Draft202012Validator(schema).iter_errors(instance))


def _single_error(instance, schema):
    errors = _errors(instance, schema)
    assert len(errors) == 1, f"expected 1 error, got {len(errors)}"
    return errors[0]


class TestBranchInterpreter(TestCase):
    def test_anyof_branch_missing_fewer_fields_is_closest(self):
        # Branch 0 misses 1 field, branch 1 misses 3 fields.
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b", "c"]},
            ],
        }
        error = _single_error({}, schema)
        report = BranchInterpreter().interpret(error)

        self.assertEqual(report.keyword, "anyOf")
        self.assertTrue(report.all_failed)
        self.assertFalse(report.conflict)
        # The branch missing 1 field is closest to passing.
        self.assertEqual(report.closest.index, 0)
        self.assertEqual(report.closest.error_count, 1)
        # The branch missing 3 fields is last.
        self.assertEqual(report.branches[1].index, 1)
        self.assertEqual(report.branches[1].error_count, 3)

    def test_oneof_satisfying_both_branches_is_success_conflict(self):
        schema = {
            "oneOf": [
                {"type": "object", "required": ["x"]},
                {"type": "object", "required": ["y"]},
            ],
        }
        error = _single_error({"x": 1, "y": 2}, schema)
        report = branch_report(error)

        self.assertTrue(report.conflict)
        self.assertFalse(report.all_failed)
        self.assertEqual(report.branches, ())
        # The conflict message lists the branches that succeeded.
        self.assertIn("is valid under each of", report.original.message)

    def test_oneof_all_branches_fail_is_not_conflict(self):
        schema = {
            "oneOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b"]},
            ],
        }
        error = _single_error({}, schema)
        report = branch_report(error)

        self.assertFalse(report.conflict)
        self.assertTrue(report.all_failed)
        self.assertEqual(len(report.branches), 2)
        self.assertEqual(report.closest.index, 0)

    def test_original_error_references_are_kept(self):
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b"]},
            ],
        }
        error = _single_error({}, schema)
        report = branch_report(error)

        # Every original context error is kept by reference, not copied.
        kept = {id(e) for b in report.branches for e in b.errors}
        original = {id(e) for e in error.context}
        self.assertEqual(kept, original)
        for branch in report.branches:
            for err in branch.errors:
                self.assertIsInstance(err, ValidationError)

    def test_sorting_by_failed_paths_then_error_count_then_order(self):
        # Branch 0: 2 distinct paths, 2 errors.
        # Branch 1: 1 distinct path, 2 errors.
        # Branch 2: 1 distinct path, 1 error.
        # Closest should be branch 2 (1 path, 1 error), then branch 1
        # (1 path, 2 errors), then branch 0 (2 paths).
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
                {
                    "type": "object",
                    "properties": {
                        "c": {"type": "integer"},
                        "e": {"type": "integer"},
                    },
                    "required": ["c", "e"],
                },
                {"type": "object", "required": ["d"]},
            ],
        }
        # a is a string (type error at ['a']), b missing (required at []),
        # c and e missing (required at []), d missing (required at []).
        error = _single_error({"a": "not an int"}, schema)
        report = branch_report(error)

        order = [b.index for b in report.branches]
        self.assertEqual(order, [2, 1, 0])
        self.assertEqual(report.closest.index, 2)

    def test_sorting_is_stable_regardless_of_error_order(self):
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b"]},
            ],
        }
        error = _single_error({}, schema)
        report1 = branch_report(error)

        # Shuffle the context; the display order must stay the same.
        shuffled = ValidationError.create_from(error)
        shuffled.context = list(reversed(error.context))
        report2 = branch_report(shuffled)
        self.assertEqual(
            [b.index for b in report1.branches],
            [b.index for b in report2.branches],
        )

    def test_ref_keeps_instance_location_and_schema_source(self):
        schema = {
            "$defs": {
                "branch": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
            "anyOf": [
                {"$ref": "#/$defs/branch"},
                {
                    "type": "object",
                    "properties": {"c": {"type": "integer"}},
                    "required": ["c"],
                },
            ],
        }
        error = _single_error({"a": "not an int"}, schema)
        report = branch_report(error)

        for branch in report.branches:
            for err in branch.errors:
                # Both the instance location and the schema source are present.
                self.assertTrue(hasattr(err, "absolute_path"))
                self.assertTrue(hasattr(err, "absolute_schema_path"))
                self.assertIsInstance(err.absolute_path, Sequence)
                self.assertIsInstance(err.absolute_schema_path, Sequence)

    def test_nested_context_is_supplemented(self):
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "anyOf": [
                                {"type": "object", "required": ["x"]},
                                {"type": "object", "required": ["x", "y"]},
                            ],
                        },
                    },
                },
                {"type": "object", "required": ["z"]},
            ],
        }
        error = _single_error({"inner": {}}, schema)
        report = branch_report(error)

        nested_reports = [
            nr
            for branch in report.branches
            for nr in branch.nested_reports
        ]
        self.assertEqual(len(nested_reports), 1)
        nested = nested_reports[0]
        self.assertEqual(nested.keyword, "anyOf")
        self.assertEqual(nested.closest.index, 0)
        self.assertEqual(nested.closest.error_count, 1)

    def test_interpreter_does_not_rerun_validation(self):
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b", "c"]},
            ],
        }
        validator = Draft202012Validator(schema)
        original_iter = Draft202012Validator.iter_errors
        calls = []

        def counting_iter(self, instance, _schema=None):
            calls.append(1)
            return original_iter(self, instance, _schema)

        with mock.patch.object(
            Draft202012Validator, "iter_errors", counting_iter,
        ):
            errors = list(validator.iter_errors({}))
            count_after_validation = len(calls)
            self.assertGreaterEqual(count_after_validation, 1)

            report = branch_report(errors[0])
            self.assertEqual(len(calls), count_after_validation)

            _ = format_branch_report(report)
            self.assertEqual(len(calls), count_after_validation)

    def test_interpreter_does_not_rerun_validation_for_oneof_conflict(self):
        schema = {
            "oneOf": [
                {"type": "object", "required": ["x"]},
                {"type": "object", "required": ["y"]},
            ],
        }
        validator = Draft202012Validator(schema)
        original_iter = Draft202012Validator.iter_errors
        calls = []

        def counting_iter(self, instance, _schema=None):
            calls.append(1)
            return original_iter(self, instance, _schema)

        with mock.patch.object(
            Draft202012Validator, "iter_errors", counting_iter,
        ):
            errors = list(validator.iter_errors({"x": 1, "y": 2}))
            count_after_validation = len(calls)
            report = branch_report(errors[0])
            self.assertEqual(len(calls), count_after_validation)
            self.assertTrue(report.conflict)

    def test_interpreter_rejects_non_branch_error(self):
        schema = {"type": "object", "required": ["a"]}
        error = _single_error({}, schema)
        with self.assertRaises(ValueError):
            BranchInterpreter().interpret(error)

    def test_candidate_branch_passed_is_not_inferred_from_order(self):
        # A branch with zero errors is passed regardless of its position.
        branch = CandidateBranch(index=5, errors=(), keyword="anyOf")
        self.assertTrue(branch.passed)
        self.assertEqual(branch.error_count, 0)
        self.assertEqual(branch.failed_path_count, 0)

    def test_format_report_contains_instance_and_schema_paths(self):
        schema = {
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["a", "b"]},
            ],
        }
        error = _single_error({}, schema)
        report = branch_report(error)
        output = format_branch_report(report)

        self.assertIn("anyOf: 2 candidate branches", output)
        self.assertIn("closest to passing", output)
        self.assertIn("at instance path:", output)
        self.assertIn("at schema source:", output)
        # The ordering disclaimer must be present.
        self.assertIn("not a pass/fail determination", output)

    def test_format_conflict_report(self):
        schema = {
            "oneOf": [
                {"type": "object", "required": ["x"]},
                {"type": "object", "required": ["y"]},
            ],
        }
        error = _single_error({"x": 1, "y": 2}, schema)
        report = branch_report(error)
        output = format_branch_report(report)

        self.assertIn("success conflict", output)
        self.assertIn("is valid under each of", output)
        # Field errors must not be shown for a conflict.
        self.assertNotIn("at instance path:", output)
