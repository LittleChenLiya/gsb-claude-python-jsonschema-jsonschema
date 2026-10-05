from __future__ import annotations

from unittest import TestCase, mock
import types

from jsonschema import Draft202012Validator, ErrorStream

THREE_ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number"},
        "b": {"type": "number"},
        "c": {"type": "number"},
    },
}
THREE_ERROR_INSTANCE = {"a": "x", "b": "y", "c": "z"}


def three_errors():
    return Draft202012Validator(THREE_ERROR_SCHEMA).iter_errors(
        THREE_ERROR_INSTANCE,
    )


def three_error_paths():
    return [next(iter(error.path)) for error in three_errors()]


class TestErrorStream(TestCase):
    def test_full_iteration_marks_complete_and_preserves_order(self):
        stream = ErrorStream(three_errors())
        errors = list(stream)

        self.assertEqual(
            [next(iter(error.path)) for error in errors],
            three_error_paths(),
        )
        self.assertTrue(stream.complete)
        self.assertEqual(stream.emitted, 3)

    def test_callback_count_does_not_increase_on_full_iteration(self):
        calls = []
        stream = ErrorStream(three_errors(), on_error=calls.append)

        errors = list(stream)

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls, errors)
        # closing after exhaustion must not invoke the callback again
        stream.close()
        self.assertEqual(len(calls), 3)
        self.assertTrue(stream.complete)

    def test_take_one_then_close_is_incomplete(self):
        stream = ErrorStream(three_errors())
        first = next(stream)

        self.assertEqual(next(iter(first.path)), "a")

        stream.close()
        self.assertFalse(stream.complete)
        self.assertEqual(stream.emitted, 1)
        # a closed stream yields nothing further
        self.assertEqual(list(stream), [])

    def test_max_errors_stops_early_and_is_incomplete(self):
        stream = ErrorStream(three_errors(), max_errors=1)

        errors = list(stream)

        self.assertEqual(len(errors), 1)
        self.assertFalse(stream.complete)
        self.assertEqual(stream.emitted, 1)

    def test_zero_budget_runs_no_validation(self):
        started = []

        def errors():
            started.append(True)
            yield from three_errors()

        stream = ErrorStream(errors(), budget=lambda: False)

        self.assertEqual(list(stream), [])
        self.assertFalse(stream.complete)
        self.assertEqual(stream.emitted, 0)
        self.assertEqual(started, [])

    def test_budget_allows_then_stops(self):
        state = {"remaining": 2}

        def budget():
            if state["remaining"] <= 0:
                return False
            state["remaining"] -= 1
            return True

        stream = ErrorStream(three_errors(), budget=budget)

        errors = list(stream)

        self.assertEqual(len(errors), 2)
        self.assertFalse(stream.complete)
        self.assertEqual(stream.emitted, 2)

    def test_close_releases_underlying_generator(self):
        closed = []

        def errors():
            try:
                yield from three_errors()
            finally:
                closed.append(True)

        stream = ErrorStream(errors())
        next(stream)
        stream.close()

        self.assertEqual(closed, [True])
        self.assertFalse(stream.complete)

    def test_context_is_not_flattened(self):
        schema = {"anyOf": [{"type": "number"}, {"type": "boolean"}]}
        stream = ErrorStream(
            Draft202012Validator(schema).iter_errors("string"),
        )

        errors = list(stream)

        # one top-level error, its nested errors stay in ``context``
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].context), 2)
        self.assertTrue(stream.complete)

    def test_is_valid_is_not_called(self):
        with mock.patch.object(
            Draft202012Validator,
            "is_valid",
            wraps=Draft202012Validator.is_valid,
        ) as is_valid:
            stream = ErrorStream(
                Draft202012Validator(THREE_ERROR_SCHEMA).iter_errors(
                    THREE_ERROR_INSTANCE,
                ),
            )
            list(stream)

        is_valid.assert_not_called()
        self.assertTrue(stream.complete)

    def test_default_iter_errors_entry_point_unchanged(self):
        result = three_errors()
        self.assertIsInstance(result, types.GeneratorType)
        self.assertNotIsInstance(result, ErrorStream)

    def test_close_is_idempotent(self):
        stream = ErrorStream(three_errors())
        next(stream)
        stream.close()
        stream.close()  # must not raise
        self.assertFalse(stream.complete)

    def test_context_manager_closes_stream(self):
        closed = []

        def errors():
            try:
                yield from three_errors()
            finally:
                closed.append(True)

        with ErrorStream(errors()) as stream:
            next(stream)

        self.assertEqual(closed, [True])
        self.assertFalse(stream.complete)
