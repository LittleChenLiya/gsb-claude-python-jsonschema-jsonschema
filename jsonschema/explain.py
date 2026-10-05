"""
A read-only interpreter for :kw:`anyOf` and :kw:`oneOf` validation errors.

When an instance fails an :kw:`anyOf` or :kw:`oneOf` keyword, the
resulting `jsonschema.exceptions.ValidationError` carries the errors
produced by each failing subschema ("branch") in its ``context``. This
module reorganizes that context -- purely by reading the already
produced errors, never by re-running the validator -- into a *candidate
branch report* which makes it easy for callers to see which branch came
closest to passing.

Branches are sorted for display by:

#. the number of distinct failing instance paths in the branch
   (ascending),
#. the number of errors in the branch (ascending), and
#. the original order of the branch in the schema (the sort is stable).

.. note::

    This ordering is a *presentation* heuristic only. It must not be
    treated as a pass/fail determination: an instance which failed
    :kw:`anyOf` or :kw:`oneOf` is invalid regardless of how the
    branches are ranked here.

:kw:`oneOf` can fail in two fundamentally different ways, which are
reported as distinct ``kind`` values:

``ALL_BRANCHES_FAILED``
    every branch produced errors (the error carries ``context``), or
``MULTIPLE_BRANCHES_MATCHED``
    the instance was valid under more than one branch (a *success
    conflict* -- the error carries no ``context``, so there are no
    field errors to rank).

All errors referenced by the report are the *original* error objects
taken from ``error.context``; nothing is copied, re-created or dropped,
so each reported error keeps both its instance location
(``absolute_path`` / ``json_path``) and its schema source
(``absolute_schema_path``), including paths which passed through
:kw:`$ref`.

Explanation format
------------------

``Explanation.render`` produces lines of the form::

    anyOf explanation: all branches failed
      instance: $
      schema: /anyOf
      branch #0: 1 failing path(s), 1 error(s)  <-- closest
        - $.foo: too short (schema: /anyOf/0/properties/foo/minLength)
      branch #1: 2 failing path(s), 3 error(s)
        - $.bar: ... (schema: /anyOf/1/...)

    Nested anyOf/oneOf errors are rendered indented below their line.

Small example
-------------

.. code:: python

    import jsonschema
    from jsonschema.explain import explain

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
    validator = jsonschema.Draft202012Validator(schema)
    error = next(validator.iter_errors({"a": 1}))

    explanation = explain(error)
    assert explanation.best.index == 0  # the branch missing only "b"
    print(explanation.render())
"""
from __future__ import annotations

from typing import Any, ClassVar

from attrs import field, frozen

from jsonschema.exceptions import ValidationError

__all__ = [
    "ALL_BRANCHES_FAILED",
    "MULTIPLE_BRANCHES_MATCHED",
    "BranchReport",
    "Explanation",
    "explain",
]

#: Explanation kind: every branch of the keyword produced errors.
ALL_BRANCHES_FAILED = "all_branches_failed"

#: Explanation kind: a :kw:`oneOf` failed because the instance was valid
#: under more than one branch (a success conflict, not a field error).
MULTIPLE_BRANCHES_MATCHED = "multiple_branches_matched"

_BRANCH_KEYWORDS = frozenset(["anyOf", "oneOf"])


def _json_pointer(parts) -> str:
    """
    Format a (schema) path as a JSON Pointer, using ``/`` for the root.
    """
    if not parts:
        return "/"
    return "/" + "/".join(
        str(part).replace("~", "~0").replace("/", "~1") for part in parts
    )


@frozen
class BranchReport:
    """
    A read-only report for a single candidate branch of an :kw:`anyOf`
    or :kw:`oneOf` error.

    ``errors`` holds the *original* error objects from the parent
    error's ``context`` -- references are preserved, not copied.
    """

    index: int | None = field()
    """
    The index of the branch within the original :kw:`anyOf` / :kw:`oneOf`
    array, or ``None`` if it could not be determined.
    """

    errors: tuple[ValidationError, ...] = field()
    """
    The original errors this branch produced, in their original order.
    """

    @property
    def error_count(self) -> int:
        """
        The number of errors the branch produced.
        """
        return len(self.errors)

    @property
    def failing_paths(self) -> tuple[tuple[str | int, ...], ...]:
        """
        The distinct absolute instance paths which failed in this
        branch, in first-seen order.
        """
        seen: list[tuple[str | int, ...]] = []
        for error in self.errors:
            path = tuple(error.absolute_path)
            if path not in seen:
                seen.append(path)
        return tuple(seen)

    @property
    def nested(self) -> tuple[Explanation, ...]:
        """
        Explanations for any errors in this branch which are themselves
        :kw:`anyOf` / :kw:`oneOf` errors with their own context.
        """
        return tuple(
            explanation
            for error in self.errors
            if (explanation := explain(error)) is not None
        )


@frozen
class Explanation:
    """
    A read-only interpretation of one :kw:`anyOf` or :kw:`oneOf` error.
    """

    ALL_BRANCHES_FAILED: ClassVar[str] = ALL_BRANCHES_FAILED
    MULTIPLE_BRANCHES_MATCHED: ClassVar[str] = MULTIPLE_BRANCHES_MATCHED

    error: ValidationError = field()
    """
    The original error this explanation was derived from.
    """

    keyword: str = field()
    """
    The keyword which failed: ``"anyOf"`` or ``"oneOf"``.
    """

    kind: str = field()
    """
    Either ``ALL_BRANCHES_FAILED`` or ``MULTIPLE_BRANCHES_MATCHED``.
    """

    branches: tuple[BranchReport, ...] = field()
    """
    The candidate branches, sorted for display (see the module
    docstring). Empty for ``MULTIPLE_BRANCHES_MATCHED``.
    """

    @property
    def best(self) -> BranchReport | None:
        """
        The branch which appears closest to passing.

        This is a display hint only -- it is *not* a validity
        determination. ``None`` when there are no failing branches to
        rank (e.g. a :kw:`oneOf` success conflict).
        """
        if self.kind != ALL_BRANCHES_FAILED or not self.branches:
            return None
        return self.branches[0]

    def render(self) -> str:
        """
        Render the explanation as human-readable text.

        See the module docstring for the format.
        """
        lines = [
            f"{self.keyword} explanation: "
            f"{self.kind.replace('_', ' ')}",
            f"  instance: {self.error.json_path}",
            f"  schema: {_json_pointer(self.error.absolute_schema_path)}",
        ]
        if self.kind == MULTIPLE_BRANCHES_MATCHED:
            lines.append(f"  {self.error.message}")
            lines.append(
                "  the instance is valid under more than one branch; "
                "there are no field errors to rank",
            )
            return "\n".join(lines)

        best = self.best
        for branch in self.branches:
            marker = "  <-- closest" if branch is best else ""
            lines.append(
                f"  branch #{branch.index}: "
                f"{len(branch.failing_paths)} failing path(s), "
                f"{branch.error_count} error(s){marker}",
            )
            for error in branch.errors:
                lines.append(
                    f"    - {error.json_path}: {error.message} "
                    f"(schema: {_json_pointer(error.absolute_schema_path)})",
                )
                nested = explain(error)
                if nested is not None:
                    lines.extend(
                        f"    {line}"
                        for line in nested.render().splitlines()
                    )
        return "\n".join(lines)


def explain(error: ValidationError) -> Explanation | None:
    """
    Interpret an :kw:`anyOf` or :kw:`oneOf` validation error as a
    candidate branch report.

    This function is read-only: it inspects ``error`` and its
    ``context`` and never re-runs any validation.

    Arguments:

        error (jsonschema.exceptions.ValidationError):

            the error to interpret

    Returns:

        an `Explanation`, or ``None`` if the error was not produced by
        :kw:`anyOf` or :kw:`oneOf`
    """
    if error.validator not in _BRANCH_KEYWORDS:
        return None

    if not error.context:
        # A oneOf error without context means the instance matched more
        # than one branch (a success conflict). anyOf errors always
        # carry context when they fail, but handle an empty one sanely.
        kind = (
            MULTIPLE_BRANCHES_MATCHED
            if error.validator == "oneOf"
            else ALL_BRANCHES_FAILED
        )
        return Explanation(
            error=error, keyword=error.validator, kind=kind, branches=(),
        )

    # Group the context errors by the branch which produced them. The
    # first element of each child's relative schema path is the index
    # of the branch within the anyOf / oneOf array.
    groups: dict[Any, list[ValidationError]] = {}
    for child in error.context:
        head = (
            child.relative_schema_path[0]
            if child.relative_schema_path
            else None
        )
        index = head if isinstance(head, int) else None
        groups.setdefault(index, []).append(child)

    # Build branches in original schema order (unknown indexes last)...
    def _schema_order(index):
        return (index is None, index if isinstance(index, int) else -1)

    branches = [
        BranchReport(index=index, errors=tuple(groups[index]))
        for index in sorted(groups, key=_schema_order)
    ]
    # ... then re-sort for display. The sort is stable, so branches
    # which tie keep their original schema order.
    branches.sort(key=lambda branch: (
        len(branch.failing_paths), branch.error_count,
    ))

    return Explanation(
        error=error,
        keyword=error.validator,
        kind=ALL_BRANCHES_FAILED,
        branches=tuple(branches),
    )
