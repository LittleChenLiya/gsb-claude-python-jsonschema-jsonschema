"""
Validation errors, and some surrounding helpers.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pprint import pformat
from textwrap import dedent, indent
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
import re
import warnings

from attrs import define
from referencing.exceptions import Unresolvable as _Unresolvable

from jsonschema import _utils

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        MutableMapping,
        Sequence,
    )

    from jsonschema import _types

WEAK_MATCHES: frozenset[str] = frozenset(["anyOf", "oneOf"])
STRONG_MATCHES: frozenset[str] = frozenset()

_JSON_PATH_COMPATIBLE_PROPERTY_PATTERN = re.compile("^[a-zA-Z][a-zA-Z0-9_]*$")

_unset = _utils.Unset()


def _pretty(thing: Any, prefix: str):
    """
    Format something for an error message as prettily as we currently can.
    """
    return indent(pformat(thing, width=72, sort_dicts=False), prefix).lstrip()


def __getattr__(name):
    if name == "RefResolutionError":
        warnings.warn(
            _RefResolutionError._DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        return _RefResolutionError
    raise AttributeError(f"module {__name__} has no attribute {name}")


class _Error(Exception):

    _word_for_schema_in_error_message: ClassVar[str]
    _word_for_instance_in_error_message: ClassVar[str]

    def __init__(
        self,
        message: str,
        validator: str = _unset,  # type: ignore[assignment]
        path: Iterable[str | int] = (),
        cause: Exception | None = None,
        context=(),
        validator_value: Any = _unset,
        instance: Any = _unset,
        schema: Mapping[str, Any] | bool = _unset,  # type: ignore[assignment]
        schema_path: Iterable[str | int] = (),
        parent: _Error | None = None,
        type_checker: _types.TypeChecker = _unset,  # type: ignore[assignment]
    ) -> None:
        super().__init__(
            message,
            validator,
            path,
            cause,
            context,
            validator_value,
            instance,
            schema,
            schema_path,
            parent,
        )
        self.message = message
        self.path = self.relative_path = deque(path)
        self.schema_path = self.relative_schema_path = deque(schema_path)
        self.context = list(context)
        self.cause = self.__cause__ = cause
        self.validator = validator
        self.validator_value = validator_value
        self.instance = instance
        self.schema = schema
        self.parent = parent
        self._type_checker = type_checker

        for error in context:
            error.parent = self

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.message!r}>"

    def __str__(self) -> str:
        essential_for_verbose = (
            self.validator, self.validator_value, self.instance, self.schema,
        )
        if any(m is _unset for m in essential_for_verbose):
            return self.message

        schema_path = _utils.format_as_index(
            container=self._word_for_schema_in_error_message,
            indices=list(self.relative_schema_path)[:-1],
        )
        instance_path = _utils.format_as_index(
            container=self._word_for_instance_in_error_message,
            indices=self.relative_path,
        )
        prefix = 16 * " "

        return dedent(
            f"""\
            {self.message}

            Failed validating {self.validator!r} in {schema_path}:
                {_pretty(self.schema, prefix=prefix)}

            On {instance_path}:
                {_pretty(self.instance, prefix=prefix)}
            """.rstrip(),
        )

    @classmethod
    def create_from(cls, other: _Error):
        return cls(**other._contents())

    @property
    def absolute_path(self) -> Sequence[str | int]:
        parent = self.parent
        if parent is None:
            return self.relative_path

        path = deque(self.relative_path)
        path.extendleft(reversed(parent.absolute_path))
        return path

    @property
    def absolute_schema_path(self) -> Sequence[str | int]:
        parent = self.parent
        if parent is None:
            return self.relative_schema_path

        path = deque(self.relative_schema_path)
        path.extendleft(reversed(parent.absolute_schema_path))
        return path

    @property
    def json_path(self) -> str:
        path = "$"
        for elem in self.absolute_path:
            if isinstance(elem, int):
                path += "[" + str(elem) + "]"
            elif _JSON_PATH_COMPATIBLE_PROPERTY_PATTERN.match(elem):
                path += "." + elem
            else:
                escaped_elem = elem.replace("\\", "\\\\").replace("'", r"\'")
                path += "['" + escaped_elem + "']"
        return path

    def _set(
        self,
        type_checker: _types.TypeChecker | None = None,
        **kwargs: Any,
    ) -> None:
        if type_checker is not None and self._type_checker is _unset:
            self._type_checker = type_checker

        for k, v in kwargs.items():
            if getattr(self, k) is _unset:
                setattr(self, k, v)

    def _contents(self):
        attrs = (
            "message", "cause", "context", "validator", "validator_value",
            "path", "schema_path", "instance", "schema", "parent",
        )
        return {attr: getattr(self, attr) for attr in attrs}

    def _matches_type(self) -> bool:
        try:
            # We ignore this as we want to simply crash if this happens
            expected = self.schema["type"]  # type: ignore[index]
        except (KeyError, TypeError):
            return False

        if isinstance(expected, str):
            return self._type_checker.is_type(self.instance, expected)

        return any(
            self._type_checker.is_type(self.instance, expected_type)
            for expected_type in expected
        )


_E = TypeVar("_E", bound=_Error)


class ValidationError(_Error):
    """
    An instance was invalid under a provided schema.
    """

    _word_for_schema_in_error_message = "schema"
    _word_for_instance_in_error_message = "instance"


class SchemaError(_Error):
    """
    A schema was invalid under its corresponding metaschema.
    """

    _word_for_schema_in_error_message = "metaschema"
    _word_for_instance_in_error_message = "schema"


@define(slots=False)
class _RefResolutionError(Exception):  # noqa: PLW1641
    """
    A ref could not be resolved.
    """

    _DEPRECATION_MESSAGE = (
        "jsonschema.exceptions.RefResolutionError is deprecated as of version "
        "4.18.0. If you wish to catch potential reference resolution errors, "
        "directly catch referencing.exceptions.Unresolvable."
    )

    _cause: Exception

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented  # pragma: no cover -- uncovered but deprecated  # noqa: E501
        return self._cause == other._cause

    def __str__(self) -> str:
        return str(self._cause)


class _WrappedReferencingError(_RefResolutionError, _Unresolvable):  # pragma: no cover -- partially uncovered but to be removed  # noqa: E501
    def __init__(self, cause: _Unresolvable):
        object.__setattr__(self, "_wrapped", cause)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return self._wrapped == other._wrapped
        elif other.__class__ is self._wrapped.__class__:
            return self._wrapped == other
        return NotImplemented

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)

    def __hash__(self):
        return hash(self._wrapped)

    def __repr__(self):
        return f"<WrappedReferencingError {self._wrapped!r}>"

    def __str__(self):
        return f"{self._wrapped.__class__.__name__}: {self._wrapped}"


class UndefinedTypeCheck(Exception):
    """
    A type checker was asked to check a type it did not have registered.
    """

    def __init__(self, type: str) -> None:
        self.type = type

    def __str__(self) -> str:
        return f"Type {self.type!r} is unknown to this type checker"


class UnknownType(Exception):
    """
    A validator was asked to validate an instance against an unknown type.
    """

    def __init__(self, type, instance, schema):
        self.type = type
        self.instance = instance
        self.schema = schema

    def __str__(self):
        prefix = 16 * " "

        return dedent(
            f"""\
            Unknown type {self.type!r} for validator with schema:
                {_pretty(self.schema, prefix=prefix)}

            While checking instance:
                {_pretty(self.instance, prefix=prefix)}
            """.rstrip(),
        )


class FormatError(Exception):
    """
    Validating a format failed.
    """

    def __init__(self, message, cause=None):
        super().__init__(message, cause)
        self.message = message
        self.cause = self.__cause__ = cause

    def __str__(self):
        return self.message


class ErrorTree:
    """
    ErrorTrees make it easier to check which validations failed.

    Arguments:

        errors:

            the errors to populate the tree with

        instance:

            the instance the tree corresponds to, if known, which
            enables indexing the tree at indices not present in it to
            raise the same error that indexing the instance itself would

    """

    def __init__(
        self,
        errors: Iterable[ValidationError] = (),
        *,
        instance: Any = _unset,
    ):
        self.errors: MutableMapping[str, ValidationError] = {}
        self._contents: MutableMapping[str | int, ErrorTree] = {}
        self._instance = instance

        for error in errors:
            container = self
            for element in error.path:
                container = container._contents.setdefault(
                    element, self.__class__(),
                )
            container.errors[error.validator] = error

            container._instance = error.instance

    def __contains__(self, index: str | int):
        """
        Check whether ``instance[index]`` has any errors.
        """
        return index in self._contents

    def __getitem__(self, index):
        """
        Retrieve the child tree one level down at the given ``index``.

        If the index is not in the instance that this tree corresponds
        to and is not known by this tree, whatever error would be raised
        by ``instance.__getitem__`` will be propagated (usually this is
        some subclass of `LookupError`.
        """
        if index in self:
            return self._contents[index]
        if self._instance is _unset:
            return self.__class__()
        return self.__class__(instance=self._instance[index])

    def __setitem__(self, index: str | int, value: ErrorTree):
        """
        Add an error to the tree at the given ``index``.

        .. deprecated:: v4.20.0

            Setting items on an `ErrorTree` is deprecated without replacement.
            To populate a tree, provide all of its sub-errors when you
            construct the tree.
        """
        warnings.warn(
            "ErrorTree.__setitem__ is deprecated without replacement.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._contents[index] = value

    def __iter__(self):
        """
        Iterate (non-recursively) over the indices in the instance with errors.
        """
        return iter(self._contents)

    def __len__(self):
        """
        Return the `total_errors`.
        """
        return self.total_errors

    def __repr__(self):
        total = len(self)
        errors = "error" if total == 1 else "errors"
        return f"<{self.__class__.__name__} ({total} total {errors})>"

    @property
    def total_errors(self):
        """
        The total number of errors in the entire tree, including children.
        """
        child_errors = sum(len(tree) for _, tree in self._contents.items())
        return len(self.errors) + child_errors


def by_relevance(weak=WEAK_MATCHES, strong=STRONG_MATCHES):
    """
    Create a key function that can be used to sort errors by relevance.

    Arguments:
        weak (set):
            a collection of validation keywords to consider to be
            "weak".  If there are two errors at the same level of the
            instance and one is in the set of weak validation keywords,
            the other error will take priority. By default, :kw:`anyOf`
            and :kw:`oneOf` are considered weak keywords and will be
            superseded by other same-level validation errors.

        strong (set):
            a collection of validation keywords to consider to be
            "strong"

    """

    def relevance(error):
        validator = error.validator
        return (                        # prefer errors which are ...
            -len(error.path),           # shorter path thereby more general
            validator not in weak,      # for a non-low-priority keyword
            validator in strong,        # for a high priority keyword
            not error._matches_type(),  # at least match the instance's type
        )                               # otherwise we'll treat them the same

    return relevance


relevance = by_relevance()
"""
A key function (e.g. to use with `sorted`) which sorts errors by relevance.

Example:

.. code:: python

    sorted(validator.iter_errors(12), key=jsonschema.exceptions.relevance)
"""


def best_match(errors, key=relevance):
    """
    Try to find an error that appears to be the best match among given errors.

    In general, errors that are higher up in the instance (i.e. for which
    `ValidationError.path` is shorter) are considered better matches,
    since they indicate "more" is wrong with the instance.

    If the resulting match is either :kw:`oneOf` or :kw:`anyOf`, the
    *opposite* assumption is made -- i.e. the deepest error is picked
    among the most relevant errors in each separate subschema (preferring
    subschemas which produced fewer errors when tied), since these
    keywords only need to match once, and any other errors may not be
    relevant.

    Arguments:
        errors (collections.abc.Iterable):

            the errors to select from. Do not provide a mixture of
            errors from different validation attempts (i.e. from
            different instances or schemas), since it won't produce
            sensical output.

        key (collections.abc.Callable):

            the key to use when sorting errors. See `relevance` and
            transitively `by_relevance` for more details (the default is
            to sort with the defaults of that function). Changing the
            default is only useful if you want to change the function
            that rates errors but still want the error context descent
            done by this function.

    Returns:
        the best matching error, or ``None`` if the iterable was empty

    .. note::

        This function is a heuristic. Its return value may change for a given
        set of inputs from version to version if better heuristics are added.

    """
    _, best = _most_relevant(errors, key=key)
    if best is None:
        return

    while best.context:
        # Group the errors by the subschema which produced them.
        by_subschema: dict[Any, list[_Error]] = defaultdict(list)
        for error in best.context:
            index = error.schema_path[0] if error.schema_path else None
            by_subschema[index].append(error)

        # Rank each subschema by how deep its most relevant error is,
        # and amongst those equally deep, by how few errors it produced
        # (i.e. how close it was to being valid). Lower ranks are better.
        best_rank, best_in_subschema, tied = None, None, False
        for errors_in_subschema in by_subschema.values():
            error_key, error = _most_relevant(errors_in_subschema, key=key)
            rank = error_key, len(errors_in_subschema)
            if best_rank is None or rank < best_rank:
                best_rank, best_in_subschema, tied = rank, error, False
            elif rank == best_rank:
                tied = True

        # If multiple subschemas rank equally we can't tell which was
        # intended, so we stop here rather than descend into one of them.
        if tied:
            break
        best = best_in_subschema
    return best


def _most_relevant(
    errors: Iterable[_E],
    key: Callable[[_E], Any],
) -> tuple[Any, _E | None]:
    """
    Find the most relevant error along with its key, computing each key once.

    Equally relevant errors are settled by picking the one which appears
    earlier in the instance, which makes the choice independent of the
    order in which the errors happened to be produced.

    Returns ``(None, None)`` if there were no errors.
    """
    best_key, best = None, None
    for error in errors:
        error_key = key(error)
        if (
            best is None
            or error_key > best_key
            or (error_key == best_key and error.path < best.path)
        ):
            best_key, best = error_key, error
    return best_key, best


# ---------------------------------------------------------------------------
# Candidate branch reports for anyOf/oneOf errors
# ---------------------------------------------------------------------------

_BRANCH_KEYWORDS: frozenset[str] = frozenset({"anyOf", "oneOf"})


@define(frozen=True)
class CandidateBranch:
    """
    A single candidate subschema branch of an :kw:`anyOf` or :kw:`oneOf`.

    A branch is identified by its position in the keyword's subschema array
    (`index`) and holds the *original* validation errors it produced. The
    error objects are kept by reference -- they are never copied or
    re-created, and the validator is never re-run.

    Attributes:

        index:

            the branch's position in the keyword's subschema array

        errors:

            the original validation errors produced by this branch

        keyword:

            ``"anyOf"`` or ``"oneOf"``

    """

    index: int | None
    errors: tuple[ValidationError, ...] = ()
    keyword: str = "anyOf"

    @property
    def passed(self) -> bool:
        """
        Whether this branch produced no errors.

        This is taken directly from the branch's errors, never from its
        position in the display ordering.
        """
        return not self.errors

    @property
    def error_count(self) -> int:
        """
        The number of errors produced by this branch.
        """
        return len(self.errors)

    @property
    def failed_paths(self) -> tuple[tuple[Any, ...], ...]:
        """
        The distinct instance paths that failed in this branch.

        Paths are returned in the order they are first encountered, so the
        result is stable regardless of error ordering. Each path is the
        error's *absolute* instance path, so the instance location is kept
        even after a :kw:`$ref`.
        """
        seen: list[tuple[Any, ...]] = []
        for error in self.errors:
            path = tuple(error.absolute_path)
            if path not in seen:
                seen.append(path)
        return tuple(seen)

    @property
    def failed_path_count(self) -> int:
        """
        The number of distinct instance paths that failed.
        """
        return len(self.failed_paths)

    @property
    def nested_reports(self) -> tuple[BranchReport, ...]:
        """
        Reports for nested anyOf/oneOf errors within this branch's errors.

        This supplements the branch report with nested context without
        re-running validation -- it simply interprets errors that are
        themselves branch errors.
        """
        reports = [
            branch_report(error)
            for error in self.errors
            if error.validator in _BRANCH_KEYWORDS
        ]
        return tuple(reports)


@define(frozen=True)
class BranchReport:
    """
    A read-only report organizing an anyOf/oneOf error into candidate branches.

    The report sorts branches for *display* only -- by the number of
    distinct instance paths that failed, then by error count, then by the
    branch's original position in the subschema array. This ordering is
    never used to decide whether a branch passed or failed; pass/fail is
    taken directly from whether the branch produced errors.

    Attributes:

        keyword:

            ``"anyOf"`` or ``"oneOf"``

        branches:

            the candidate branches, sorted for display

        original:

            the original validation error this report was built from

        conflict:

            for :kw:`oneOf`, whether more than one branch succeeded

    """

    keyword: str
    branches: tuple[CandidateBranch, ...]
    original: ValidationError
    conflict: bool = False

    @property
    def all_failed(self) -> bool:
        """
        Whether every branch failed (no branch passed).
        """
        return (
            not self.conflict
            and all(not branch.passed for branch in self.branches)
        )

    @property
    def closest(self) -> CandidateBranch | None:
        """
        The branch closest to passing (the first branch in display order).
        """
        return self.branches[0] if self.branches else None

    def __str__(self) -> str:
        return format_branch_report(self)


class BranchInterpreter:
    """
    A read-only interpreter that turns anyOf/oneOf errors into branch reports.

    The interpreter only inspects an already-produced error tree. It never
    re-runs validation, never mutates errors, and never copies error
    objects -- each `CandidateBranch` holds the original
    `ValidationError` references.

    Example:

        >>> from jsonschema import Draft202012Validator
        >>> from jsonschema.exceptions import BranchInterpreter
        >>> schema = {
        ...     "anyOf": [
        ...         {"type": "object", "required": ["a"]},
        ...         {"type": "object", "required": ["a", "b", "c"]},
        ...     ],
        ... }
        >>> errors = list(Draft202012Validator(schema).iter_errors({}))
        >>> report = BranchInterpreter().interpret(errors[0])
        >>> report.closest.index
        0
        >>> report.closest.error_count
        1

    """

    def interpret(self, error: ValidationError) -> BranchReport:
        """
        Build a `BranchReport` for an :kw:`anyOf`/:kw:`oneOf` error.

        Arguments:

            error:

                an :kw:`anyOf` or :kw:`oneOf` validation error

        Returns:

            a read-only branch report

        Raises:

            ValueError:

                if ``error`` is not an :kw:`anyOf`/:kw:`oneOf` error

        """
        keyword = error.validator
        if keyword not in _BRANCH_KEYWORDS:
            raise ValueError(
                f"Expected an anyOf/oneOf validation error, got {keyword!r}",
            )

        if keyword == "oneOf" and not error.context:
            # More than one branch succeeded -- a success conflict, not a
            # field failure. There are no field errors to group.
            return BranchReport(
                keyword=keyword,
                branches=(),
                original=error,
                conflict=True,
            )

        by_index: dict[Any, list[ValidationError]] = {}
        for child in error.context:
            index = child.schema_path[0] if child.schema_path else None
            by_index.setdefault(index, []).append(child)

        branches = [
            CandidateBranch(index=index, errors=tuple(errors), keyword=keyword)
            for index, errors in by_index.items()
        ]
        # Display sort only: distinct failed paths, then error count, then
        # the branch's original position. Python's sort is stable, so the
        # original order is preserved for fully tied branches.
        branches.sort(
            key=lambda branch: (
                branch.failed_path_count,
                branch.error_count,
                branch.index if branch.index is not None else float("inf"),
            ),
        )
        return BranchReport(
            keyword=keyword,
            branches=tuple(branches),
            original=error,
        )


def branch_report(error: ValidationError) -> BranchReport:
    """
    Build a `BranchReport` for an :kw:`anyOf`/:kw:`oneOf` error.

    This is a convenience wrapper around `BranchInterpreter.interpret`.

    Arguments:

        error:

            an :kw:`anyOf` or :kw:`oneOf` validation error

    """
    return BranchInterpreter().interpret(error)


def format_branch_report(report: BranchReport, *, indent: int = 0) -> str:
    """
    Format a `BranchReport` as a human-readable string.

    The format shows, for each candidate branch (in display order), the
    distinct instance paths that failed and the original errors with both
    their instance location and schema source. The ordering is display-only
    and is not a pass/fail determination.

    Example:

        >>> from jsonschema import Draft202012Validator
        >>> from jsonschema.exceptions import (
        ...     branch_report, format_branch_report,
        ... )
        >>> schema = {
        ...     "anyOf": [
        ...         {"type": "object", "required": ["a"]},
        ...         {"type": "object", "required": ["a", "b", "c"]},
        ...     ],
        ... }
        >>> errors = list(Draft202012Validator(schema).iter_errors({}))
        >>> report = branch_report(errors[0])
        >>> output = format_branch_report(report)
        >>> print(output)  # doctest: +NORMALIZE_WHITESPACE
        anyOf: 2 candidate branches
          (branches are ordered for display by number of distinct failed
          instance paths, then error count, then original branch order;
          this is not a pass/fail determination)
          Branch 0 -- closest to passing
            Failed instance paths (1 path):
              - []
            Errors (1 error):
              - 'a' is a required property
                  at instance path: []
                  at schema source: ['anyOf', 0, 'required']
          Branch 1
            Failed instance paths (1 path):
              - []
            Errors (3 errors):
              - 'a' is a required property
                  at instance path: []
                  at schema source: ['anyOf', 1, 'required']
              - 'b' is a required property
                  at instance path: []
                  at schema source: ['anyOf', 1, 'required']
              - 'c' is a required property
                  at instance path: []
                  at schema source: ['anyOf', 1, 'required']

    """
    pad = " " * indent
    lines: list[str] = []

    if report.conflict:
        lines.append(
            f"{pad}{report.keyword}: success conflict "
            f"(more than one branch passed)",
        )
        lines.append(f"{pad}  {report.original.message}")
        return "\n".join(lines)

    branch_count = len(report.branches)
    plural = "branch" if branch_count == 1 else "branches"
    lines.append(f"{pad}{report.keyword}: {branch_count} candidate {plural}")
    lines.append(
        f"{pad}  (branches are ordered for display by number of distinct "
        f"failed instance paths, then error count, then original branch "
        f"order; this is not a pass/fail determination)",
    )

    for branch in report.branches:
        is_closest = (
            report.closest is not None and branch.index == report.closest.index
        )
        marker = " -- closest to passing" if is_closest else ""
        lines.append(f"{pad}  Branch {branch.index}{marker}")

        path_count = branch.failed_path_count
        path_word = "path" if path_count == 1 else "paths"
        lines.append(
            f"{pad}    Failed instance paths ({path_count} {path_word}):",
        )
        lines.extend(
            f"{pad}      - {list(path)}" for path in branch.failed_paths
        )

        error_count = branch.error_count
        error_word = "error" if error_count == 1 else "errors"
        lines.append(f"{pad}    Errors ({error_count} {error_word}):")
        for error in branch.errors:
            lines.append(f"{pad}      - {error.message}")
            lines.append(
                f"{pad}          at instance path: "
                f"{list(error.absolute_path)}",
            )
            lines.append(
                f"{pad}          at schema source: "
                f"{list(error.absolute_schema_path)}",
            )

        for nested in branch.nested_reports:
            lines.append(f"{pad}    Nested {nested.keyword} report:")
            lines.append(format_branch_report(nested, indent=indent + 6))

    return "\n".join(lines)
