"""
Explicit, early-stoppable error streams for `Validator.iter_errors`.

`ErrorStream` wraps the generator returned by `iter_errors` and adds:

* an explicit completion marker (`ErrorStream.complete`) that is only set
  once the underlying generator has been exhausted normally. Stopping
  early -- because a caller-provided error count or budget was reached --
  leaves the stream incomplete, so callers must *not* treat the absence
  of further errors as proof that an instance is valid;
* deterministic release of the underlying generator (and any resources it
  holds, e.g. reference-resolver scopes) via `ErrorStream.close`;
* a single optional ``on_error`` callback that is invoked at most once per
  emitted error, exactly when the error is yielded, and never on close or
  completion. It can be used to count how many errors were actually
  produced.

The stream deliberately does *not* flatten `ValidationError.context` into
additional errors (nested errors stay attached to the error that produced
them), does not reorder errors (they come out in exactly the order the
validator produces them), and never calls `Validator.is_valid`. The
default `Validator.iter_errors` entry point is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from jsonschema.exceptions import ValidationError

__all__ = ["ErrorStream"]


class ErrorStream:
    """
    An explicit wrapper around a validator's `iter_errors` generator.

    Iterating an `ErrorStream` yields the same errors, in the same order,
    as iterating the wrapped generator directly. The difference is that the
    stream tracks whether validation actually finished:

    >>> from jsonschema import Draft202012Validator

    >>> schema = {
    ...     "type": "object",
    ...     "properties": {
    ...         "a": {"type": "number"},
    ...         "b": {"type": "number"},
    ...         "c": {"type": "number"},
    ...     },
    ... }
    >>> instance = {"a": "x", "b": "y", "c": "z"}
    >>> validator = Draft202012Validator(schema)
    >>> stream = ErrorStream(validator.iter_errors(instance))
    >>> first = next(stream)
    >>> stream.close()
    >>> stream.complete
    False

    A stream is only marked complete once its underlying generator is
    exhausted normally:

    >>> stream = ErrorStream(validator.iter_errors(instance))
    >>> [error.path[0] for error in stream]
    ['a', 'b', 'c']
    >>> stream.complete
    True

    Arguments:

        errors:

            the generator returned by `Validator.iter_errors` (or any
            iterable of `ValidationError`).

        max_errors:

            an optional error count. Once this many errors have been
            emitted the stream stops itself and is left incomplete.

        budget:

            an optional zero-argument callable polled before each error.
            It should return `True` to allow validation to continue and
            `False` to stop the stream. A budget that is exhausted from
            the start (i.e. returns `False` on the first poll) means no
            validation work is done at all -- the underlying generator is
            never started.

        on_error:

            an optional callback invoked with each emitted error. It is
            called at most once per error, exactly when the error is
            yielded, and never on close or completion, so it doubles as a
            reliable count of how many errors were produced.

    """

    def __init__(
        self,
        errors: Iterable[ValidationError],
        *,
        max_errors: int | None = None,
        budget: Callable[[], bool] | None = None,
        on_error: Callable[[ValidationError], Any] | None = None,
    ) -> None:
        self._errors: Iterator[ValidationError] = iter(errors)
        self._max_errors = max_errors
        self._budget = budget
        self._on_error = on_error
        self._emitted = 0
        self._closed = False
        self._complete = False

    def __iter__(self) -> ErrorStream:
        return self

    def __next__(self) -> ValidationError:
        if self._closed or self._complete:
            raise StopIteration
        if self._max_errors is not None and self._emitted >= self._max_errors:
            self.close()
            raise StopIteration
        if self._budget is not None and not self._budget():
            self.close()
            raise StopIteration

        try:
            error = next(self._errors)
        except StopIteration:
            self._complete = True
            self.close()
            raise
        except BaseException:
            # Make sure an unexpected failure still releases the wrapped
            # generator rather than leaving it (and its resources) alive.
            self.close()
            raise

        self._emitted += 1
        try:
            if self._on_error is not None:
                self._on_error(error)
        except BaseException:
            # A misbehaving callback must not leave the wrapped generator
            # (and its resources) alive.
            self.close()
            raise
        return error

    def close(self) -> None:
        """
        Release the underlying generator.

        Closing a stream that has not been exhausted normally leaves it
        incomplete (`complete` is `False`). Closing is idempotent.
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self._errors, "close", None)
        if close is not None:
            close()

    @property
    def complete(self) -> bool:
        """
        Whether the underlying generator was exhausted normally.

        `True` only once the underlying generator has been exhausted
        normally; an early stop (via `max_errors`, `budget`, or `close`)
        leaves this `False`.
        """
        return self._complete

    @property
    def emitted(self) -> int:
        """
        The number of errors that have actually been emitted.
        """
        return self._emitted

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> bool:
        self.close()
        return False
