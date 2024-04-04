"""Definition of exceptions and errors."""
from __future__ import annotations


class NotFittedError(Exception):
    """Error thrown by models that have not been fitted."""

    def __init__(self, message: str = "the model has not yet been fitted") -> None:
        """Initialize the NotFittedError.

        Args:
            message: Exception message.
        """
        self.message = message
        super().__init__(self.message)


class NoSequencesError(Exception):
    """Error thrown by functions that need at least one sequence to work as expected."""

    def __init__(self, message: str = "at least one sequence needed") -> None:
        """Initialize the NoSequencesError.

        Args:
            message: Exception message.
        """
        self.message = message
        super().__init__(self.message)
