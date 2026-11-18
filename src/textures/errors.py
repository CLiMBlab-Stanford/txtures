"""Exceptions that textures reports to users."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class TexturesError(Exception):
    """An error caused by user input, definitions, or settings rather than a bug."""


class SettingsError(TexturesError):
    """A settings document or setting value is invalid."""


@dataclass(frozen=True)
class Problem:
    """One validation failure, located by file path or definition name."""

    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


class DefinitionError(TexturesError):
    """One or more definitions are invalid.

    `problems` lists every failure found, so callers can report them together.
    """

    def __init__(self, problems: Iterable[Problem]):
        self.problems = tuple(dict.fromkeys(problems))
        super().__init__("\n".join(str(problem) for problem in self.problems))
