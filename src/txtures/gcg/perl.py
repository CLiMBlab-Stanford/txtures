"""Perl's substitution semantics on one string, for the rules translated from ModelBlocks.

The ModelBlocks tree scripts rewrite one line, Perl's `$_`, with `s///`
substitutions whose patterns and replacements are kept verbatim in
`txtures.gcg.rules` and `txtures.gcg.fixes`. `PerlString` applies them as
Perl would:

- `$NAME` and `${NAME}` in a pattern or a replacement are replaced by the
  script's variables, as Perl interpolates them;
- a replacement is a Perl double-quoted string: `\\1` and `$1` insert a group
  (empty when it did not match), `\\n` and `\\t` are a newline and a tab, and
  any other escaped character stands for itself;
- `g` replaces every match, and otherwise only the first is replaced;
- patterns are matched with `re.ASCII`, so `\\s` and `\\w` mean what they mean
  for Perl on bytes.

Python's `re` and Perl's engine agree on the constructs the rules use
(classes, lookaheads, backreferences, lazy and greedy quantifiers), and the
string keeps its trailing newline, as `$_` does, so that `$` and `.` behave
the same.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

_VARIABLE = re.compile(r"\$\{(\w+)\}|\$([A-Z_][A-Z0-9_]*)")
_REPLACEMENT_VARIABLE = re.compile(r"\$(\d+)|\$\{(\d+)\}|\$\{(\w+)\}|\$([A-Za-z_]\w*)")
_ESCAPES = {"n": "\n", "t": "\t"}


def interpolate(pattern: str, variables: Mapping[str, str]) -> str:
    """Replace `$NAME` and `${NAME}` in a pattern with the variables' values."""
    return _VARIABLE.sub(lambda m: variables[m.group(1) or m.group(2)], pattern)


def replacement_pieces(replacement: str, variables: Mapping[str, str]) -> list[str | int]:
    """A Perl replacement as literal strings and group numbers."""
    pieces: list[str | int] = []
    literal: list[str] = []
    i = 0
    while i < len(replacement):
        char = replacement[i]
        if char == "\\" and i + 1 < len(replacement):
            following = replacement[i + 1]
            if following.isdigit():
                pieces += ["".join(literal), int(following)]
                literal = []
            else:
                literal.append(_ESCAPES.get(following, following))
            i += 2
            continue
        if char == "$":
            match = _REPLACEMENT_VARIABLE.match(replacement, i)
            if match and (match.group(1) or match.group(2)):
                pieces += ["".join(literal), int(match.group(1) or match.group(2))]
                literal = []
                i = match.end()
                continue
            if match:
                literal.append(variables[match.group(3) or match.group(4)])
                i = match.end()
                continue
        literal.append(char)
        i += 1
    pieces.append("".join(literal))
    return [piece for piece in pieces if piece != ""]


_Compiled = tuple[re.Pattern[str], Callable[[re.Match[str]], str]]
_CACHE: dict[tuple[str, str, int], _Compiled] = {}


def _compile(pattern: str, replacement: str, variables: Mapping[str, str]) -> _Compiled:
    key = (pattern, replacement, id(variables))
    found = _CACHE.get(key)
    if found is None:
        regex = re.compile(interpolate(pattern, variables), re.ASCII)
        pieces = replacement_pieces(replacement, variables)

        def expand(match: re.Match[str]) -> str:
            return "".join(p if isinstance(p, str) else (match.group(p) or "") for p in pieces)

        found = _CACHE[key] = (regex, expand)
    return found


class PerlString:
    """The line a script rewrites, Perl's `$_`, and the script's scalar variables.

    `variables` must be a module-level mapping: compiled rules are cached by
    its identity.
    """

    def __init__(self, text: str, variables: Mapping[str, str] | None = None):
        self.text = text
        self.variables = variables if variables is not None else {}
        self.scalars: dict[str, float] = {}

    def sub(self, pattern: str, replacement: str, g: bool = False) -> bool:
        """Apply `s/pattern/replacement/` (with `g`, `s///g`); whether anything changed."""
        regex, expand = _compile(pattern, replacement, self.variables)
        self.text, count = regex.subn(expand, self.text, count=0 if g else 1)
        return count > 0

    def search(self, pattern: str) -> bool:
        """Whether `$_ =~ /pattern/` matches."""
        regex, _ = _compile(pattern, "", self.variables)
        return regex.search(self.text) is not None

    def set(self, name: str, value: float) -> bool:
        """Assign a scalar, as `($j=1.5)`; true when the value is, as in Perl."""
        self.scalars[name] = value
        return bool(value)

    def get(self, name: str) -> float:
        return self.scalars.get(name, 0)

    def mark_children(self) -> None:
        """Turn the outermost `[` and `]` between the two `^` marks into `<` and `>`.

        This is the character loop at the start of `annotate-gcg15.pl`'s
        top-down pass, which marks the children of the current constituent.
        """
        chars = list(self.text)
        depth = 0
        for i in range(self.text.index("^"), self.text.rindex("^")):
            if chars[i] == "[":
                if depth == 0:
                    chars[i] = "<"
                depth += 1
            if chars[i] == "]":
                depth -= 1
                if depth == 0:
                    chars[i] = ">"
        self.text = "".join(chars)
