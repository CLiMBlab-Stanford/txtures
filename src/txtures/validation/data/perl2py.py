"""Translate the restricted Perl of the ModelBlocks tree scripts into Python.

This one-off translator wrote the first versions of `txtures/gcg/fixes.py`
(from `make-trees-fixed.pl`) and `txtures/gcg/rules.py` (from
`annotate-gcg15.pl`): it copies every substitution verbatim and turns the
control flow into Python. It is kept to document how those files were made;
they have been edited since, so do not regenerate them blindly. Run it as a
script: `python perl2py.py SCRIPT.pl NAME > out.py`.


Handles: s/// substitutions, m// matches against $_, while/if/else blocks,
&&, ||, !, parenthesized expressions, ($x=N) assignments, $x==N tests,
debug(...) calls (false), comments, and single-quoted pattern variables.
Usage: perl2py.py SCRIPT.pl FUNCNAME > out.py  (body of `while (<>) {...}`).
"""

import re
import sys

whole = open(sys.argv[1]).read()
func = sys.argv[2]
_start = re.search(r"^while \( <> \) \{", whole, re.M).start()
src = whole[_start:]
PRE_VARS = {}
for _m in re.finditer(r"^\s*\$(\w+) = '((?:[^'\\]|\\.)*)';", whole[:_start], re.M):
    PRE_VARS[_m.group(1)] = re.sub(r"\\([\\'])", r"\1", _m.group(2))

# ---------------------------------------------------------------- tokenizer
TOK = []
i = 0
n = len(src)
line = whole[:_start].count(chr(10)) + 1


def add(kind, val, ln):
    TOK.append((kind, val, ln))


while i < n:
    c = src[i]
    if c == "\n":
        line += 1
        i += 1
        continue
    if c in " \t\r":
        i += 1
        continue
    if c == "#":
        j = src.index("\n", i)
        add("comment", src[i + 1 : j].rstrip(), line)
        i = j
        continue
    # substitution s/../../flags (not part of an identifier)
    if c == "s" and src[i + 1] == "/" and not (src[i - 1].isalnum() or src[i - 1] in "_$"):
        j = i + 2
        parts = []
        start = j
        while len(parts) < 2:
            if src[j] == "\\":
                j += 2
                continue
            if src[j] == "/":
                parts.append(src[start:j])
                start = j + 1
            j += 1
        k = j
        while src[k].isalpha():
            k += 1
        add("subst", (parts[0], parts[1], src[j:k]), line)
        i = k
        continue
    if src.startswith("=~", i):
        add("op", "=~", line)
        i += 2
        continue
    if c == "/" and TOK and TOK[-1][1] == "=~":
        j = i + 1
        while src[j] != "/":
            j += 2 if src[j] == "\\" else 1
        add("match", src[i + 1 : j], line)
        i = j + 1
        continue
    if c == '"':
        j = i + 1
        while src[j] != '"':
            j += 2 if src[j] == "\\" else 1
        add("dstr", src[i + 1 : j], line)
        i = j + 1
        continue
    if c == "'":
        j = i + 1
        buf = []
        while src[j] != "'":
            if src[j] == "\\" and src[j + 1] in "\\'":
                buf.append(src[j + 1])
                j += 2
            else:
                buf.append(src[j])
                j += 1
        add("str", "".join(buf), line)
        i = j + 1
        continue
    m = re.compile(
        r"\$\w+(\{\w\})?|\d+(\.\d+)?|\w+|&&|\|\||==|\+\+|<>|[(){};=!,.<\-+\[\]%>*]"
    ).match(src, i)
    if not m:
        raise SystemExit(f"cannot tokenize at line {line}: {src[i : i + 40]!r}")
    v = m.group()
    add("num" if v[0].isdigit() else "id" if (v[0].isalpha() or v[0] == "$") else "op", v, line)
    i = m.end()

# --------------------------------------------------------------- parser
pos = 0


def peek(k=0):
    return TOK[pos + k] if pos + k < len(TOK) else ("eof", None, None)


def take(val=None):
    global pos
    t = TOK[pos]
    if val is not None and t[1] != val:
        raise SystemExit(f"expected {val!r} at line {t[2]}, got {t}")
    pos += 1
    return t


out = []
VARS = dict(PRE_VARS)


def emit(depth, text):
    out.append("    " * depth + text)


def comments(depth):
    while peek()[0] == "comment":
        emit(depth, "#" + take()[1])


def skip_to(val):
    while take()[1] != val:
        pass


def pyrepr(s):
    if "\\" not in s:
        return repr(s)
    if (len(s) - len(s.rstrip("\\"))) % 2 == 1:
        return repr(s)
    for q in ("'", '"', "'''", '"""'):
        if q not in s and not s.endswith(q[0]):
            return "r" + q + s + q
    return repr(s)


def parse_primary():
    t = peek()
    kind, val, ln = t
    if kind == "subst":
        take()
        p, r, f = val
        return f"tree.sub({pyrepr(p)}, {pyrepr(r)}{', g=True' if 'g' in f else ''})"
    if val == "(":
        # ($j=N) or (expr)
        if peek(1)[0] == "id" and peek(2)[1] == "=" and peek(3)[0] == "num":
            take("(")
            name = take()[1][1:]
            take("=")
            num = take()[1]
            take(")")
            return f"tree.set({name!r}, {num})"
        take("(")
        e = parse_or()
        while peek()[0] == "comment":
            take()
        take(")")
        return f"({e})"
    if val == "!":
        take()
        return f"not {parse_primary()}"
    if val == "debug":
        take()
        take("(")
        d = 1
        while d:
            v = take()[1]
            d += {"(": 1, ")": -1}.get(v, 0)
        return "False"
    if kind == "num":
        take()
        return {"0": "False", "1": "True"}.get(val, val)
    if val == "$_" and peek(1)[1] == "=~":
        take()
        take()
        t2 = take()
        if t2[0] == "match":
            return f"tree.search({pyrepr(t2[1])})"
        p, r, f = t2[1]
        return f"tree.sub({pyrepr(p)}, {pyrepr(r)}{', g=True' if 'g' in f else ''})"
    if kind == "id" and val.startswith("$opts{"):
        take()
        return "False"  # no options are given in the ModelBlocks pipeline
    if val == "$NOTPSG":
        take()
        return "False"
    if kind == "id" and val.startswith("$") and peek(1)[1] == "==":
        take()
        take()
        num = take()[1]
        return f"tree.get({val[1:]!r}) == {num}"
    raise SystemExit(f"unexpected {t}")


def lead_comments():
    found = []
    while peek()[0] == "comment":
        found.append("#" + take()[1])
    return "".join(c + "\n" for c in found)


def parse_and():
    parts = [lead_comments() + parse_primary()]
    while True:
        trailing = lead_comments()
        if peek()[1] == "&&":
            take()
            parts.append(trailing + lead_comments() + parse_primary())
        else:
            parts[-1] += ("\n" + trailing.rstrip("\n")) if trailing else ""
            return " and ".join(parts)


def parse_or():
    parts = [parse_and()]
    while True:
        if peek()[1] == "||":
            take()
            parts.append(parse_and())
        else:
            kept = [p for p in parts if p.strip() != "False"]
            parts = kept or ["False"]
            multi = len(parts) > 3 or any("\n" in p for p in parts)
            return "\n or ".join(parts) if multi else " or ".join(parts)


def chain_cond():
    """Parse a condition; return python text (wrapped for multi-line)."""
    e = parse_or()
    return f"(\n        {e}\n    )" if "\n" in e else e


def parse_block(depth):
    take("{")
    body_start = len(out)
    while peek()[1] != "}":
        parse_stmt(depth)
    take("}")
    if len(out) == body_start or all(l.strip().startswith("#") for l in out[body_start:]):
        emit(depth, "pass")


def parse_stmt(depth):
    comments(depth)
    kind, val, ln = peek()
    if val == "}":
        return
    if val == ";":
        take()
        return
    if val == "while":
        take()
        take("(")
        e = parse_or()
        while peek()[0] == "comment":
            take()
        take(")")
        emit(depth, f"while (\n{e}\n):" if "\n" in e else f"while {e}:")
        parse_block(depth + 1)
        return
    if val == "if":
        take()
        take("(")
        e = parse_or()
        while peek()[0] == "comment":
            take()
        take(")")
        emit(depth, f"if (\n{e}\n):" if "\n" in e else f"if {e}:")
        parse_block(depth + 1)
        if peek()[1] == "else":
            take()
            emit(depth, "else:")
            parse_block(depth + 1)
        return
    if val == "for":
        # the mark-children loop of annotate-gcg
        skip_to("{")
        d = 1
        while d:
            v = take()[1]
            d += {"{": 1, "}": -1}.get(v, 0)
        emit(depth, "tree.mark_children()")
        return
    if val == "$line":
        skip_to("}")  # progress report
        return
    if val in ("print", "chomp"):
        skip_to(";")
        return
    if kind == "id" and val.startswith("$") and peek(1)[1] == "=" and peek(2)[0] == "num":
        name = take()[1][1:]
        take("=")
        num = take()[1]
        take(";")
        emit(depth, f"tree.set({name!r}, {num})")
        return
    if kind == "id" and val.startswith("$") and peek(1)[1] == "=" and peek(2)[0] == "str":
        name = take()[1][1:]
        take("=")
        s = take()[1]
        take(";")
        VARS[name] = s
        emit(depth, f"# ${name} is VARIABLES[{name!r}]")
        return
    if val == "$line++":
        skip_to("}")
        return
    e = parse_or()
    take(";") if peek()[1] == ";" else None
    if e.strip() == "False":
        return
    if e.endswith(" and False") and "\n" not in e:
        e = e[: -len(" and False")]
    emit(depth, f"(\n{e}\n)" if "\n" in e else e)


# find `while ( <> ) {`
while not (peek()[1] == "while" and peek(2)[1] == "<>"):
    kind, val, ln = take()
    if kind == "id" and val.startswith("$") and peek()[1] == "=" and peek(1)[0] == "str":
        take("=")
        VARS[val[1:]] = take()[1]
take("while")
take("(")
take("<>")
take(")")
emit(1, "")
parse_block(1)

print(f"VARIABLES_{func.upper()} = {{")
for k, v in VARS.items():
    print(f"    {k!r}: {pyrepr(v)},")
print("}\n")
print(f"def {func}(tree):")
print("\n".join(out))
