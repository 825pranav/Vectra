"""Filter expressions: parse, evaluate over attribute columns, estimate selectivity.

Grammar (keywords are case-insensitive)::

    expr    := and ("OR" and)*
    and     := not ("AND" not)*
    not     := "NOT" not | "(" expr ")" | cmp
    cmp     := IDENT op literal | IDENT "IN" "(" literal ("," literal)* ")"
    op      := "==" | "!=" | "<" | "<=" | ">" | ">="
    literal := number | "string" | 'string' | true | false

Semantics: a record without the attribute never satisfies a comparison (so
``price != 5`` excludes records with no price), and ``NOT`` is plain boolean
negation of that result.

The same AST evaluates in two ways: over whole columns (a bitmap for the
bitmap / brute-force strategies) and over a handful of candidate rows (the
post-filter strategy), so both paths share one definition of "matches".
"""

# Imports: regex for the tokenizer, dataclasses for the AST, NumPy for column masks.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from storage.meta import Column


# Raised for any bad filter text or type mismatch; it is a ValueError so callers map it to 400s.
class FilterError(ValueError):
    pass


# ---- AST --------------------------------------------------------------------


# AST leaf: one comparison like price < 50 (attribute name, operator, literal).
@dataclass(frozen=True)
class Cmp:
    key: str
    op: str
    value: Any


# AST leaf: membership test like category IN ("a", "b").
@dataclass(frozen=True)
class In:
    key: str
    values: tuple


# AST node: boolean NOT of its child.
@dataclass(frozen=True)
class Not:
    arg: Any


# AST node: all children must match.
@dataclass(frozen=True)
class And:
    args: tuple


# AST node: at least one child must match.
@dataclass(frozen=True)
class Or:
    args: tuple


# Any filter tree node; this is what parse() returns and evaluate()/estimate() consume.
Node = Cmp | In | Not | And | Or


# Turn a tree into normalised text (AND/OR children sorted) so equal filters share one cache entry.
def canonical(node: Node) -> str:
    """Stable text form, used as the mask-cache key."""
    if isinstance(node, Cmp):
        return f"{node.key}{node.op}{node.value!r}"
    if isinstance(node, In):
        return f"{node.key} IN {sorted(map(repr, node.values))}"
    if isinstance(node, Not):
        return f"NOT({canonical(node.arg)})"
    op = " AND " if isinstance(node, And) else " OR "
    return "(" + op.join(sorted(canonical(a) for a in node.args)) + ")"


# ---- parser -------------------------------------------------------------------

# One regex that matches the next token: number, quoted string, operator, punctuation or word.
_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
      | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
      | (?P<op>==|!=|<=|>=|<|>)
      | (?P<punct>[(),])
      | (?P<word>[A-Za-z_][A-Za-z0-9_.]*)
    )""",
    re.VERBOSE,
)
# Words that are grammar keywords or boolean literals rather than attribute names.
_KEYWORDS = {"and", "or", "not", "in", "true", "false"}


# Lexer: filter text -> list of (kind, value) tokens, ending with an "end" token.
def _tokenize(text: str) -> list[tuple[str, Any]]:
    toks: list[tuple[str, Any]] = []
    pos = 0
    text = text.rstrip()
    # Match tokens left to right; anything the regex can't match is a syntax error.
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise FilterError(f"unexpected character at {pos}: {text[pos : pos + 10]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        # Classify each match: numbers become floats, quoted strings are unescaped, keywords
        # lowercased.
        if kind == "num":
            toks.append(("lit", float(val)))
        elif kind == "str":
            toks.append(("lit", bytes(val[1:-1], "utf-8").decode("unicode_escape")))
        elif kind == "word" and val.lower() in _KEYWORDS:
            low = val.lower()
            if low in ("true", "false"):
                toks.append(("lit", low == "true"))
            else:
                toks.append((low, low))
        elif kind == "word":
            toks.append(("ident", val))
        else:
            toks.append((val, val))
    toks.append(("end", None))
    return toks


# Hand-written recursive-descent parser; one method per grammar rule in the module docstring.
class _Parser:
    # Tokenize up front and keep an index into the token list.
    def __init__(self, text: str) -> None:
        self.toks = _tokenize(text)
        self.i = 0

    # Look at the next token's kind without consuming it.
    def peek(self) -> str:
        return self.toks[self.i][0]

    # Friendly names for token kinds, used in error messages.
    _NAMES = {"lit": "a value", "ident": "an attribute name", "end": "end of input"}

    # Consume the next token if it has the expected kind, else raise a readable FilterError.
    def take(self, kind: str) -> Any:
        k, v = self.toks[self.i]
        if k != kind:
            want = self._NAMES.get(kind, repr(kind))
            found = "end of input" if k == "end" else repr(v)
            raise FilterError(f"expected {want}, found {found}")
        self.i += 1
        return v

    # Parse the whole input and insist nothing is left over.
    def parse(self) -> Node:
        node = self.expr()
        self.take("end")
        return node

    # expr: one or more AND-groups joined by OR (OR binds loosest).
    def expr(self) -> Node:
        args = [self.conj()]
        while self.peek() == "or":
            self.take("or")
            args.append(self.conj())
        return args[0] if len(args) == 1 else Or(tuple(args))

    # conj: one or more NOT-terms joined by AND.
    def conj(self) -> Node:
        args = [self.neg()]
        while self.peek() == "and":
            self.take("and")
            args.append(self.neg())
        return args[0] if len(args) == 1 else And(tuple(args))

    # neg: NOT <term>, a parenthesised expression, or a single comparison.
    def neg(self) -> Node:
        if self.peek() == "not":
            self.take("not")
            return Not(self.neg())
        if self.peek() == "(":
            self.take("(")
            node = self.expr()
            self.take(")")
            return node
        return self.cmp()

    # cmp: attribute name followed by IN (...) or by an operator and one literal.
    def cmp(self) -> Node:
        key = self.take("ident")
        if self.peek() == "in":
            self.take("in")
            self.take("(")
            vals = [self.take("lit")]
            while self.peek() == ",":
                self.take(",")
                vals.append(self.take("lit"))
            self.take(")")
            return In(key, tuple(vals))
        op = self.peek()
        if op not in ("==", "!=", "<", "<=", ">", ">="):
            raise FilterError(f"expected a comparison after {key!r}")
        self.take(op)
        return Cmp(key, op, self.take("lit"))


# Public entry: filter text -> AST. Called by Collection.search() before planning.
def parse(text: str) -> Node:
    return _Parser(text).parse()


# ---- evaluation ---------------------------------------------------------------


# Reject comparing a column with a literal of the wrong type (e.g. a string tag with a number).
def _check_value(col: Column, key: str, v: Any) -> None:
    if col.kind == "str" and not isinstance(v, str):
        raise FilterError(f"attribute {key!r} is a string; compared with {v!r}")
    if col.kind == "num" and (isinstance(v, bool) or not isinstance(v, float)):
        raise FilterError(f"attribute {key!r} is numeric; compared with {v!r}")
    if col.kind == "bool" and not isinstance(v, bool):
        raise FilterError(f"attribute {key!r} is boolean; compared with {v!r}")


# Evaluate one comparison over a slice of a column, giving a boolean array.
def _eval_cmp(col: Column, data: np.ndarray, node: Cmp) -> np.ndarray:
    _check_value(col, node.key, node.value)
    # String columns hold integer codes, so only == and != make sense; look up the literal's code.
    # For !=, rows with no value (negative code) are excluded.
    if col.kind == "str":
        if node.op not in ("==", "!="):
            raise FilterError(f"operator {node.op} is not defined for strings")
        code = col.lookup.get(node.value, -2)  # -2 matches nothing, not even missing
        if node.op == "==":
            return data == code
        return (data != code) & (data >= 0)
    # Number and bool columns are float64 with NaN for missing; NaN compares False, so it never
    # matches.
    v = float(node.value)
    with np.errstate(invalid="ignore"):
        if node.op == "==":
            return data == v
        if node.op == "!=":
            return (data != v) & ~np.isnan(data)
        if node.op == "<":
            return data < v
        if node.op == "<=":
            return data <= v
        if node.op == ">":
            return data > v
        return data >= v


# Turn a filter tree into a boolean mask over rows; used for planner masks and post-filter checks.
def evaluate(node: Node, columns: dict[str, Column], rows: np.ndarray | int) -> np.ndarray:
    """Boolean mask. ``rows`` is either a count (evaluate ids 0..rows-1) or an
    array of internal ids (evaluate just those, e.g. post-filter candidates)."""

    # Fetch the column for a tag and slice either the first `rows` rows or just the given row ids.
    def data_of(key: str) -> tuple[Column, np.ndarray]:
        col = columns.get(key)
        if col is None:
            raise FilterError(f"unknown attribute {key!r}")
        d = col.data
        return col, (d[:rows] if isinstance(rows, int) else d[rows])

    # Recursive walk: leaves compare columns, NOT inverts, AND/OR combine child masks.
    def go(n: Node) -> np.ndarray:
        if isinstance(n, Cmp):
            col, d = data_of(n.key)
            return _eval_cmp(col, d, n)
        if isinstance(n, In):
            col, d = data_of(n.key)
            out = np.zeros(d.shape[0], dtype=np.bool_)
            for v in n.values:
                out |= _eval_cmp(col, d, Cmp(n.key, "==", v))
            return out
        if isinstance(n, Not):
            return ~go(n.arg)
        parts = [go(a) for a in n.args]
        out = parts[0].copy()
        for p in parts[1:]:
            if isinstance(n, And):
                out &= p
            else:
                out |= p
        return out

    return go(node)


# ---- selectivity estimation -----------------------------------------------------


# Estimate the share of rows a filter matches using stored per-tag stats (no column scan).
# Called by the planner on big collections; None means "no stats, count exactly instead".
def estimate(node: Node, stats: dict[str, dict[str, Any]]) -> float | None:
    """Estimated fraction of records matching, from per-attribute statistics.

    Numeric attributes use a 101-point quantile sketch (plus exact value counts
    when there are few distinct values), strings/booleans use value
    frequencies. Conjunctions assume independence. Returns None when an
    attribute has no statistics yet (the caller then measures exactly).
    """

    # Estimated fraction for one comparison against one tag's stats.
    def cmp_frac(st: dict[str, Any], op: str, v: Any) -> float:
        present = 1.0 - st["null_frac"]
        freq = st.get("freq")
        # Few distinct values: exact counts per value give the answer directly.
        if freq is not None:
            total = max(st["n"], 1)
            key = _freq_key(v)
            if op in ("==", "!="):
                eq = freq.get(key, 0) / total
                return eq if op == "==" else present - eq
            vals = np.array([float(x) for x in freq], dtype=np.float64)
            cnt = np.array(list(freq.values()), dtype=np.float64)
            m = _num_op(vals, op, float(v))
            return float(cnt[m].sum() / total)
        # Many values: use the quantile sketch; equality assumes values are spread evenly.
        qs = np.asarray(st.get("quantiles") or [], dtype=np.float64)
        if qs.size == 0:
            return 0.0
        if op in ("==", "!="):
            eq = present / max(st.get("n_distinct", 1), 1) if qs[0] <= v <= qs[-1] else 0.0
            return eq if op == "==" else present - eq
        # Range ops: interpolate where v falls among the quantiles to get the fraction below it.
        below = float(np.interp(v, qs, np.linspace(0, 1, qs.size), left=0.0, right=1.0))
        frac = below if op in ("<", "<=") else 1.0 - below
        return present * frac

    # Combine leaves: NOT is 1 - p, AND multiplies (assumes independence), OR is 1 - prod(1 - p).
    def go(n: Node) -> float | None:
        if isinstance(n, Cmp | In):
            st = stats.get(n.key)
            if st is None:
                return None
            if isinstance(n, Cmp):
                return cmp_frac(st, n.op, n.value)
            return min(1.0, sum(cmp_frac(st, "==", v) for v in n.values))
        if isinstance(n, Not):
            a = go(n.arg)
            return None if a is None else 1.0 - a
        parts = [go(a) for a in n.args]
        if any(p is None for p in parts):
            return None
        if isinstance(n, And):
            return float(np.prod(parts))
        out = 0.0
        for p in parts:
            out = out + p - out * p
        return out

    # Clamp the final estimate to [0, 1].
    s = go(node)
    return None if s is None else float(min(max(s, 0.0), 1.0))


# Key used to look up a value in the frequency table.
def _freq_key(v: Any) -> str:
    # numbers and booleans are keyed by repr(float), matching Column.stats
    return v if isinstance(v, str) else repr(float(v))


# Apply a range operator to an array of distinct values (used with the frequency table).
def _num_op(vals: np.ndarray, op: str, v: float) -> np.ndarray:
    return {
        "<": vals < v,
        "<=": vals <= v,
        ">": vals > v,
        ">=": vals >= v,
    }[op]
