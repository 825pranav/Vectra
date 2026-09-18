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

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from storage.meta import Column


class FilterError(ValueError):
    pass


# ---- AST --------------------------------------------------------------------


@dataclass(frozen=True)
class Cmp:
    key: str
    op: str
    value: Any


@dataclass(frozen=True)
class In:
    key: str
    values: tuple


@dataclass(frozen=True)
class Not:
    arg: Any


@dataclass(frozen=True)
class And:
    args: tuple


@dataclass(frozen=True)
class Or:
    args: tuple


Node = Cmp | In | Not | And | Or


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
_KEYWORDS = {"and", "or", "not", "in", "true", "false"}


def _tokenize(text: str) -> list[tuple[str, Any]]:
    toks: list[tuple[str, Any]] = []
    pos = 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise FilterError(f"unexpected character at {pos}: {text[pos : pos + 10]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
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


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self) -> str:
        return self.toks[self.i][0]

    def take(self, kind: str) -> Any:
        k, v = self.toks[self.i]
        if k != kind:
            raise FilterError(f"expected {kind}, found {v if v is not None else 'end of input'}")
        self.i += 1
        return v

    def parse(self) -> Node:
        node = self.expr()
        self.take("end")
        return node

    def expr(self) -> Node:
        args = [self.conj()]
        while self.peek() == "or":
            self.take("or")
            args.append(self.conj())
        return args[0] if len(args) == 1 else Or(tuple(args))

    def conj(self) -> Node:
        args = [self.neg()]
        while self.peek() == "and":
            self.take("and")
            args.append(self.neg())
        return args[0] if len(args) == 1 else And(tuple(args))

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


def parse(text: str) -> Node:
    return _Parser(text).parse()


# ---- evaluation ---------------------------------------------------------------


def _check_value(col: Column, key: str, v: Any) -> None:
    if col.kind == "str" and not isinstance(v, str):
        raise FilterError(f"attribute {key!r} is a string; compared with {v!r}")
    if col.kind == "num" and (isinstance(v, bool) or not isinstance(v, float)):
        raise FilterError(f"attribute {key!r} is numeric; compared with {v!r}")
    if col.kind == "bool" and not isinstance(v, bool):
        raise FilterError(f"attribute {key!r} is boolean; compared with {v!r}")


def _eval_cmp(col: Column, data: np.ndarray, node: Cmp) -> np.ndarray:
    _check_value(col, node.key, node.value)
    if col.kind == "str":
        if node.op not in ("==", "!="):
            raise FilterError(f"operator {node.op} is not defined for strings")
        code = col.lookup.get(node.value, -2)  # -2 matches nothing, not even missing
        if node.op == "==":
            return data == code
        return (data != code) & (data >= 0)
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


def evaluate(node: Node, columns: dict[str, Column], rows: np.ndarray | int) -> np.ndarray:
    """Boolean mask. ``rows`` is either a count (evaluate ids 0..rows-1) or an
    array of internal ids (evaluate just those, e.g. post-filter candidates)."""

    def data_of(key: str) -> tuple[Column, np.ndarray]:
        col = columns.get(key)
        if col is None:
            raise FilterError(f"unknown attribute {key!r}")
        d = col.data
        return col, (d[:rows] if isinstance(rows, int) else d[rows])

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


def estimate(node: Node, stats: dict[str, dict[str, Any]]) -> float | None:
    """Estimated fraction of records matching, from per-attribute statistics.

    Numeric attributes use a 101-point quantile sketch (plus exact value counts
    when there are few distinct values), strings/booleans use value
    frequencies. Conjunctions assume independence. Returns None when an
    attribute has no statistics yet (the caller then measures exactly).
    """

    def cmp_frac(st: dict[str, Any], op: str, v: Any) -> float:
        present = 1.0 - st["null_frac"]
        freq = st.get("freq")
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
        qs = np.asarray(st.get("quantiles") or [], dtype=np.float64)
        if qs.size == 0:
            return 0.0
        if op in ("==", "!="):
            eq = present / max(st.get("n_distinct", 1), 1) if qs[0] <= v <= qs[-1] else 0.0
            return eq if op == "==" else present - eq
        below = float(np.interp(v, qs, np.linspace(0, 1, qs.size), left=0.0, right=1.0))
        frac = below if op in ("<", "<=") else 1.0 - below
        return present * frac

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

    s = go(node)
    return None if s is None else float(min(max(s, 0.0), 1.0))


def _freq_key(v: Any) -> str:
    # numbers and booleans are keyed by repr(float), matching Column.stats
    return v if isinstance(v, str) else repr(float(v))


def _num_op(vals: np.ndarray, op: str, v: float) -> np.ndarray:
    return {
        "<": vals < v,
        "<=": vals <= v,
        ">": vals > v,
        ">=": vals >= v,
    }[op]
