"""Filter language + planner strategies.

The property tests generate random predicates with a seeded RNG and check each
planner strategy against an independent pure-Python evaluator (no NumPy
columns, no parser): results must never contain a non-matching or deleted id,
and the brute-force strategy must return exactly the true filtered top-k.
"""

import numpy as np
import pytest

from engine.collection import Collection, InvalidArgument
from engine.filters import And, Cmp, FilterError, In, Not, Or, canonical, estimate, evaluate, parse
from storage.meta import Column

CATS = ["shoes", "hats", "bags", "coats", "socks", "belts"]
CAT_P = [0.4, 0.25, 0.15, 0.1, 0.07, 0.03]

# ---- parser -------------------------------------------------------------------


def test_parse_precedence():
    n = parse('a < 1 OR b == "x" AND NOT c >= 2')
    assert n == Or((Cmp("a", "<", 1.0), And((Cmp("b", "==", "x"), Not(Cmp("c", ">=", 2.0))))))


def test_parse_parens_in_and_literals():
    n = parse("(price <= 5e2 or flag == true) and cat in ('a', \"b\")")
    assert n == And(
        (Or((Cmp("price", "<=", 500.0), Cmp("flag", "==", True))), In("cat", ("a", "b")))
    )
    assert parse("x == -3.5") == Cmp("x", "==", -3.5)


@pytest.mark.parametrize(
    "bad", ["", "price <", "price < 5 AND", "(price < 5", "price ~ 5", "5 < price", "a IN ()"]
)
def test_parse_errors(bad):
    with pytest.raises(FilterError):
        parse(bad)


def test_canonical_ignores_operand_order():
    assert canonical(parse("a < 1 AND b == 'x'")) == canonical(parse("b == 'x' AND a < 1"))
    assert canonical(parse("a < 1")) != canonical(parse("a <= 1"))


# ---- evaluation ---------------------------------------------------------------


@pytest.fixture
def cols():
    price = Column("num", 8)
    cat = Column("str", 8)
    stock = Column("bool", 8)
    for i, (p, c, s) in enumerate(
        [(10, "a", True), (20, "b", False), (30, "a", None), (None, "c", True), (50, None, False)]
    ):
        if p is not None:
            price.set(i, p)
        if c is not None:
            cat.set(i, c)
        if s is not None:
            stock.set(i, s)
    return {"price": price, "cat": cat, "stock": stock}


def test_evaluate_semantics(cols):
    def ev(t):
        return evaluate(parse(t), cols, 5).tolist()

    assert ev("price < 25") == [True, True, False, False, False]
    assert ev("price != 20") == [True, False, True, False, True]  # missing never matches
    assert ev("NOT price == 20") == [True, False, True, True, True]
    assert ev("cat == 'a'") == [True, False, True, False, False]
    assert ev("cat != 'a'") == [False, True, False, True, False]
    assert ev("cat == 'zzz'") == [False] * 5
    assert ev("cat IN ('b', 'c')") == [False, True, False, True, False]
    assert ev("stock == true") == [True, False, False, True, False]
    assert ev("price >= 20 AND stock == false") == [False, True, False, False, True]
    # evaluating a subset of rows gives the same answer as the full mask
    assert evaluate(parse("price < 25"), cols, np.array([4, 0])).tolist() == [False, True]


@pytest.mark.parametrize(
    "bad", ["price == 'x'", "cat < 'a'", "stock == 1", "nope == 1", "cat == 3"]
)
def test_evaluate_type_errors(cols, bad):
    with pytest.raises(FilterError):
        evaluate(parse(bad), cols, 5)


def test_estimate_close_to_truth():
    rng = np.random.default_rng(0)
    n = 20_000
    price, cat, rating = Column("num", n), Column("str", n), Column("num", n)
    price.data[:] = rng.uniform(0, 1000, n)
    rating.data[:] = rng.integers(1, 6, n)
    for c in CATS:
        cat.code(c)
    cat.data[:] = rng.choice(len(CATS), n, p=CAT_P)
    cols = {"price": price, "cat": cat, "rating": rating}
    stats = {k: c.stats(n) for k, c in cols.items()}
    for text in [
        "price < 100",
        "price >= 900",
        "cat == 'shoes'",
        "cat IN ('bags', 'belts')",
        "rating == 3",
        "rating <= 2",
        "price < 500 AND cat == 'hats'",
        "cat == 'socks' OR price > 950",
        "NOT cat == 'shoes'",
    ]:
        node = parse(text)
        truth = evaluate(node, cols, n).mean()
        assert abs(estimate(node, stats) - truth) < 0.02, text
    assert estimate(parse("unknown == 1"), stats) is None


# ---- property tests over the three strategies -----------------------------------


def ref_match(attrs: dict, node) -> bool:
    """Independent evaluator over a record's attribute dict."""
    if isinstance(node, Cmp):
        if node.key not in attrs:
            return False
        a, v = attrs[node.key], node.value
        if isinstance(a, bool) or isinstance(a, str):
            return (a == v) if node.op == "==" else (a != v)
        a = float(a)
        return {
            "==": a == v,
            "!=": a != v,
            "<": a < v,
            "<=": a <= v,
            ">": a > v,
            ">=": a >= v,
        }[node.op]
    if isinstance(node, In):
        return node.key in attrs and attrs[node.key] in node.values
    if isinstance(node, Not):
        return not ref_match(attrs, node.arg)
    if isinstance(node, And):
        return all(ref_match(attrs, a) for a in node.args)
    return any(ref_match(attrs, a) for a in node.args)


def random_predicate(rng: np.random.Generator, depth: int = 0) -> str:
    r = rng.random()
    if depth < 2 and r < 0.35:
        op = "AND" if rng.random() < 0.6 else "OR"
        return f"({random_predicate(rng, depth + 1)} {op} {random_predicate(rng, depth + 1)})"
    if depth < 2 and r < 0.45:
        return f"NOT {random_predicate(rng, depth + 1)}"
    kind = rng.integers(0, 4)
    if kind == 0:
        op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
        return f"price {op} {int(rng.integers(0, 1000))}"
    if kind == 1:
        if rng.random() < 0.3:
            vals = rng.choice(CATS, size=int(rng.integers(1, 4)), replace=False)
            return "cat IN (" + ", ".join(f"'{v}'" for v in vals) + ")"
        return f"cat {rng.choice(['==', '!='])} '{rng.choice(CATS)}'"
    if kind == 2:
        return f"in_stock == {'true' if rng.random() < 0.5 else 'false'}"
    return f"rating {rng.choice(['==', '<=', '>'])} {int(rng.integers(1, 6))}"


@pytest.fixture(scope="module")
def filtered(tmp_path_factory):
    rng = np.random.default_rng(123)
    n, d = 4000, 16
    x = rng.normal(size=(n, d)).astype(np.float32)
    attrs = []
    for _ in range(n):
        a = {}
        if rng.random() < 0.95:
            a["price"] = float(rng.integers(0, 1000))
        if rng.random() < 0.97:
            a["cat"] = str(rng.choice(CATS, p=CAT_P))
        a["in_stock"] = bool(rng.random() < 0.7)
        if rng.random() < 0.9:
            a["rating"] = float(rng.integers(1, 6))
        attrs.append(a)
    col = Collection.create(tmp_path_factory.mktemp("f") / "c", "c", dim=d)
    ids = [f"r{i}" for i in range(n)]
    col.upsert(ids, x, attrs)
    dead = rng.choice(n, 200, replace=False)
    col.delete([ids[i] for i in dead])
    yield col, x, attrs, set(dead.tolist())
    col.close()


@pytest.mark.parametrize("strategy", [None, "post_filter", "bitmap", "brute_force"])
def test_property_never_returns_non_matching(filtered, strategy):
    col, x, attrs, dead = filtered
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(60):
        text = random_predicate(rng)
        node = parse(text)
        match = np.array([ref_match(a, node) for a in attrs]) & ~np.isin(
            np.arange(len(attrs)), list(dead)
        )
        q = rng.normal(size=x.shape[1]).astype(np.float32)
        res = col.search(q, k=10, filter=text, ef=64, strategy=strategy)
        got = [int(i[1:]) for i in res.ids]
        assert all(match[i] for i in got), (text, strategy)
        assert len(set(got)) == len(got)
        assert np.all(np.diff(res.distances) >= 0)
        want = min(10, int(match.sum()))
        if strategy in (None, "bitmap", "brute_force"):
            assert len(got) == want, (text, strategy, len(got), want)
        if strategy == "brute_force" and want:
            m = np.flatnonzero(match)
            dd = ((x[m] - q) ** 2).sum(1)
            exact = m[np.argsort(dd, kind="stable")[:10]]
            assert got == exact.tolist(), text
        checked += 1
    assert checked == 60


def test_bitmap_recall(filtered):
    col, x, attrs, dead = filtered
    rng = np.random.default_rng(5)
    hits = total = 0
    for _ in range(40):
        text = random_predicate(rng)
        q = rng.normal(size=x.shape[1]).astype(np.float32)
        exact = col.search(q, k=10, filter=text, strategy="brute_force").ids
        approx = col.search(q, k=10, filter=text, ef=128, strategy="bitmap").ids
        hits += len(set(exact) & set(approx))
        total += len(exact)
    assert hits / max(total, 1) >= 0.9


def test_planner_picks_by_selectivity(filtered):
    col = filtered[0]
    q = np.zeros(16, np.float32)
    assert col.search(q, k=5, filter="price >= 0").strategy.startswith("post_filter")
    assert col.search(q, k=5, filter="price == 7 AND cat == 'belts'").strategy == "brute_force"
    assert col.search(q, k=5, filter="cat == 'socks'").strategy == "bitmap"  # ~7% match


def test_filter_errors_are_invalid_argument(filtered):
    col = filtered[0]
    q = np.zeros(16, np.float32)
    for bad in ["price <", "nope == 1", "cat < 'a'"]:
        with pytest.raises(InvalidArgument):
            col.search(q, k=5, filter=bad)


def test_flat_collection_filters(tmp_path, rng):
    c = Collection.create(tmp_path / "flat", "flat", dim=4, overrides={"index": "flat"})
    x = rng.normal(size=(100, 4)).astype(np.float32)
    c.upsert([str(i) for i in range(100)], x, [{"even": i % 2 == 0} for i in range(100)])
    res = c.search(x[3], k=5, filter="even == true")
    assert res.strategy == "brute_force" and all(int(i) % 2 == 0 for i in res.ids)
    c.close()


def test_mask_cache_invalidated_by_writes(tmp_path, rng):
    c = Collection.create(tmp_path / "mc", "mc", dim=4)
    x = rng.normal(size=(50, 4)).astype(np.float32)
    c.upsert([str(i) for i in range(50)], x, [{"p": 1.0} for _ in range(50)])
    assert len(c.search(x[0], k=100, filter="p == 2", strategy="bitmap").ids) == 0
    c.upsert(["new"], x[:1], [{"p": 2.0}])
    assert c.search(x[0], k=100, filter="p == 2", strategy="bitmap").ids == ["new"]
    c.close()
