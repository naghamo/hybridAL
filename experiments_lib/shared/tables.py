"""Aggregation utilities — mean ± std cells, paired t-test, formatting."""
import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from scipy.stats import ttest_ind, ttest_rel
except Exception:
    ttest_ind = None
    ttest_rel = None


def _finite(values: Iterable) -> List[float]:
    out = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
            if math.isnan(f):
                continue
            out.append(f)
        except (TypeError, ValueError):
            continue
    return out


def cell(values: Iterable, fmt: str = "{:.4f}") -> str:
    """Mean ± std as a string, or '—' if empty."""
    vs = _finite(values)
    if not vs:
        return "—"
    if len(vs) == 1:
        return fmt.format(vs[0])
    return f"{fmt.format(mean(vs))}±{fmt.format(stdev(vs))}"


def safe_mean(values: Iterable) -> Optional[float]:
    vs = _finite(values)
    return mean(vs) if vs else None


def welch_p(a: Iterable, b: Iterable) -> Optional[float]:
    """Welch's two-sample t-test p-value (kept for back-compat; new code
    should use `paired_p` because the seed-matched design is paired,
    not independent-samples). Returns None if scipy missing or either
    sample has < 2 finite points."""
    if ttest_ind is None:
        return None
    av, bv = _finite(a), _finite(b)
    if len(av) < 2 or len(bv) < 2:
        return None
    try:
        return float(ttest_ind(av, bv, equal_var=False)[1])
    except Exception:
        return None


def paired_p(a: Sequence, b: Sequence,
             alternative: str = "two-sided") -> Optional[float]:
    """Paired two-sided t-test p-value across already-aligned samples.

    Inputs `a` and `b` must be ordered so that `a[i]` and `b[i]` are
    matched observations (e.g., same (backbone, dataset, seed) cell).
    Returns None if scipy missing, lengths differ, or any pair has a
    NaN/None on either side. Use `alternative='less'` or `'greater'`
    for directional tests."""
    if ttest_rel is None:
        return None
    if len(a) != len(b):
        return None
    av, bv = [], []
    for x, y in zip(a, b):
        if x is None or y is None:
            return None
        try:
            xf, yf = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if math.isnan(xf) or math.isnan(yf):
            return None
        av.append(xf); bv.append(yf)
    if len(av) < 2:
        return None
    try:
        return float(ttest_rel(av, bv, alternative=alternative)[1])
    except Exception:
        return None


def paired_arrays(rows: Iterable, key_fn, value_fn,
                  group_a, group_b) -> Tuple[List[float], List[float]]:
    """Build paired arrays (a, b) by matching rows of group_a and
    group_b on key_fn (the pairing key, e.g., (data, seed)). Returns
    two parallel lists ordered by sorted key; pairs missing on either
    side are dropped."""
    av_map: Dict = {}
    bv_map: Dict = {}
    for r in rows:
        if group_a(r):
            av_map[key_fn(r)] = value_fn(r)
        if group_b(r):
            bv_map[key_fn(r)] = value_fn(r)
    keys = sorted(set(av_map) & set(bv_map))
    return ([av_map[k] for k in keys], [bv_map[k] for k in keys])


def stars_for_p(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def group_by(rows, *keys) -> Dict[Tuple, List[Dict]]:
    """Group `rows` (list of dicts) by tuple(rows[k] for k in keys)."""
    out: Dict[Tuple, List[Dict]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out
