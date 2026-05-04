"""Aggregation utilities — mean ± std cells, Welch's t-test, formatting."""
import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from scipy.stats import ttest_ind
except Exception:  # pragma: no cover
    ttest_ind = None  # type: ignore


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
    """Welch's two-sample t-test p-value (returns None if scipy missing or
    either sample has < 2 finite points)."""
    if ttest_ind is None:
        return None
    av, bv = _finite(a), _finite(b)
    if len(av) < 2 or len(bv) < 2:
        return None
    try:
        return float(ttest_ind(av, bv, equal_var=False)[1])
    except Exception:
        return None


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
