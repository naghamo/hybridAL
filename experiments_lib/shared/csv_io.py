"""CSV / JSON I/O helpers shared across aggregators."""
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def to_float(x: Any) -> Optional[float]:
    """Best-effort float conversion. Returns None on NaN / inf / unparsable."""
    if x is None or x == "" or x == "None":
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (ValueError, TypeError):
        return None


def to_int(x: Any) -> Optional[int]:
    f = to_float(x)
    return int(f) if f is not None else None


def write_csv(rows: List[Dict[str, Any]], path: Path, fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def dump_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
