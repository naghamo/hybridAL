"""Signal-related helpers (one source of truth for the to_switching_value
transform and the calibration-normalizer JSON loader)."""
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

from . import constants as C
from .csv_io import load_json, read_csv, to_float, to_int


SUPPORTED_SIGNALS = (
    "delta_f1", "delta_loss", "delta_accuracy",
    "gradient_norm", "l2_weight_distance", "cka",
    "delta_spectral_alpha", "delta_nc",
)


def to_switching_value(name: str, raw: Optional[float]) -> Optional[float]:
    """Map a raw per-round signal value into 'small = stable' form.

    - cka: returns 1 - raw  (raw CKA is high at convergence).
    - everything else: returns raw unchanged (already small at stability).
    """
    if raw is None:
        return None
    try:
        if not math.isfinite(raw):
            return None
    except TypeError:
        return None
    if name == "cka":
        return 1.0 - raw
    return raw


def load_normalizers(path: str = C.CALIBRATION_NORMALIZERS_PATH) -> Dict[str, float]:
    """Load `_calibration_normalizers.json` and return the
    {signal_name: normalizer_value} dict."""
    obj = load_json(Path(path))
    return obj["normalizers"]


def compute_max_2_5_from_calibration_json(calib_results_json: Path,
                                          signals: List[str] = list(SUPPORTED_SIGNALS)
                                         ) -> Dict[str, Optional[float]]:
    """Compute the `max-of-rounds-2-5` normalizer per signal from a single
    calibration-style results JSON (e.g. one produced by `_calibrate.py` or
    one of the in-flight signal-ablation Retrain runs that has
    `log_all_signals=True`).

    Returns a {signal: max_value or None} dict.
    """
    obj = load_json(calib_results_json)
    rounds = obj.get("round_val_stats", [])
    out: Dict[str, Optional[float]] = {}
    for sig in signals:
        vals: List[float] = []
        for ridx, r in enumerate(rounds, 1):
            if not (2 <= ridx <= 5):
                continue
            sig_dict = r.get("signals") or {}
            raw = sig_dict.get(sig)
            sw = to_switching_value(sig, raw)
            if sw is None:
                continue
            try:
                if not math.isfinite(sw) or sw <= 0:
                    continue
            except TypeError:
                continue
            vals.append(float(sw))
        out[sig] = max(vals) if vals else None
    return out


def trajectory_normalized_per_round(per_round_csv: Path,
                                    signal: str,
                                    normalizer: float
                                   ) -> Dict[int, List[float]]:
    """Read a `_per_round.csv` (signal-ablation aggregator output) and
    return a dict {round: [normalized_value across (dataset × seed)]} for
    the rows where the active signal == `signal`.
    """
    rows = read_csv(per_round_csv)
    by_round: Dict[int, List[float]] = defaultdict(list)
    sig_col = f"sig_{signal}"
    for r in rows:
        if r.get("signal") != signal:
            continue
        rd = to_int(r.get("round"))
        raw = to_float(r.get(sig_col))
        sw = to_switching_value(signal, raw)
        if rd is None or sw is None:
            continue
        try:
            if not math.isfinite(sw) or normalizer is None or normalizer <= 0:
                continue
        except TypeError:
            continue
        by_round[rd].append(sw / normalizer)
    return dict(by_round)


def derive_epsilon_grid(per_round_csv: Path,
                        signal: str,
                        normalizer: float,
                        target_rounds: List[int] = (5, 10, 18, 25),
                        k: int = 3,
                       ) -> Dict[str, Any]:
    """Auto-derive 4 ε candidates that target switching at the given
    rounds (mean trajectory), plus a 'no-switch' / very-late one.

    Algorithm:
      1. Compute the mean normalized trajectory across (dataset × seed).
      2. For each target round t, pick ε as the smallest value such that
         the mean trajectory has been < ε for at least k consecutive rounds
         ending at t. Fall back to the simple "mean(round t)" if no such
         streak exists.
      3. Print rationale; return the 4 ε values.

    Returns: dict with keys
      'trajectory_means' (Dict[int, float]),
      'epsilon_grid'     (List[float]),
      'targets'          (List[int]),
      'rationale'        (str),
    """
    by_round = trajectory_normalized_per_round(per_round_csv, signal, normalizer)
    if not by_round:
        return {"trajectory_means": {}, "epsilon_grid": [], "targets": list(target_rounds),
                "rationale": "no data found for signal in CSV"}

    means: Dict[int, float] = {rd: sum(vs)/len(vs) for rd, vs in sorted(by_round.items())}

    chosen: List[float] = []
    rationale_lines: List[str] = []
    sorted_rounds = sorted(means)
    max_round = max(sorted_rounds)

    for target in target_rounds:
        cand: Optional[float] = None
        # Search ε downward; pick smallest ε where some streak of length k
        # ending at or before `target` exists in the mean trajectory.
        # Use a discrete grid of candidate ε values from the mean trajectory.
        candidate_eps = sorted({means[rd] for rd in sorted_rounds if rd <= target}, reverse=True)
        for eps in candidate_eps:
            streak = 0
            switch_round = None
            for rd in sorted_rounds:
                if means[rd] < eps:
                    streak += 1
                    if streak >= k:
                        switch_round = rd
                        break
                else:
                    streak = 0
            if switch_round is not None and switch_round <= target:
                cand = eps
                rationale_lines.append(
                    f"  target round={target} → ε={eps:.4f} "
                    f"(mean trajectory drops to {eps:.4f} for {k} consecutive rounds by round {switch_round})"
                )
                break
        if cand is None:
            # Fallback: use the value at the target round as a "loose" ε.
            cand = means.get(target, max(means.values()))
            rationale_lines.append(
                f"  target round={target} → ε={cand:.4f} (no k={k} streak ≤ target; using mean({target}))"
            )
        chosen.append(round(cand, 4))

    rationale = "\n".join(rationale_lines)
    return {
        "trajectory_means": means,
        "epsilon_grid": chosen,
        "targets": list(target_rounds),
        "rationale": rationale,
    }
