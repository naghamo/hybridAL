"""ExperimentPlan dataclass + cartesian product helpers."""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentPlan:
    """A flat list of plan entries to be executed, one per (config) tuple.

    Each entry is a free-form dict of axis values (e.g.
    `{"signal": "delta_f1", "dataset": "imdb", "seed": 42}`). The runner's
    `make_cfg(entry)` function turns each entry into an ExperimentConfig.
    """
    name: str
    save_root: Path
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def make_plan_from_grid(name: str,
                        save_root: Path,
                        grid: Dict[str, List[Any]],
                        prepend: Optional[List[Dict[str, Any]]] = None,
                        ) -> ExperimentPlan:
    """Cartesian product over `grid`. Insertion order of `grid.keys()` is
    preserved, so the output is deterministic.

    `prepend` lets a caller stick a few items in front (e.g. baseline
    configurations that must run first).
    """
    keys = list(grid.keys())
    entries: List[Dict[str, Any]] = list(prepend) if prepend else []
    for combo in product(*[grid[k] for k in keys]):
        entries.append({k: v for k, v in zip(keys, combo)})
    return ExperimentPlan(name=name, save_root=Path(save_root), entries=entries)


def already_done(experiment_name: str, save_root: Path) -> bool:
    """True if the run dir exists and has a `results_*.json` file."""
    run_dir = Path(save_root) / experiment_name
    if not run_dir.is_dir():
        return False
    return any(run_dir.glob("results_*.json"))


def split_for_gpu(plan: ExperimentPlan,
                  gpu_id: int,
                  gpu_datasets: Dict[int, List[str]],
                  mode: str = "dataset",
                  num_gpus: int = 2,
                  ) -> List[Dict[str, Any]]:
    """Return the slice of `plan.entries` that belongs to `gpu_id`.

    Modes:
      "dataset"  — assign each entry to the GPU whose `gpu_datasets`
                   list contains its dataset. Best for experiments whose
                   datasets split cleanly across GPUs (Exp 3, 4 here).
      "index"    — round-robin by entry index: entry i goes to GPU
                   (i mod num_gpus). Use for experiments whose datasets
                   don't split evenly (Exp 5/6/7); guarantees a balanced
                   slice regardless of the dataset axis.

    Entries without a `dataset` key in "dataset" mode default to GPU 0.
    """
    if mode == "index":
        return [e for i, e in enumerate(plan.entries) if i % num_gpus == gpu_id]
    if mode != "dataset":
        raise ValueError(f"unknown split mode {mode!r}; expected 'dataset' or 'index'")
    if gpu_id not in gpu_datasets:
        raise ValueError(f"unknown gpu_id={gpu_id}")
    allowed = set(gpu_datasets[gpu_id])
    out: List[Dict[str, Any]] = []
    for e in plan.entries:
        ds = e.get("dataset")
        if ds is None and gpu_id == 0:
            out.append(e)
        elif ds in allowed:
            out.append(e)
    return out
