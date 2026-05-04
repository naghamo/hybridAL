"""Tiny shared base for runner CLIs. Each runner imports `dispatch(...)`
to wire argparse → make_plan → run_plan_on_gpu."""
import argparse
from typing import Any, Callable, Dict

from ..shared.runner import add_common_runner_args, run_plan_on_gpu


def make_runner_parser(description: str,
                       extra: Callable[[argparse.ArgumentParser], None] = lambda p: None,
                       ) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    add_common_runner_args(p)
    extra(p)
    return p


def dispatch(plan, args, make_cfg) -> None:
    """Standard dispatch: build the plan → run on gpu (or print on dry-run)."""
    run_plan_on_gpu(plan, args, make_cfg)
