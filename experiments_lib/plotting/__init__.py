"""Paper-figure plotting package.

Each module renders one figure (or one figure family) for the paper.
Common machinery — ACL/EMNLP style, helpers for parsing run names, and
aggregators across (dataset, seed) cells — lives in `style` and `utils`.
"""
from .style import (
    apply_acl_style,
    SINGLE_COL,
    SINGLE_COL_TALL,
    SINGLE_COL_STACKED_2,
    DOUBLE_COL,
    DOUBLE_COL_TALL,
    DOUBLE_COL_GRID,
    FULL_PAGE,
)
