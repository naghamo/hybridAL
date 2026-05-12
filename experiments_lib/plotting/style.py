"""ACL/EMNLP figure-formatting helper for matplotlib.

Call `apply_acl_style()` once at the start of any plotting script, then use
the size constants below for `figsize`. The style sets:

  * Serif fonts (Times Roman, with DejaVu Serif fallback) so figure text
    matches the paper body (Times).
  * Embedded TrueType fonts in the PDF (`pdf.fonttype = 42`) — required by
    *ACL spec.
  * Sub-10pt tick / axis-label sizes that stay legible at column width
    on an A4 page.
  * 300 dpi raster output for crisp PNG at column width.

A4 page geometry from the ACL spec (cm → inches at 1 cm = 0.3937 in):
  column width            : 7.7  cm  ≈ 3.03 in
  column gap              : 0.6  cm  ≈ 0.24 in
  text width (2 cols+gap) : 16.0 cm  ≈ 6.30 in
  page text height        : 24.7 cm  ≈ 9.72 in

Size presets (W × H in inches):
  SINGLE_COL          — 3.03 × 2.27   (single column, ~4:3)
  SINGLE_COL_TALL     — 3.03 × 3.03   (single column, square)
  SINGLE_COL_STACKED_2— 3.03 × 4.55   (single column, two stacked panels)
  DOUBLE_COL          — 6.30 × 2.84   (full text width, modest height)
  DOUBLE_COL_TALL     — 6.30 × 3.78
  DOUBLE_COL_GRID     — 6.30 × 4.50   (panel grids)
  FULL_PAGE           — 6.30 × 7.50   (multi-row figure, near full page)

Grayscale readability is strongly encouraged by the ACL spec — even when
colour is used, also vary linestyle/marker so figures stay distinguishable
in B&W print.
"""
from __future__ import annotations
import matplotlib


# -- A4 column geometry ------------------------------------------------------
CM_PER_IN  = 2.54
COL_W_IN   = 7.7  / CM_PER_IN          # ≈ 3.03
COL_GAP_IN = 0.6  / CM_PER_IN          # ≈ 0.24
TEXT_W_IN  = 2 * COL_W_IN + COL_GAP_IN  # ≈ 6.30
PAGE_H_IN  = 24.7 / CM_PER_IN          # ≈ 9.72

# -- Figure-size presets -----------------------------------------------------
SINGLE_COL           = (COL_W_IN,  COL_W_IN * 0.75)   # ≈ (3.03, 2.27)
SINGLE_COL_TALL      = (COL_W_IN,  COL_W_IN)          # ≈ (3.03, 3.03)
SINGLE_COL_STACKED_2 = (COL_W_IN,  COL_W_IN * 1.5)    # ≈ (3.03, 4.55)
DOUBLE_COL           = (TEXT_W_IN, TEXT_W_IN * 0.45)  # ≈ (6.30, 2.84)
DOUBLE_COL_TALL      = (TEXT_W_IN, TEXT_W_IN * 0.60)  # ≈ (6.30, 3.78)
DOUBLE_COL_GRID      = (TEXT_W_IN, TEXT_W_IN * 0.72)  # ≈ (6.30, 4.50)
FULL_PAGE            = (TEXT_W_IN, PAGE_H_IN * 0.77)  # ≈ (6.30, 7.50)


def apply_acl_style(scale: float = 1.0) -> None:
    """Apply ACL/EMNLP-conformant rcParams globally for the current run.

    `scale` multiplies every font size (and matching tick / line widths)
    proportionally. Use scale=1.0 for default ACL sizes (8 pt ticks /
    10 pt labels), scale=1.25 for "side-by-side" plots that sit at half
    the text width and benefit from slightly larger fonts in print.
    """
    matplotlib.rcParams.update({
        # Times-family serif so figure text matches the paper body.
        "font.family":   "serif",
        "font.serif":    ["Times New Roman", "Times", "Liberation Serif",
                          "DejaVu Serif"],
        "mathtext.fontset": "stix",      # math compatible with Times
        # Sizes — tuned for column width on an A4 page; multiplied by `scale`.
        "font.size":              9 * scale,
        "axes.titlesize":        10 * scale,
        "axes.labelsize":        10 * scale,
        "xtick.labelsize":        8 * scale,
        "ytick.labelsize":        8 * scale,
        "legend.fontsize":        8 * scale,
        "legend.title_fontsize":  8 * scale,
        "figure.titlesize":      11 * scale,
        # Embed real TrueType fonts in the PDF (the spec requires "all
        # fonts embedded"; type 42 = TT, type 3 = Type 3 PS).
        "pdf.fonttype":          42,
        "ps.fonttype":           42,
        # Save defaults — 300 dpi PNG, tight bbox.
        "savefig.dpi":          300,
        "savefig.bbox":         "tight",
        # Lines / markers / grid scale together so they stay proportionate.
        "lines.linewidth":      1.0 * scale,
        "lines.markersize":     3.5 * scale,
        "axes.linewidth":       0.7,
        "grid.alpha":           0.30,
        "grid.linewidth":       0.4,
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.major.width":    0.6,
        "ytick.major.width":    0.6,
    })
