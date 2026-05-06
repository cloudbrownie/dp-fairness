from __future__ import annotations

import matplotlib as mpl


# Print-scale defaults: figures land in a paper at ~3-7 inches, where
# matplotlib's 10pt default reads as ~7pt and is uncomfortable. Bump to
# 11/12pt; titles to LaTeX captions; tight bbox so figures don't carry
# unnecessary whitespace.
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 14,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 300,
})
