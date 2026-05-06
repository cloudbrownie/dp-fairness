from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap


#PALETTE: list[str] = [
#    "#003d5c",
#    "#006177",
#    "#008575",
#    "#00a657",
#    "#8dbf23",
#    "#ffc800",
#]


PALETTE: list[str] = [
    "#003d5c",
    "#594e90",
    "#bc4c96",
    "#ff5f66",
    "#ffa600",
]

def cmap(name: str = "tealyellow") -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, PALETTE, N=256)


def categorical(n: int) -> list[str]:
    # Evenly sample n discrete colors from the palette range.
    if n <= len(PALETTE):
        if n == 1:
            return [PALETTE[0]]
        idxs = [round(i * (len(PALETTE) - 1) / (n - 1)) for i in range(n)]
        return [PALETTE[i] for i in idxs]
    cm = cmap()
    return [cm(i / (n - 1)) for i in range(n)]
