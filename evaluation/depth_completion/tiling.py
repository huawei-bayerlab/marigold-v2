"""Tile geometry and the cross-fade merge for the tiled depth-completion row.

Each tile is completed on its own, against the sparse points inside it, with its own adapter and
its own log-affine, so every tile is metric and the merge only has to hide small disagreements in
the overlap. Tiles are cut from the RGB and the sparse map alone.
"""

from __future__ import annotations

import numpy as np

# A tile with fewer sparse points than this is not completed; its pixels come from the untiled
# prediction instead.
MIN_TILE_PTS = 8


def parse_tiles(spec):
    """'2x2' -> (2, 2). '1x1' means untiled."""
    rows, cols = (int(v) for v in spec.lower().split("x"))
    return rows, cols


def tile_boxes(h, w, rows, cols, overlap):
    """Tile boxes on the native grid, each grown by `overlap` of its size on its inner sides."""
    boxes = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * h // rows, (r + 1) * h // rows
            x0, x1 = c * w // cols, (c + 1) * w // cols
            oy, ox = int(overlap * (y1 - y0)), int(overlap * (x1 - x0))
            boxes.append(
                (
                    max(0, y0 - oy * (r > 0)),
                    min(h, y1 + oy * (r < rows - 1)),
                    max(0, x0 - ox * (c > 0)),
                    min(w, x1 + ox * (c < cols - 1)),
                )
            )
    return boxes


def feather(hh, ww, box, h, w, overlap_px):
    """Linear ramp into the overlap so tiles cross-fade instead of showing a seam.

    A side on the image border keeps weight 1: there is nothing to blend with there.
    """
    y0, y1, x0, x1 = box
    wy = np.ones(hh, np.float32)
    wx = np.ones(ww, np.float32)
    n = max(1, overlap_px)
    ramp = np.linspace(0, 1, n + 2)[1:-1]
    if y0 > 0:
        wy[:n] = ramp[: min(n, hh)]
    if y1 < h:
        wy[-n:] = ramp[::-1][: min(n, hh)]
    if x0 > 0:
        wx[:n] = ramp[: min(n, ww)]
    if x1 < w:
        wx[-n:] = ramp[::-1][: min(n, ww)]
    return np.outer(wy, wx).astype(np.float32)


def overlap_px(box, rows, cols, overlap):
    """Ramp width for one box, in pixels of its own short inner side."""
    y0, y1, x0, x1 = box
    return int(
        overlap
        * min(
            (y1 - y0) / rows if rows > 1 else 1e9,
            (x1 - x0) / cols if cols > 1 else 1e9,
        )
    )


def run_tiled(predict, rgb, sparse, hw, tiles, overlap, fallback=None, verbose=False):
    """Run `predict(rgb_tile, sparse_tile) -> depth_tile` per tile and cross-fade the results.

    `predict` receives native-resolution crops and returns a metric depth map of the same shape,
    so whatever resizing a method does internally applies to the tile. Pixels that no completed
    tile covers are filled from `fallback() -> depth at hw`, the untiled prediction; without a
    fallback they raise. Returns (depth at `hw`, per-tile bookkeeping).
    """
    rows, cols = parse_tiles(tiles)
    h, w = hw
    acc = np.zeros((h, w), np.float64)
    wsum = np.zeros((h, w), np.float64)
    per_tile = []
    for i, box in enumerate(tile_boxes(h, w, rows, cols, overlap)):
        y0, y1, x0, x1 = box
        rgb_t, sparse_t = rgb[y0:y1, x0:x1], sparse[y0:y1, x0:x1]
        n_pts = int((sparse_t > 0).sum())
        if n_pts < MIN_TILE_PTS:
            if verbose:
                print(f"    tile {i}: only {n_pts} sparse points, skipped", flush=True)
            per_tile.append({"tile": i, "n_sparse": n_pts, "skipped": True})
            continue
        depth_t = predict(rgb_t, sparse_t)
        f = feather(
            y1 - y0, x1 - x0, box, h, w, max(1, overlap_px(box, rows, cols, overlap))
        )
        acc[y0:y1, x0:x1] += depth_t.astype(np.float64) * f
        wsum[y0:y1, x0:x1] += f
        per_tile.append({"tile": i, "n_sparse": n_pts, "hw": [y1 - y0, x1 - x0]})
    uncovered = wsum <= 0
    if uncovered.any():
        n_uncovered = int(uncovered.sum())
        if fallback is None:
            raise RuntimeError(
                f"{n_uncovered} pixels are covered by no completed tile; "
                "use fewer tiles or run untiled"
            )
        if verbose:
            print(
                f"    {n_uncovered} uncovered pixels filled from the untiled prediction",
                flush=True,
            )
        acc[uncovered] = fallback()[uncovered]
        wsum[uncovered] = 1.0
    return (acc / wsum).astype(np.float32), per_tile
