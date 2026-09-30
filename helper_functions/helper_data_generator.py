"""
Helpers for designing the synthetic defect layouts used to build the
simulated training data.

A layout is a 2D depth mask [H, W] of the specimen surface: 0 where the
material is sound, and the normalised defect depth inside each defect. All
defects are flat-bottom circles with one constant depth each. The workflow
(see data_generation_scheme.ipynb) is:

  1. generate_mask_with_spacing   - place random circular defects on the mask,
  2. plot_dataset_histogram,
     plot_depth_histogram         - check the width / depth distributions,
  3. mask_to_defect_list          - turn each defect into position, diameter
                                    and depth in physical units, which is
                                    the input for the FEM simulation.

The same mask is later used as the ground truth: every row of it is the
depth target of the corresponding B-scan.
"""

import numpy as np
from matplotlib import pyplot as plt


def generate_mask_with_spacing(
    H,
    W,
    radii,
    depths,
    N_circles=20,
    border_margin=20,
    spacing_margin=10,
    seed=None
):
    """
    Places non-overlapping circular defects at random positions on a mask.

    Every defect is kept at least `spacing_margin` pixels away from the other
    defects, and away from the specimen edges, so that the thermal response
    of one defect is not disturbed by a neighbour or by heat loss at the
    edge.

    Parameters
    ----------
    H, W : int
        Mask size in pixels.
    radii : array-like of int
        Allowed defect radii in pixels.
    depths : array-like of float
        Allowed normalised defect depths; one is drawn uniformly per defect.
    N_circles : int
        Number of defects to place.
    border_margin : int
        Extra distance in pixels between the defects and the mask border
        (added on top of spacing_margin).
    spacing_margin : int
        Minimum gap in pixels between the edges of two defects.
    seed : int or None
        Seed for numpy's global random generator, for reproducible layouts.

    Returns
    -------
    mask : np.ndarray, [H, W], float32
        0 outside the defects, the defect depth inside them.

    Notes
    -----
    Placement is done by rejection sampling: a random defect is proposed and
    discarded if it is too close to one already placed. After 10000 attempts
    the function stops, so on a crowded mask fewer than N_circles defects can
    be returned; the printed "Placed x/N" shows how many were placed.
    """

    if seed is not None:
        np.random.seed(seed)

    # mask     : the depth map that is returned.
    # occupied : every pixel within spacing_margin of a placed defect; a new
    #            defect may not overlap any of these pixels.
    mask = np.zeros((H, W), dtype=np.float32)
    occupied = np.zeros((H, W), dtype=np.uint8)

    # Open grids of pixel coordinates (column vector Y, row vector X); they
    # broadcast to [H, W] when the circle equations are evaluated below.
    Y, X = np.ogrid[:H, :W]

    # Radius sampling weights proportional to 1 / diameter. A circle of
    # radius r is crossed by about 2r rows, i.e. appears in about 2r B-scans.
    # With these weights large defects are drawn less often, so every radius
    # contributes roughly the same number of B-scans to the training set.
    radii = np.array(radii)
    probs = 1.0 / (2 * radii)
    probs /= probs.sum()

    placed = 0
    attempts = 0
    max_attempts = 10000

    while placed < N_circles and attempts < max_attempts:
        attempts += 1

        r = np.random.choice(radii, p=probs)
        depth = np.random.choice(depths)

        # The centre is drawn so that the circle plus its spacing ring lies
        # at least border_margin pixels inside the mask.
        effective_r = r + spacing_margin

        cx = np.random.randint(border_margin + effective_r, W - border_margin - effective_r)
        cy = np.random.randint(border_margin + effective_r, H - border_margin - effective_r)

        # Pixels of the defect itself.
        circle = (X - cx)**2 + (Y - cy)**2 <= r**2

        # Defect enlarged by spacing_margin: the area that must be free.
        forbidden = (X - cx)**2 + (Y - cy)**2 <= (r + spacing_margin)**2

        # Reject the proposal if the enlarged circle touches the enlarged
        # circle of any defect placed earlier.
        if np.any(occupied[forbidden]):
            continue

        mask[circle] = depth
        occupied[forbidden] = 1

        placed += 1

    print(f"Placed {placed}/{N_circles} circles")

    return mask

def extract_bscan_widths(mask):
    """
    Widths (in pixels) of every defect cross-section seen in the rows of a
    mask.

    Each row of the mask is the target of one B-scan. A defect crossed by a
    row appears as a run of consecutive non-zero pixels; the length of that
    run is the width of the defect in that B-scan. Rows near the top or
    bottom of a circle give short widths, rows through its centre give the
    full diameter.

    Parameters
    ----------
    mask : np.ndarray, [H, W]

    Returns
    -------
    list of int
        One width per defect run, over all rows.
    """
    widths = []

    for row in mask:
        # Scan the row from left to right and record where every run of
        # non-zero values starts and ends.
        inside = False
        start = 0

        for i, val in enumerate(row):
            if val > 0 and not inside:
                inside = True
                start = i

            elif val == 0 and inside:
                inside = False
                widths.append(i - start)

        # A run that reaches the end of the row is closed here.
        if inside:
            widths.append(len(row) - start)

    return widths

def plot_dataset_histogram(masks, bins=50):
    """
    Histogram of defect widths in the B-scans of a whole set of masks.

    Used to check that the chosen radii and sampling weights give a
    reasonably even spread of defect sizes as seen by the network.

    Parameters
    ----------
    masks : iterable of np.ndarray, each [H, W]
    bins : int
        Number of histogram bins.
    """
    all_widths = []

    for mask in masks:
        w = extract_bscan_widths(mask)
        all_widths.extend(w)

    if len(all_widths) == 0:
        print("WARNING: No widths found!")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(all_widths, bins=bins, edgecolor='black')
    plt.title("B-scan Width Distribution (Dataset Level)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    # Number of defect runs, i.e. B-scan rows crossing a defect (a row
    # crossing two defects is counted twice).
    print(f"Total B-scans: {len(all_widths)}")

def extract_defect_depths(mask):
    """
    Returns one depth value per defect in a mask.

    Defects are found as connected groups of non-zero pixels using a flood
    fill with 8-connectivity (diagonal neighbours count as connected). Since
    a defect has a single constant depth, the value of its first pixel is
    taken as the depth of the whole defect.

    Parameters
    ----------
    mask : np.ndarray, [H, W]

    Returns
    -------
    list of float
        One depth per defect.
    """

    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    depths = []

    for y in range(H):
        for x in range(W):
            # An unvisited defect pixel is the first pixel of a new defect.
            if mask[y, x] > 0 and not visited[y, x]:

                stack = [(y, x)]
                depth_val = mask[y, x]

                # Iterative flood fill (an explicit stack instead of
                # recursion, which would exceed Python's recursion limit on
                # large defects). All pixels of this defect are marked as
                # visited so they are not counted again.
                while stack:
                    cy, cx = stack.pop()

                    if (cy < 0 or cy >= H or cx < 0 or cx >= W):
                        continue

                    if visited[cy, cx] or mask[cy, cx] == 0:
                        continue

                    visited[cy, cx] = True

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            stack.append((cy + dy, cx + dx))

                depths.append(depth_val)

    return depths

def plot_depth_histogram(masks, bins=20):
    """
    Histogram of defect depths over a whole set of masks (one entry per
    defect), used to check that all depths are represented evenly.

    Parameters
    ----------
    masks : iterable of np.ndarray, each [H, W]
    bins : int
        Number of histogram bins.
    """
    all_depths = []

    for mask in masks:
        d = extract_defect_depths(mask)
        all_depths.extend(d)

    if len(all_depths) == 0:
        print("WARNING: No depths found!")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(all_depths, bins=bins, edgecolor='black')
    plt.title("Defect Depth Distribution (Dataset Level)")
    plt.xlabel("Depth (normalized 0–1)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

    print(f"Total defects: {len(all_depths)}")


def mask_to_defect_list(mask, sample_size=0.1):
    """
    Converts a mask into the list of defects passed to the FEM simulation.

    Each defect is found with the same 8-connected flood fill as in
    extract_defect_depths. Its centre, diameter and depth are measured in
    pixels and converted to metres using the specimen size.

    Parameters
    ----------
    mask : np.ndarray, [H, W]
        Mask from generate_mask_with_spacing.
    sample_size : float
        Physical width of the specimen in metres (default 0.1 m = 100 mm).
        The mask is assumed to cover the whole specimen with square pixels,
        so the same pixel size is used for x and y.

    Returns
    -------
    defect_list : list of dict
        One dict per defect with the keys
          pos_x, pos_y : centre position in metres (rounded to 1 mm),
          size         : diameter in metres (rounded to 1 mm),
          depth        : depth in the convention of the simulation,
                         (1 - mask depth) * 100, e.g. 0.8 in the mask -> 20.
    """

    H, W = mask.shape
    scale = sample_size / W  # meters per pixel

    visited = np.zeros_like(mask, dtype=bool)
    defects = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] > 0 and not visited[y, x]:

                # Collect the (row, column) coordinates of every pixel of
                # this defect.
                stack = [(y, x)]
                coords = []

                while stack:
                    cy, cx = stack.pop()

                    if (cy < 0 or cy >= H or cx < 0 or cx >= W):
                        continue

                    if visited[cy, cx] or mask[cy, cx] == 0:
                        continue

                    visited[cy, cx] = True
                    coords.append((cy, cx))

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            stack.append((cy + dy, cx + dx))

                coords = np.array(coords)

                ys = coords[:, 0]
                xs = coords[:, 1]

                # Centre of the defect = mean position of its pixels.
                cy = ys.mean()
                cx = xs.mean()

                # Radius = distance from the centre to the farthest pixel of
                # the defect.
                r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max()

                # All pixels of a defect have the same depth, so any pixel
                # can be used.
                depth_val = mask[ys[0], xs[0]]

                # Pixels -> metres. The simulation takes the diameter, not
                # the radius.
                pos_x = round(cx * scale, 3)
                pos_y = round(cy * scale, 3)
                size  = round(2 * r * scale, 3)

                # The simulation describes depth with the complementary
                # value, as an integer percentage: (1 - depth) * 100.
                depth_percent = int(round((1 - depth_val) * 100))

                defects.append({
                    "pos_x": float(pos_x),
                    "pos_y": float(pos_y),
                    "size": float(size),
                    "depth": float(depth_percent)
                })

    return defects