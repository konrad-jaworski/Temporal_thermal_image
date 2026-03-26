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
    Generate mask with:
    - circular defects
    - no overlap
    - spacing between defects
    - distance from borders
    """

    if seed is not None:
        np.random.seed(seed)

    mask = np.zeros((H, W), dtype=np.float32)
    occupied = np.zeros((H, W), dtype=np.uint8)

    Y, X = np.ogrid[:H, :W]

    # Balanced sampling
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

        # --- enforce boundary margin ---
        effective_r = r + spacing_margin

        cx = np.random.randint(border_margin + effective_r, W - border_margin - effective_r)
        cy = np.random.randint(border_margin + effective_r, H - border_margin - effective_r)

        # actual defect
        circle = (X - cx)**2 + (Y - cy)**2 <= r**2

        # forbidden region (expanded circle)
        forbidden = (X - cx)**2 + (Y - cy)**2 <= (r + spacing_margin)**2

        # check spacing
        if np.any(occupied[forbidden]):
            continue

        # place defect
        mask[circle] = depth
        occupied[forbidden] = 1  # reserve space

        placed += 1

    print(f"Placed {placed}/{N_circles} circles")

    return mask

def extract_bscan_widths(mask):
    widths = []

    for row in mask:
        inside = False
        start = 0

        for i, val in enumerate(row):
            if val > 0 and not inside:
                inside = True
                start = i

            elif val == 0 and inside:
                inside = False
                widths.append(i - start)

        if inside:
            widths.append(len(row) - start)

    return widths

def plot_dataset_histogram(masks, bins=50):
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

    print(f"Total B-scans: {len(all_widths)}")

def extract_defect_depths(mask):
    """
    Extract one depth value per defect (blob-based).
    """

    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    depths = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] > 0 and not visited[y, x]:

                stack = [(y, x)]
                depth_val = mask[y, x]

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
    Convert mask into defect list for simulation.

    Returns:
    --------
    defect_list : list of dict
    """

    H, W = mask.shape
    scale = sample_size / W  # meters per pixel

    visited = np.zeros_like(mask, dtype=bool)
    defects = []

    for y in range(H):
        for x in range(W):
            if mask[y, x] > 0 and not visited[y, x]:

                # --- flood fill to extract one defect ---
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

                # --- compute properties ---
                ys = coords[:, 0]
                xs = coords[:, 1]

                cy = ys.mean()
                cx = xs.mean()

                # radius estimate (max distance)
                r = np.sqrt((xs - cx)**2 + (ys - cy)**2).max()

                # depth (assume constant inside defect)
                depth_val = mask[ys[0], xs[0]]

                # --- convert to physical ---
                pos_x = cx * scale
                pos_y = cy * scale
                size = 2 * r * scale  # diameter in meters

                # reverse depth convention
                depth_percent = (1 - depth_val) * 100

                defects.append({
                    "pos_x": float(pos_x),
                    "pos_y": float(pos_y),
                    "size": float(size),
                    "depth": float(depth_percent)
                })

    return defects

def save_dataset_npz(filename, mask, defect_list):
    """
    Save mask + defect list into npz
    """

    # convert list of dicts → structured arrays
    pos_x = np.array([d["pos_x"] for d in defect_list])
    pos_y = np.array([d["pos_y"] for d in defect_list])
    size = np.array([d["size"] for d in defect_list])
    depth = np.array([d["depth"] for d in defect_list])

    np.savez(
        filename,
        mask=mask,
        pos_x=pos_x,
        pos_y=pos_y,
        size=size,
        depth=depth
    )

def load_dataset_npz(filename):
    data = np.load(filename)

    mask = data["mask"]

    defect_list = []
    for i in range(len(data["pos_x"])):
        defect_list.append({
            "pos_x": float(data["pos_x"][i]),
            "pos_y": float(data["pos_y"][i]),
            "size": float(data["size"][i]),
            "depth": float(data["depth"][i]),
        })

    return mask, defect_list