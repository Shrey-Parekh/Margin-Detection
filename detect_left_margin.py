import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

_reader = None


BAND_OUTWARD_FACTOR = 2.5
BAND_INWARD_FACTOR = 1.5

# Row bucket for de-duplication, in multiples of median word height.
ROW_FACTOR = 0.6

# Minimum measurements needed to form three sections.
MIN_MEASUREMENTS = 3


def get_reader():
    """Shared EasyOCR reader. Constructing one is expensive, so reuse it."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=True)
    return _reader


def remove_outliers(data):
    # No data means no measurement -- return empty and let the caller mark
    # the section undetermined. Never substitute a placeholder value, which
    # would be indistinguishable from a real reading downstream.
    if not data:
        return []
    median = statistics.median(data)
    mad = statistics.median([abs(x - median) for x in data])
    threshold = 2.35 * mad
    filtered_data = [x for x in data if abs(x - median) <= threshold]
    # If filtering removed everything we still have real readings, so fall
    # back to the unfiltered values rather than inventing one.
    return filtered_data if filtered_data else list(data)


def split_sections(items):
    """Cut a y-sorted list into three equal-COUNT sections.

    Splitting by rank rather than by pixel height is what guarantees no section
    is ever empty: with n >= 3 every section gets at least one measurement. The
    old pixel-thirds split drew its boundaries across the full text span, so a
    stray detection near the page bottom could push a section below where any
    data actually was.
    """
    n = len(items)
    i1, i2 = n // 3, 2 * n // 3
    return items[:i1], items[i1:i2], items[i2:]


def analyze_left_margin(image_path, results=None, image=None,
                        visualize=False, verbose=False, mirror=False):
    """Measure margin offsets for one image.

    With mirror=True the word coordinates are reflected about the page's
    vertical centre, so the RIGHT margin is measured by this exact same code.
    Only the coordinates are flipped -- the image is never mirrored, because
    EasyOCR is trained on normal text.

    Returns a dict of per-section offsets, or None if fewer than
    MIN_MEASUREMENTS lines could be measured.
    """
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")

    if results is None:
        results = get_reader().readtext(image)

    if not results:
        if verbose:
            print("No text detected in the image.")
        return None

    height, width = image.shape[:2]

    # Build the word list, optionally reflected so the right margin becomes the
    # left one. orig keeps the true page coordinates for drawing.
    words = []
    for bbox, text, prob in results:
        ox1, oy1 = bbox[0]
        ox2, oy2 = bbox[2]
        x1, x2 = (width - ox2, width - ox1) if mirror else (ox1, ox2)
        words.append({"x1": x1, "y1": oy1, "x2": x2, "y2": oy2,
                      "mid_y": (oy1 + oy2) / 2.0,
                      "orig": (ox1, oy1, ox2, oy2)})

    mw = float(np.median([w["x2"] - w["x1"] for w in words])) or 1.0
    mh = float(np.median([w["y2"] - w["y1"] for w in words])) or 1.0

    if mirror:
        # Reading order in the flipped frame, so the reference word is the
        # topmost word nearest the (new) left edge. The row bucket is coarse
        # (3x word height) because a finer one splits a slanted line across
        # buckets and the "first" word then is not the extreme one.
        words.sort(key=lambda w: (round(w["y1"] / max(1.0, 3.0 * mh)), w["x1"]))

    x3 = words[0]["x1"]
    outward = BAND_OUTWARD_FACTOR * mw
    inward = BAND_INWARD_FACTOR * mw

    band = [w for w in words if (x3 - outward) <= w["x1"] <= (x3 + inward)]

    # One measurement per text row: keep the most extreme word in each row.
    # Without this the widened band also admits the second word of a line,
    # which drags the measurement inward.
    rows = {}
    for w in band:
        key = round(w["mid_y"] / max(1.0, ROW_FACTOR * mh))
        if key not in rows or w["x1"] < rows[key]["x1"]:
            rows[key] = w
    measurements = sorted(rows.values(), key=lambda w: w["mid_y"])

    if len(measurements) < MIN_MEASUREMENTS:
        if verbose:
            print(f"Only {len(measurements)} measurable lines "
                  f"(need {MIN_MEASUREMENTS}) -- undetermined.")
        return None

    top, mid, bottom = split_sections(measurements)

    filtered_top = remove_outliers([int(round(w["x1"] - x3)) for w in top])
    filtered_mid = remove_outliers([int(round(w["x1"] - x3)) for w in mid])
    filtered_bottom = remove_outliers([int(round(w["x1"] - x3)) for w in bottom])

    # Section boundaries in page coordinates, for drawing only.
    b1 = (top[-1]["mid_y"] + mid[0]["mid_y"]) / 2.0
    b2 = (mid[-1]["mid_y"] + bottom[0]["mid_y"]) / 2.0

    if verbose:
        side = "RIGHT" if mirror else "LEFT"
        print(f"{side} margin: {len(words)} words, {len(band)} in band, "
              f"{len(measurements)} measurable lines "
              f"(inward {inward:.0f}px, outward {outward:.0f}px)")
        print(f"  top    ({len(top):2d}): {filtered_top}")
        print(f"  mid    ({len(mid):2d}): {filtered_mid}")
        print(f"  bottom ({len(bottom):2d}): {filtered_bottom}")

    if visualize:
        vis = image.copy()
        thick = max(2, int(round(height / 500)))
        for yb in (b1, b2):
            cv2.line(vis, (0, int(yb)), (width, int(yb)), (160, 160, 160), thick)
        for w in measurements:
            ox1, oy1, ox2, oy2 = w["orig"]
            cv2.rectangle(vis, (int(ox1), int(oy1)), (int(ox2), int(oy2)),
                          (0, 140, 255), thick)
        plt.figure(figsize=(11, 14))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title(f"{'RIGHT' if mirror else 'LEFT'} margin - "
                  f"{len(measurements)} measured lines, sections "
                  f"{len(top)}/{len(mid)}/{len(bottom)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "filtered_top": filtered_top,
        "filtered_mid": filtered_mid,
        "filtered_bottom": filtered_bottom,
        "x_plot1": x3,
        "n_words": len(words),
        "n_left_words": len(measurements),
        "section_counts": (len(top), len(mid), len(bottom)),
        "selected": [w["orig"] for w in measurements],
        "zone_bounds": (b1, b2),
        "mirrored": mirror,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\Shrey\Documents\Margin-Detection\images\Image_79.jpg'
    analyze_left_margin(path, visualize=True, verbose=True)
