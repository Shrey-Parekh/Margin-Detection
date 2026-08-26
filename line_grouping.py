"""Group OCR words into text lines.

Shared by the right-margin and baseline-slope detectors so both see the same
lines. Kept separate from either detector because line grouping is an input to
both, not part of either one's logic.
"""
import numpy as np
from sklearn.cluster import DBSCAN

# Vertical clustering radius as a fraction of median word height. Scaling to the
# handwriting rather than the page is what keeps tall scans from merging every
# line into one cluster.
LINE_EPS_FACTOR = 0.6


def to_word_boxes(results):
    """Convert EasyOCR output to dicts. bbox[0] is top-left, bbox[2] bottom-right."""
    boxes = []
    for bbox, text, prob in results:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        boxes.append({
            "text": text,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "mid_y": (y1 + y2) / 2,
            "height": y2 - y1,
            "width": x2 - x1,
            "prob": prob,
        })
    return boxes


def median_word_height(word_boxes):
    hs = [w["height"] for w in word_boxes if w["height"] > 0]
    return float(np.median(hs)) if hs else 0.0


def median_word_width(word_boxes):
    ws = [w["width"] for w in word_boxes if w["width"] > 0]
    return float(np.median(ws)) if ws else 0.0


def group_lines(word_boxes, image_height=None):
    """Cluster words into lines by vertical midpoint.

    Returns (lines, eps_used). Lines are sorted top to bottom.

    min_samples=1 because a one-word line is still a line -- with min_samples=2
    DBSCAN labels it noise and drops it, which disproportionately removes short
    final lines, exactly the ones the bottom zone depends on.
    """
    if not word_boxes:
        return [], 0.0

    wh = median_word_height(word_boxes)
    if wh > 0:
        eps = LINE_EPS_FACTOR * wh
    else:
        eps = 0.04 * (image_height or 1000)

    mids = np.array([[w["mid_y"]] for w in word_boxes])
    labels = DBSCAN(eps=eps, min_samples=1).fit(mids).labels_

    lines_dict = {}
    for label, word in zip(labels, word_boxes):
        if label == -1:
            continue
        lines_dict.setdefault(label, []).append(word)

    lines = sorted(lines_dict.values(),
                   key=lambda line: np.mean([w["mid_y"] for w in line]))
    return lines, eps


def line_y(line):
    """Representative vertical position of a line."""
    return float(np.mean([w["mid_y"] for w in line]))


def split_into_thirds(items, key):
    """Split items into top/mid/bottom thirds by the y range they span.

    Mirrors the left-margin zone split: the text block's own vertical extent is
    divided into three, not the page.
    """
    if not items:
        return [], [], []
    ys = [key(i) for i in items]
    y_min, y_max = min(ys), max(ys)
    if y_max == y_min:
        return list(items), [], []
    b1 = y_min + (y_max - y_min) / 3.0
    b2 = y_min + 2.0 * (y_max - y_min) / 3.0

    top, mid, bottom = [], [], []
    for item in items:
        y = key(item)
        if y <= b1:
            top.append(item)
        elif y <= b2:
            mid.append(item)
        else:
            bottom.append(item)
    return top, mid, bottom
