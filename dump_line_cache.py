
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")

import cv2

from detect_left_margin import get_reader
from line_grouping import (to_word_boxes, group_lines, line_y,
                           median_word_width, median_word_height)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_cache.json")


def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 0, name)


def main():
    names = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                   key=natural_key)
    reader = get_reader()
    out = {}
    t0 = time.time()

    for i, name in enumerate(names, 1):
        path = os.path.join(IMAGE_DIR, name)
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("unreadable")
            results = reader.readtext(image)
            wb = to_word_boxes(results)
            lines, eps = group_lines(wb, image.shape[0])
            out[name] = {
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "n_words": len(wb),
                "median_word_width": median_word_width(wb),
                "median_word_height": median_word_height(wb),
                "eps": eps,
                "lines": [
                    {
                        "y": line_y(L),
                        "right": float(max(w["x2"] for w in L)),
                        "left": float(min(w["x1"] for w in L)),
                        "n_words": len(L),
                    }
                    for L in lines
                ],
            }
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}", "lines": []}

        if i % 20 == 0 or i == len(names):
            print(f"  {i}/{len(names)}  {time.time() - t0:.0f}s")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"cached {len(out)} images -> {os.path.basename(OUT)} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
