"""Cache raw per-WORD OCR boxes so line-grouping algorithms can be compared
offline without re-running OCR.

line_cache.json stores lines (already grouped). This stores the words themselves,
which is what any alternative grouping method needs as input.
"""
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")

import cv2

from detect_left_margin import get_reader

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "word_cache.json")


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
            words = []
            for bbox, text, prob in results:
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]
                words.append([round(float(x1), 1), round(float(y1), 1),
                              round(float(x2), 1), round(float(y2), 1),
                              text, round(float(prob), 3)])
            out[name] = {
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "words": words,
            }
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}", "words": []}

        if i % 20 == 0 or i == len(names):
            print(f"  {i}/{len(names)}  {time.time() - t0:.0f}s", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"cached {len(out)} images -> {os.path.basename(OUT)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
