"""Run all FOUR margin detectors (left, right, top slant, bottom slant) over
every image in images/ and write one row of binary flags per image.

Output columns: image, then eighteen 0/1 flags --
    SLM WLM DAFLM RFLM CCLM CVLM   SRM WRM DAFRM RRM CCRM CVRM   TT TS TDA BT BS BDA

Note: top/bottom use the DBSCAN line grouping in line_grouping.py, which is
known to under-group lines on some images (slanted handwriting can chain
separate lines into one cluster). Left/right do not depend on this and are
unaffected. Treat TT/TS/TDA/BT/BS/BDA as less reliable than the left/right
columns until that grouping is revisited.
"""
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")  # never open a window during a batch run

import cv2

from detect_left_margin import analyze_left_margin, get_reader
from detect_right_margin import analyze_right_margin
from detect_baseline_slope import analyze_baseline_slope
from extract_features import build_binary_row_all, write_rows, ALL_BINARY_HEADERS

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
OUT_CSV = os.path.join(HERE, "features_auto_binary_all.csv")


def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 0, name)


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else OUT_CSV
    if os.path.exists(out_csv):
        os.remove(out_csv)

    names = sorted(
        [f for f in os.listdir(IMAGE_DIR)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=natural_key)
    print(f"{len(names)} images -> {os.path.basename(out_csv)}", flush=True)

    reader = get_reader()
    rows, failures = [], []
    t0 = time.time()

    for i, name in enumerate(names, 1):
        path = os.path.join(IMAGE_DIR, name)
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("unreadable image")
            # One OCR pass, shared by all three detectors.
            results = reader.readtext(image)
            left = analyze_left_margin(path, results=results, image=image)
            right = analyze_right_margin(path, results=results, image=image)
            slope = analyze_baseline_slope(path, results=results, image=image)
            row = build_binary_row_all(name, left, right, slope)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            row = build_binary_row_all(name, None, None, None)

        rows.append(row)

        if i % 20 == 0 or i == len(names):
            el = time.time() - t0
            print(f"  {i}/{len(names)}  {el:.0f}s  ({el / i:.1f}s/img)", flush=True)

    write_rows(rows, out_csv, headers=ALL_BINARY_HEADERS)
    print(f"\nwrote {len(rows)} rows x {len(ALL_BINARY_HEADERS)} columns "
          f"in {time.time() - t0:.0f}s")
    if failures:
        print(f"\n{len(failures)} failures:")
        for n, e in failures[:15]:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
