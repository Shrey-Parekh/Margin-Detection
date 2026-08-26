
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")  # never open a window during a batch run

import cv2

from detect_left_margin import analyze_left_margin, get_reader
from detect_right_margin import analyze_right_margin
from extract_features import build_binary_row, write_rows, BINARY_HEADERS

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(HERE, "images")
OUT_CSV = os.path.join(HERE, "features_auto_binary.csv")


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
    rows, failures, undetermined = [], [], []
    t0 = time.time()

    for i, name in enumerate(names, 1):
        path = os.path.join(IMAGE_DIR, name)
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError("unreadable image")
            # One OCR pass, shared by both detectors.
            results = reader.readtext(image)
            left = analyze_left_margin(path, results=results, image=image)
            right = analyze_right_margin(path, results=results, image=image)
            row = build_binary_row(left, right)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            row = build_binary_row(None, None)

        if sum(row[:6]) == 0 or sum(row[6:]) == 0:
            undetermined.append(name)
        rows.append(row)

        if i % 20 == 0 or i == len(names):
            el = time.time() - t0
            print(f"  {i}/{len(names)}  {el:.0f}s  ({el / i:.1f}s/img)", flush=True)

    write_rows(rows, out_csv, headers=BINARY_HEADERS)
    print(f"\nwrote {len(rows)} rows x {len(BINARY_HEADERS)} columns "
          f"in {time.time() - t0:.0f}s")
    if undetermined:
        print(f"{len(undetermined)} rows have an all-zero side (undetermined): "
              f"{', '.join(undetermined[:10])}"
              f"{' ...' if len(undetermined) > 10 else ''}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for n, e in failures[:15]:
            print(f"  {n}: {e}")


if __name__ == "__main__":
    main()
