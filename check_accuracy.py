
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = os.path.join(HERE, "features_auto_binary.csv")
MANUAL = os.path.join(HERE, "features_manual_29.csv")
IMAGE_DIR = os.path.join(HERE, "images")

# detector name -> expert column name
LEFT_MAP = [("SLM", "slm"), ("WLM", "wlm"), ("DAFLM", "daflm"),
            ("RFLM", "rlm"), ("CCLM", "cclm"), ("CVLM", "cvlm")]
LABELS = ["Straight", "Wavy", "Widening", "Narrowing", "Bulge", "Pinch"]


def natural_key(name):
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)) if m else 0, name)


def class_of(row, cols):
    """Index of the set flag, or None if the row is all zeros."""
    for i, c in enumerate(cols):
        if row[c] == 1:
            return i
    return None


def main():
    auto = pd.read_csv(AUTO)
    man = pd.read_csv(MANUAL)
    n = len(man)

    names = sorted([f for f in os.listdir(IMAGE_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))],
                   key=natural_key)

    print(f"detector rows : {len(auto)}")
    print(f"expert rows   : {n}")
    print(f"comparing the first {n} images: {names[0]} ... {names[n-1]}")
    print()

    a = auto.iloc[:n].reset_index(drop=True)
    det_cols = [d for d, _ in LEFT_MAP]
    exp_cols = [e for _, e in LEFT_MAP]

    det = [class_of(a.iloc[i], det_cols) for i in range(n)]
    exp = [class_of(man.iloc[i], exp_cols) for i in range(n)]

    pairs = [(d, e) for d, e in zip(det, exp) if d is not None and e is not None]
    if not pairs:
        print("no comparable rows")
        return

    correct = sum(1 for d, e in pairs if d == e)
    chance = 100.0 / len(LEFT_MAP)

    print("=" * 58)
    print(f"LEFT MARGIN ACCURACY: {correct}/{len(pairs)} = "
          f"{correct / len(pairs) * 100:.1f}%   (chance {chance:.1f}%)")
    print("=" * 58)

    # A model that always guesses the expert's most common class.
    base = max(set(e for _, e in pairs), key=[e for _, e in pairs].count)
    base_acc = sum(1 for _, e in pairs if e == base) / len(pairs) * 100
    print(f"majority-class baseline: always guess "
          f"'{LABELS[base]}' -> {base_acc:.1f}%")
    print()

    print("per-class recall (of the expert's calls, how many were matched)")
    for i, lab in enumerate(LABELS):
        tot = sum(1 for _, e in pairs if e == i)
        hit = sum(1 for d, e in pairs if e == i and d == i)
        if tot:
            print(f"  {lab:10s} {hit:2d}/{tot:2d}  ({hit / tot * 100:5.1f}%)")
        else:
            print(f"  {lab:10s}  -     expert never used this class")
    print()

    print("confusion matrix   rows = expert, cols = detector")
    m = pd.DataFrame(0, index=[f"exp_{l}" for l in LABELS],
                     columns=[f"det_{l}" for l in LABELS])
    for d, e in pairs:
        m.iloc[e, d] += 1
    print(m.to_string())
    print()

    print("class distribution")
    print(f"  {'class':10s} {'expert':>8s} {'detector':>9s}")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:10s} {sum(1 for _, e in pairs if e == i):8d} "
              f"{sum(1 for d, _ in pairs if d == i):9d}")
    print()
    print("NOTE: the right margin has no expert labels, so it is not scored.")


if __name__ == "__main__":
    main()
