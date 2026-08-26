import csv
import os
import statistics as st

# Perceptual deadband in pixels. Differences smaller than this are not
# resolvable by the human eye. This single constant governs every left-margin
# decision below. Calibrate it against the expert-labelled set rather than
# leaving it at the default.
SLM_DEADBAND_PX = 6

# Slope magnitude below which a baseline counts as level.
SLOPE_DEADBAND = 0.009

CSV_HEADERS = ["image",
               "SLM", "WLM", "DAFLM", "RFLM", "CCLM", "CVLM",
               "SRM", "WRM", "DAFRM", "RRM", "CCRM", "CVRM",
               "TT", "TS", "TDA", "BT", "BS", "BDA",
               "left_valid", "right_valid", "top_valid", "bottom_valid",
               "left_top_px", "left_mid_px", "left_bottom_px",
               "left_spread_px", "left_scatter_px",
               "right_top_px", "right_mid_px", "right_bottom_px",
               "right_spread_px", "right_scatter_px",
               "top_gradient", "bottom_gradient",
               "n_words", "n_left_band_words", "n_right_band_words", "n_lines"]


def zone_scatter(*zones):
    # Pooled median absolute deviation of individual line offsets from their
    # OWN zone mean. Measuring within zones removes the trend, so what is left
    # is how much the lines wander -- the signal that distinguishes a wavy
    # margin from a steadily drifting one.
    devs = []
    for zone in zones:
        m = st.mean(zone)
        devs.extend(abs(v - m) for v in zone)
    return st.median(devs) if devs else 0.0


def zone_cmp(a, b, deadband=SLM_DEADBAND_PX):
    # Deadband-aware comparison: differences under the perceptual threshold
    # count as "no step", not as a trend.
    if b - a > deadband:
        return -1
    if a - b > deadband:
        return 1
    return 0


def classify_margin(filtered_top, filtered_mid, filtered_bottom, names,
                    trend_deadband=SLM_DEADBAND_PX, scatter_deadband=None):
    """Classify a margin's drift down the page.

    One implementation shared by the left and right margins -- the sides differ
    only in their scatter threshold, because word wrap makes the right edge
    ragged by construction. `names` maps roles to flag names:
    (straight, wavy, widening, narrowing, bulge, pinch).
    """
    straight, wavy, widening, narrowing, bulge, pinch = names
    flags = {n: 0 for n in names}
    mags = dict(avg_top=None, avg_mid=None, avg_bottom=None,
                spread=None, scatter=None, valid=0)

    if scatter_deadband is None:
        scatter_deadband = trend_deadband

    # A zone with no surviving readings cannot be classified. Mark the margin
    # undetermined rather than guessing -- all six flags stay 0.
    if not (filtered_top and filtered_mid and filtered_bottom):
        return flags, mags

    avg_top = st.mean(filtered_top)
    avg_mid = st.mean(filtered_mid)
    avg_bottom = st.mean(filtered_bottom)
    spread = max(avg_top, avg_mid, avg_bottom) - min(avg_top, avg_mid, avg_bottom)
    scatter = zone_scatter(filtered_top, filtered_mid, filtered_bottom)

    mags.update(avg_top=avg_top, avg_mid=avg_mid, avg_bottom=avg_bottom,
                spread=spread, scatter=scatter, valid=1)

    if spread <= trend_deadband and scatter <= scatter_deadband:
        # No perceptible drift and no perceptible wander.
        flags[straight] = 1
    elif scatter > max(scatter_deadband, spread):
        # Lines wander more than they trend: irregular, not directional.
        flags[wavy] = 1
    else:
        d1 = zone_cmp(avg_top, avg_mid, trend_deadband)
        d2 = zone_cmp(avg_mid, avg_bottom, trend_deadband)
        if d1 == 0 and d2 == 0:
            # Neither individual step clears the deadband but the overall drift
            # does: a gradual, uniform trend. Classify on the endpoints.
            if zone_cmp(avg_top, avg_bottom, trend_deadband) < 0:
                flags[widening] = 1
            else:
                flags[narrowing] = 1
        elif d1 <= 0 and d2 <= 0:
            flags[widening] = 1
        elif d1 >= 0 and d2 >= 0:
            flags[narrowing] = 1
        elif d1 < 0 and d2 > 0:
            flags[bulge] = 1
        else:
            flags[pinch] = 1

    return flags, mags


# Left:  Straight / Wavy / Widening / Narrowing / Bulge / Pinch
LEFT_NAMES = ("SLM", "WLM", "DAFLM", "RFLM", "CCLM", "CVLM")
# Right: same six roles, measured by mirroring the coordinates and running the
# left-margin code, so the thresholds are identical too.
RIGHT_NAMES = ("SRM", "WRM", "DAFRM", "RRM", "CCRM", "CVRM")


def classify_left(filtered_top, filtered_mid, filtered_bottom):
    return classify_margin(filtered_top, filtered_mid, filtered_bottom,
                           LEFT_NAMES)


def classify_right(filtered_top, filtered_mid, filtered_bottom):
    return classify_margin(filtered_top, filtered_mid, filtered_bottom,
                           RIGHT_NAMES)


def classify_slope(gradient, valid, prefix):
    """Classify one baseline slope. prefix is 'T' or 'B'.

    An invalid fit (too few words) yields all-zero flags rather than a
    fabricated "level" reading.
    """
    flags = {f"{prefix}T": 0, f"{prefix}S": 0, f"{prefix}DA": 0}
    if not valid:
        return flags
    if gradient < -SLOPE_DEADBAND:
        flags[f"{prefix}DA"] = 1
    elif gradient > SLOPE_DEADBAND:
        flags[f"{prefix}T"] = 1
    else:
        flags[f"{prefix}S"] = 1
    return flags


def build_row(image_name, left, slope, right=None):
    """Assemble one CSV row from the detector outputs."""
    if left is None:
        lf, lm = classify_left([], [], [])
    else:
        lf, lm = classify_left(left["filtered_top"], left["filtered_mid"],
                               left["filtered_bottom"])

    if right is None:
        rf, rm = classify_right([], [], [])
        n_right_words = 0
    else:
        rf, rm = classify_right(right["filtered_top"], right["filtered_mid"],
                                right["filtered_bottom"])
        n_right_words = right.get("n_left_words", 0)  # words the mirrored band kept

    if slope is None:
        tf = classify_slope(0.0, False, "T")
        bf = classify_slope(0.0, False, "B")
        tg = bg = None
        tv = bv = 0
        n_lines = 0
    else:
        tf = classify_slope(slope["top_gradient"], slope["top_valid"], "T")
        bf = classify_slope(slope["bottom_gradient"], slope["bottom_valid"], "B")
        tg = slope["top_gradient"] if slope["top_valid"] else None
        bg = slope["bottom_gradient"] if slope["bottom_valid"] else None
        tv = int(slope["top_valid"])
        bv = int(slope["bottom_valid"])
        n_lines = slope["n_lines"]

    def num(v, nd=3):
        # Undetermined stays blank in the CSV so it can never be read as a 0.
        return "" if v is None else round(v, nd)

    return [image_name,
            lf["SLM"], lf["WLM"], lf["DAFLM"], lf["RFLM"], lf["CCLM"], lf["CVLM"],
            rf["SRM"], rf["WRM"], rf["DAFRM"], rf["RRM"], rf["CCRM"], rf["CVRM"],
            tf["TT"], tf["TS"], tf["TDA"], bf["BT"], bf["BS"], bf["BDA"],
            lm["valid"], rm["valid"], tv, bv,
            num(lm["avg_top"]), num(lm["avg_mid"]), num(lm["avg_bottom"]),
            num(lm["spread"]), num(lm["scatter"]),
            num(rm["avg_top"]), num(rm["avg_mid"]), num(rm["avg_bottom"]),
            num(rm["spread"]), num(rm["scatter"]),
            num(tg, 6), num(bg, 6),
            left["n_words"] if left else 0,
            left.get("n_left_words", 0) if left else 0,
            n_right_words, n_lines]


# Binary-only output: the twelve margin flags, nothing else. Left and right
# only -- the top/bottom baseline-slope features are deliberately excluded.
BINARY_HEADERS = list(LEFT_NAMES) + list(RIGHT_NAMES)


def build_binary_row(left, right):
    """One row of twelve 0/1 flags: six left, six right.

    An undetermined margin yields all zeros for that side, which stays
    distinguishable from any real class without needing a separate column.
    """
    if left is None:
        lf, _ = classify_left([], [], [])
    else:
        lf, _ = classify_left(left["filtered_top"], left["filtered_mid"],
                              left["filtered_bottom"])
    if right is None:
        rf, _ = classify_right([], [], [])
    else:
        rf, _ = classify_right(right["filtered_top"], right["filtered_mid"],
                               right["filtered_bottom"])
    return [lf[n] for n in LEFT_NAMES] + [rf[n] for n in RIGHT_NAMES]


# Binary-only output for all FOUR margins: left, right, top slant, bottom
# slant. 18 columns, every value 0 or 1, no magnitudes.
ALL_BINARY_HEADERS = (["image"] + list(LEFT_NAMES) + list(RIGHT_NAMES)
                      + ["TT", "TS", "TDA", "BT", "BS", "BDA"])


def build_binary_row_all(image_name, left, right, slope):
    """One row of eighteen 0/1 flags: six left, six right, three top, three
    bottom. An undetermined margin yields all zeros for that group.
    """
    if left is None:
        lf, _ = classify_left([], [], [])
    else:
        lf, _ = classify_left(left["filtered_top"], left["filtered_mid"],
                              left["filtered_bottom"])
    if right is None:
        rf, _ = classify_right([], [], [])
    else:
        rf, _ = classify_right(right["filtered_top"], right["filtered_mid"],
                               right["filtered_bottom"])
    if slope is None:
        tf = classify_slope(0.0, False, "T")
        bf = classify_slope(0.0, False, "B")
    else:
        tf = classify_slope(slope["top_gradient"], slope["top_valid"], "T")
        bf = classify_slope(slope["bottom_gradient"], slope["bottom_valid"], "B")

    return ([image_name]
            + [lf[n] for n in LEFT_NAMES] + [rf[n] for n in RIGHT_NAMES]
            + [tf["TT"], tf["TS"], tf["TDA"], bf["BT"], bf["BS"], bf["BDA"]])


def write_rows(rows, csv_filename, headers=CSV_HEADERS):
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(rows)


if __name__ == "__main__":
    import sys
    from detect_left_margin import analyze_left_margin, get_reader
    from detect_baseline_slope import analyze_baseline_slope
    from detect_right_margin import analyze_right_margin
    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\Shrey\Documents\Margin-Detection\images\Image_79.jpg'
    img = cv2.imread(path)
    res = get_reader().readtext(img)
    row = build_row(os.path.basename(path),
                    analyze_left_margin(path, results=res, image=img),
                    analyze_baseline_slope(path, results=res, image=img),
                    analyze_right_margin(path, results=res, image=img))
    for k, v in zip(CSV_HEADERS, row):
        print(f"  {k:26s} {v}")
