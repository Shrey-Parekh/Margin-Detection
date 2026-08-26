import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

from detect_left_margin import get_reader
from line_grouping import to_word_boxes, group_lines


def remove_outliers(line, is_bottom):
    y_coords = np.array([w["y2"] if is_bottom else w["y1"] for w in line])
    median = np.median(y_coords)
    mad = np.median(np.abs(y_coords - median))
    if mad == 0:
        return line
    threshold = 2.5 * mad
    filtered = [w for w, y in zip(line, y_coords) if abs(y - median) <= threshold]
    return filtered if filtered else line


def fit_line(line, is_bottom):
    """Fit a baseline. Returns (gradient, model, valid).

    valid is False when there are too few words to fit anything -- the caller
    must not treat the returned 0.0 as a real "level line" measurement.
    """
    line = remove_outliers(line, is_bottom)
    x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in line]).reshape(-1, 1)
    y_coords = np.array([w["y2"] if is_bottom else w["y1"] for w in line])
    if len(line) < 2:
        return 0.0, None, False
    model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
    model.fit(x_coords, y_coords)
    return model.estimator_.coef_[0], model, True


def draw_fitted_line(image, line, model, color):
    if model is None:
        return
    x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in line])
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    pts = np.vstack([x_range.flatten(), y_range]).T.astype(np.int32)
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2)


def analyze_baseline_slope(image_path, results=None, image=None,
                           visualize=False, verbose=False):
    """Measure first/last text-line slope for one image.

    Returns a dict with both gradients and their validity flags, or None if OCR
    found no text. Grouping and fitting logic is unchanged from the original
    script; the module-level exit() is gone so a batch run cannot be killed by
    one bad image.
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

    height, width, _ = image.shape
    word_boxes = to_word_boxes(results)

    if not word_boxes:
        return None

    # Shared grouping: eps scales with handwriting size, not page height.
    lines, eps = group_lines(word_boxes, height)
    n_noise = 0

    method = "dbscan"
    if len(lines) < 2:
        method = "spacing"
        word_boxes_sorted = sorted(word_boxes, key=lambda x: x["mid_y"])
        lines = []
        current_line = [word_boxes_sorted[0]]
        line_spacing_threshold = 0.03 * height
        for word in word_boxes_sorted[1:]:
            if abs(word["mid_y"] - current_line[-1]["mid_y"]) <= line_spacing_threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)
        lines = sorted(lines, key=lambda line: np.mean([w["mid_y"] for w in line]))

    if len(lines) < 2:
        # Fallback: treat the topmost/bottommost N words as the two lines.
        method = "topbottom_n"
        word_boxes_sorted = sorted(word_boxes, key=lambda x: x["mid_y"])
        N = max(2, len(word_boxes) // 10)
        top_line = word_boxes_sorted[:N]
        bottom_line = word_boxes_sorted[-N:]
    else:
        top_line = lines[0]
        bottom_line = lines[-1]

    top_gradient, top_model, top_valid = fit_line(top_line, is_bottom=False)
    bottom_gradient, bottom_model, bottom_valid = fit_line(bottom_line, is_bottom=True)

    if visualize:
        image_vis = image.copy()
        for line in (top_line, bottom_line):
            for w in line:
                cv2.rectangle(image_vis, (int(w["x1"]), int(w["y1"])),
                              (int(w["x2"]), int(w["y2"])), (0, 140, 255), 2)
        draw_fitted_line(image_vis, top_line, top_model, (0, 128, 0))
        draw_fitted_line(image_vis, bottom_line, bottom_model, (128, 0, 0))
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
        plt.title("Top and Bottom Line Gradients")
        plt.axis("off")
        plt.show()

    if verbose:
        print(f"Top Line Gradient: {top_gradient}")
        print(f"Bottom Line Gradient: {bottom_gradient}")

    return {
        "top_gradient": float(top_gradient),
        "bottom_gradient": float(bottom_gradient),
        "top_valid": bool(top_valid),
        "bottom_valid": bool(bottom_valid),
        "n_lines": len(lines),
        "n_top_words": len(top_line),
        "n_bottom_words": len(bottom_line),
        "n_noise_words": n_noise,
        "line_method": method,
    }


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r'C:\Users\Shrey\Documents\Margin-Detection\images\Image_79.jpg'
    analyze_baseline_slope(path, visualize=True, verbose=True)
