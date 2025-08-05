import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import DBSCAN

image_path = r'C:\Users\Shrey\Documents\Margin-Detection\images\Image_79.jpg'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

reader = easyocr.Reader(['en'], gpu=True)
results = reader.readtext(image)

if not results:
    raise ValueError("No text detected in the image.")

height, width, _ = image.shape
word_boxes = []
for bbox, text, prob in results:
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    word_boxes.append({
        "text": text,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "mid_y": (y1 + y2) / 2,
        "prob": prob
    })

if not word_boxes:
    raise ValueError("No words detected in the image.")

mid_ys = np.array([[w["mid_y"]] for w in word_boxes])
db = DBSCAN(eps=0.04 * height, min_samples=2).fit(mid_ys)
labels = db.labels_

lines_dict = {}
for label, word in zip(labels, word_boxes):
    if label == -1:
        continue
    lines_dict.setdefault(label, []).append(word)

lines = list(lines_dict.values())
lines = sorted(lines, key=lambda line: np.mean([w["mid_y"] for w in line]))

if len(lines) < 2:
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

# Alternative method if still not enough lines
if len(lines) < 2:
    print("Not enough lines detected, using alternative method based on word positions.")
    word_boxes_sorted = sorted(word_boxes, key=lambda x: x["mid_y"])
    N = max(2, len(word_boxes) // 10)
    top_words = word_boxes_sorted[:N]
    bottom_words = word_boxes_sorted[-N:]
    def fit_line_from_words(words, is_bottom):
        x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in words]).reshape(-1, 1)
        y_coords = np.array([w["y2"] if is_bottom else w["y1"] for w in words])
        if len(words) < 2:
            return 0.0, None
        model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
        model.fit(x_coords, y_coords)
        return model.estimator_.coef_[0], model
    top_line_gradient, top_model = fit_line_from_words(top_words, is_bottom=False)
    bottom_line_gradient, bottom_model = fit_line_from_words(bottom_words, is_bottom=True)
    image_vis = image.copy()
    for w in top_words:
        cv2.rectangle(image_vis, (int(w["x1"]), int(w["y1"])), (int(w["x2"]), int(w["y2"])), (0, 140, 255), 2)
    for w in bottom_words:
        cv2.rectangle(image_vis, (int(w["x1"]), int(w["y1"])), (int(w["x2"]), int(w["y2"])), (0, 140, 255), 2)
    def draw_fitted_line(image, words, model, color):
        if model is None:
            return
        x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in words])
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
        y_range = model.predict(x_range)
        pts = np.vstack([x_range.flatten(), y_range]).T.astype(np.int32)
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2)
    draw_fitted_line(image_vis, top_words, top_model, (0, 128, 0))
    draw_fitted_line(image_vis, bottom_words, bottom_model, (128, 0, 0))
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
    plt.title("Top and Bottom Line Gradients (Alternative Method)")
    plt.axis("off")
    plt.show()
    print(f"Top Line Gradient (alternative): {top_line_gradient}")
    print(f"Bottom Line Gradient (alternative): {bottom_line_gradient}")
    exit()

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
    line = remove_outliers(line, is_bottom)
    x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in line]).reshape(-1, 1)
    y_coords = np.array([w["y2"] if is_bottom else w["y1"] for w in line])
    if len(line) < 2:
        return 0.0, None
    model = RANSACRegressor(estimator=LinearRegression(), random_state=42)
    model.fit(x_coords, y_coords)
    return model.estimator_.coef_[0], model

single_top_line = [lines[0]]
single_bottom_line = [lines[-1]]

def average_gradient(lines, is_bottom):
    gradients = []
    models = []
    for line in lines:
        grad, model = fit_line(line, is_bottom)
        gradients.append(grad)
        models.append((line, model))
    return np.mean(gradients), gradients, models

top_line_gradient, top_line_gradients, top_models = average_gradient(single_top_line, is_bottom=False)
bottom_line_gradient, bottom_line_gradients, bottom_models = average_gradient(single_bottom_line, is_bottom=True)

def draw_lines(image, lines):
    for line in lines:
        for word in line:
            cv2.rectangle(image, (int(word["x1"]), int(word["y1"])), (int(word["x2"]), int(word["y2"])), (0, 140, 255), 2)

def draw_fitted_line(image, line, model, color):
    if model is None:
        return
    x_coords = np.array([(w["x1"] + w["x2"]) / 2 for w in line])
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    x_range = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    pts = np.vstack([x_range.flatten(), y_range]).T.astype(np.int32)
    cv2.polylines(image, [pts], isClosed=False, color=color, thickness=2)

image_vis = image.copy()
draw_lines(image_vis, single_top_line)
draw_lines(image_vis, single_bottom_line)
for (line, model) in top_models:
    draw_fitted_line(image_vis, line, model, (0, 128, 0))
for (line, model) in bottom_models:
    draw_fitted_line(image_vis, line, model, (128, 0, 0))

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))
plt.title("Top and Bottom Line Gradients")
plt.axis("off")
plt.show()

print(f"Top Line Gradient: {top_line_gradient}")
print(f"Bottom Line Gradient: {bottom_line_gradient}")
