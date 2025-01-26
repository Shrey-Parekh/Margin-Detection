import easyocr
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Load the image
image_path = r'D:\Margin Detection\Margin-Detection\new images\Image_4.jpg'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

# Step 2: Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])
results = reader.readtext(image)

if not results:
    raise ValueError("No text detected in the image.")

# Step 3: Process Detected Words
height, width, _ = image.shape
word_boxes = []
for bbox, text, prob in results:
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    word_boxes.append({"text": text, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "mid_y": (y1 + y2) / 2, "prob": prob})

# Step 4: Group Words into Lines
word_boxes = sorted(word_boxes, key=lambda x: x["mid_y"])  # Sort by vertical midpoint
lines = []
current_line = [word_boxes[0]]
line_spacing_threshold = 0.03 * height  # Dynamic threshold based on image height
for word in word_boxes[1:]:
    if abs(word["mid_y"] - current_line[-1]["mid_y"]) <= line_spacing_threshold:
        current_line.append(word)
    else:
        lines.append(current_line)
        current_line = [word]
lines.append(current_line)

# Step 5: Analyze Top Line and Bottom Line
def calculate_weighted_gradient(line, is_bottom):
    x_coords = [(word["x1"] + word["x2"]) / 2 for word in line]
    y_coords = [word["y2"] if is_bottom else word["y1"] for word in line]
    weights = [word["prob"] for word in line]

    x_coords = np.array(x_coords).reshape(-1, 1)
    y_coords = np.array(y_coords).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_coords, y_coords, sample_weight=weights)  # Weighted regression
    return model.coef_[0][0], model

# Consider the first and last two lines for averaging
top_lines = lines[:2]
bottom_lines = lines[-2:]

def average_gradient(lines, is_bottom):
    gradients = []
    for line in lines:
        gradient, _ = calculate_weighted_gradient(line, is_bottom)
        gradients.append(gradient)
    return np.mean(gradients)

top_line_gradient = average_gradient(top_lines, is_bottom=False)
bottom_line_gradient = average_gradient(bottom_lines, is_bottom=True)

# Step 6: Visualization
def draw_lines(image, lines, color):
    for line in lines:
        for word in line:
            cv2.rectangle(image, (int(word["x1"]), int(word["y1"])), (int(word["x2"]), int(word["y2"])), color, 2)

draw_lines(image, top_lines, (0, 255, 0))  # Green for top lines
draw_lines(image, bottom_lines, (255, 0, 0))  # Blue for bottom lines

def draw_fitted_line(model, x_coords, color):
    x_range = np.linspace(min(x_coords), max(x_coords), 1)
    y_range = model.predict(x_range.reshape(-1, 1))
    for x, y in zip(x_range, y_range):
        cv2.circle(image, (int(x), int(y)), 2, color, 2)

# Display Results
# plt.figure(figsize=(12, 8))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Top Line and Bottom Line")
# plt.axis("off")
# plt.show()

print(f"Top Line Gradient: {top_line_gradient}")
print(f"Bottom Line Gradient: {bottom_line_gradient}")