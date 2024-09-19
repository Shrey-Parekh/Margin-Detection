import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

reader = easyocr.Reader(['en'])

image_path = r'C:\Users\Shrey Parekh\Documents\Margin-Detection\images\Image_25.jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

results = reader.readtext(image)
height, width, _ = image.shape

min_x, min_y = width, height
max_x, max_y = 0, 0
x3, y3 = None, None
y4 = x4 = 0
first_y1 = None

first_word_midpoint = []
x_diff = []

if results:
    last_y1 = None
    for id, (bbox, text, prob) in enumerate(results):
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        y_midpoint = (y1 + y2) / 2
        if last_y1 is None or abs(y1 - last_y1) > 30:
            first_word_midpoint.append((x1, y_midpoint))
        last_y1 = y1

        if id == 0:
            x3, y3 = x1, y1
            first_y1 = y1
        y4 = max(y4, y2)
        x4 = max(x4, x2)

        if (x3 - 150) <= x1 <= (x3 + 150):
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    y_plot1 = np.arange(y3, y4, 0.1)
    x_plot1 = x3
    for y_val in y_plot1:
        cv2.circle(image, (int(x_plot1), int(y_val)), 1, (0, 0, 255), -1)

    # sorted_first_word_midpoint = sorted(first_word_midpoint, key=lambda x: x[0])

    for x, y_midpoint in first_word_midpoint:
        if (x_plot1 - 150) <= x <= (x_plot1 + 150):
            cv2.line(image, (int(x_plot1), int(y_midpoint)), (int(x), int(y_midpoint)), (0, 0, 0), 3)
            x_diff.append(x - x_plot1)
    for i in x_diff:
        print(i)

    x_plot2 = np.arange(x3, x4, 0.1)
    y_plot2 = y3
    for x_val in x_plot2:
        cv2.circle(image, (int(x_val), int(y_plot2)), 1, (0, 0, 255), -1)

    for x_val2 in x_plot2:
        cv2.circle(image, (int(x_val2), int(y4)), 1, (0, 0, 255), -1)

    for y_val2 in y_plot1:
        cv2.circle(image, (int(x4), int(y_val2)), 1, (0, 0, 255), -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Processed Image with Margins")
    ax1.axis('off')

    ax2.scatter(range(len(x_diff)), x_diff, color='b', marker='o')

    if len(x_diff) > 1:
        x_vals = np.arange(len(x_diff))
        coefficients = np.polyfit(x_vals, x_diff, 1)
        poly = np.poly1d(coefficients)
        fit_line = poly(x_vals)
        ax2.plot(x_vals, fit_line, color='r', linestyle='-', label='Best Fit Line')

    ax2.set_title('Scatter Plot of x_diff from Left Margin')
    ax2.set_xlabel('Line Index (Order of Lines)')
    ax2.set_ylabel('x_diff (Distance from Left Margin)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

else:
    print("No text detected in the image.")
