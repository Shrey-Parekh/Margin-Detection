import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Path to the image
image_path = r'C:\Users\Shrey Parekh\Documents\Margin-Detection\images\Image_4.jpg'
image = cv2.imread(image_path)

# Check if the image exists
if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

# OCR: Detect text in the image
results = reader.readtext(image)
height, width, _ = image.shape

# Set initial values for bounding boxes and coordinates
min_x, min_y = width, height
max_x, max_y = 0, 0
x3, y3 = None, None
y4 = x4 = 0
first_y1 = None

# Grouping words into lines based on their vertical (y) position
line_threshold = 15  # Increased threshold for grouping into the same line
lines = []  # Will store the words grouped by lines

# If results were found
if results:
    # Iterate through each detected bounding box and text
    for idx, (bbox, text, prob) in enumerate(results):
        x1, y1 = bbox[0]  # Top-left corner
        x2, y2 = bbox[2]  # Bottom-right corner

        # If it's the first box, record the first x and y for plotting
        if idx == 0:
            x3, y3 = x1, y1
            first_y1 = y1
        y4 = max(y4, y2)
        x4 = max(x4, x2)

        # Draw bounding box on the image (for visualization)
        color = (0, 255, 0)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Group the detected words into lines based on y-position similarity
        added_to_line = False
        for line in lines:
            _, existing_y1 = line[0][0][0]  # Get the y1 of the first word in the line
            _, existing_y2 = line[0][0][2]  # Get the y2 of the first word in the line

            # If y-range matches, consider the word part of the same line
            if (y1 >= existing_y1 - line_threshold and y1 <= existing_y2 + line_threshold) or \
               (y2 >= existing_y1 - line_threshold and y2 <= existing_y2 + line_threshold):
                line.append((bbox, text))
                added_to_line = True
                break

        if not added_to_line:
            lines.append([(bbox, text)])

    # Now, let's get the first word of each line and mark it
    first_words = []
    for line in lines:
        # Sort the line based on x1 (leftmost point)
        line_sorted = sorted(line, key=lambda b: b[0][0][0])
        first_word_bbox, first_word_text = line_sorted[0]
        first_words.append((first_word_bbox, first_word_text))

    # Draw bounding boxes for the first word in yellow
    for bbox, text in first_words:
        x1, y1 = bbox[0]  # Top-left corner of the first word's bbox
        x2, y2 = bbox[2]  # Bottom-right corner

        # Draw a yellow rectangle around the first word of the line
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

        # Optionally, print the bounding box coordinates
        print(f"First word: '{text}', Bounding box: {bbox}")

    # Draw vertical line along the y-axis from the first word of the first line
    y_plot1 = np.arange(y3, y4, 0.1)
    x_plot1 = x3
    for y_val in y_plot1:
        cv2.circle(image, (int(x_plot1), int(y_val)), 1, (0, 0, 255), -1)

    # Draw horizontal line along the x-axis
    x_plot2 = np.arange(x3, x4, 0.1)
    y_plot2 = y3
    for x_val in x_plot2:
        cv2.circle(image, (int(x_val), int(y_plot2)), 1, (0, 0, 255), -1)

    # Draw horizontal line at the bottom-most y-value
    for x_val2 in x_plot2:
        cv2.circle(image, (int(x_val2), int(y4)), 1, (0, 0, 255), -1)

    for y_val2 in y_plot1:
        cv2.circle(image, (int(x4), int(y_val2)), 1, (0, 0, 255), -1)
    # Display the image with the drawn annotations
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

else:
    print("No text detected in the image.")
