import os
import glob
import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Directory containing images
image_dir = r'D:\Margin-Detection\new images'
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

if not image_paths:
    raise ValueError(f"No images found in directory: {image_dir}")

# Function to process a single image
def process_image(image_path):
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
    x_diff = []

    left_margin_bbox = []
    top_margin_bbox = []
    bottom_margin_bbox = []

    if results:
        last_y1 = None
        for id, (bbox, text, prob) in enumerate(results):
            x1, y1 = bbox[0]
            x2, y2 = bbox[2]
            y_midpoint = (y1 + y2) / 2
            x_midpoint = (x1 + x2) / 2  
            if id == 0:
                x3, y3 = x1, y1
                first_y1 = y1
            y4 = max(y4, y2) 
            x4 = max(x4, x2)  
            
            # Left margin detection
            if (x3 - 250) <= x1 <= (x3 + 75):
                color = (255, 0, 0)  
                left_margin_bbox.append([x1, y_midpoint])
            elif (y3 - 250) <= y1 <= (y3 + 35):
                color = (0, 0, 0)  # top margin
                top_margin_bbox.append([x2, y1])
            else:
                continue

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        y_plot1 = np.arange(y3, y4, 0.1)

        def get_split_2indices(lst):
            length = len(lst)
            index1 = length // 3       
            index2 = 2 * length // 3    
            return index1, index2
        
        def split_list(lst):
            length = len(lst)
            index1 = length // 2   
            return index1

        n1, n2 = get_split_2indices(y_plot1)

        n1_horizontal = y_plot1[n1]
        n2_horizontal = y_plot1[n2]

        x_plot1 = x3
        # left margin
        for y_val in y_plot1:
            cv2.circle(image, (int(x_plot1), int(y_val)), 2, (0, 0, 255), -1)

        x_plot2 = np.arange(x3, x4, 0.1)
        y_plot2 = y3
        for x_val in x_plot2:
            cv2.circle(image, (int(x_val), int(y_plot2)), 2, (0, 0, 255), -1)
            cv2.circle(image, (int(x_val), int(y4)), 2, (0, 0, 255), -1)
            
        for y_val2 in y_plot1:
            cv2.circle(image, (int(x4), int(y_val2)), 2, (255, 0, 255), -1)

        n3 = split_list(x_plot2)
        n3_verticle = x_plot2[n3]

        for y_val2 in y_plot1: 
            cv2.circle(image, (int(n3_verticle), int(y_val2)), 2, (255,0,100), -1)
            
        # Left Margin Filtering
        top = []
        mid = []
        bottom = []

        for x, y in left_margin_bbox[1:]:
            if y <= n1_horizontal:
                top.append([x, y])
            elif y <= n2_horizontal and y > n1_horizontal:
                mid.append([x, y])
            elif y > n2_horizontal:
                bottom.append([x, y])
            else:
                print(f"No bbox for bottom region at y={y}")
                pass

        top_diff = []
        mid_diff = []
        bottom_diff = []

        for x in top:
            top_diff.append(int(x[0] - x_plot1))

        for x in mid:
            mid_diff.append(int(x[0] - x_plot1))

        for x in bottom:
            bottom_diff.append(int(x[0] - x_plot1))

        def remove_outliers(data):
            if not data:
                return [int(80)]
            median = statistics.median(data)
            mad = statistics.median([abs(x - median) for x in data])
            threshold = 2.35 * mad
            filtered_data = [x for x in data if abs(x - median) <= threshold]
            return filtered_data if filtered_data else [int(80)]    

        filtered_top = remove_outliers(top_diff)
        filtered_mid = remove_outliers(mid_diff)
        filtered_bottom = remove_outliers(bottom_diff)

        # The rest of your code for filtering and processing margins remains unchanged...
        # ...

    return image

# Process all images
annotated_images = [process_image(image_path) for image_path in image_paths]

# Display all results in a grid
num_images = len(annotated_images)
cols = 4
rows = (num_images + cols - 1) // cols  # Dynamically calculate rows
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))

for idx, image in enumerate(annotated_images):
    ax = axes[idx // cols, idx % cols]
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    ax.set_title(f"Image {idx + 1}")

# Turn off unused subplots
for ax in axes.ravel()[len(annotated_images):]:
    ax.axis('off')

plt.tight_layout()
plt.show()
