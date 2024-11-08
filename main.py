import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

image_path = r'Margin-Detection\images\Image_29.jpg'
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
bottom_margin_bbox = []  # To store bottom margin bounding boxes closest to the line

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
        y4 = max(y4, y2)  # Keep updating y4 to get the maximum y2 (bottom-most position)
        x4 = max(x4, x2)  # Keep updating x4 to get the rightmost x2
        
        # Left margin detection
        if (x3 - 250) <= x1 <= (x3 + 76):
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
    #left margin
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
        top_diff.append(x[0] - x_plot1)

    for x in mid:
        mid_diff.append(x[0] - x_plot1)

    for x in bottom:
        bottom_diff.append(x[0] - x_plot1)

    def remove_outliers(data):
        if not data:
            return [80]
        median = statistics.median(data)
        mad = statistics.median([abs(x - median) for x in data])
        threshold = 2.36 * mad
        filtered_data = [x for x in data if abs(x - median) <= threshold]
        return filtered_data if filtered_data else [80]    

    filtered_top = remove_outliers(top_diff)
    filtered_mid = remove_outliers(mid_diff)
    filtered_bottom = remove_outliers(bottom_diff)

    
    def list_avg(lst):
        return 80 if not lst else sum(lst) / len(lst)


    # Top Margin Filtering
    top_left = []
    top_right = []
    
    for x,y in top_margin_bbox[:]:
        if x <=n3_verticle:
            top_left.append([x,y])
        elif x> n3_verticle: 
            top_right.append([x,y])
        else:
            print(f"No bbox for bottom region at y={y}")
            pass
    top_left_diff =[]
    top_right_diff = []
    
    for x in top_left:
        top_left_diff.append(x[1] - y_plot2)
    for x in top_right: 
        top_right_diff.append(x[1] - y_plot2)
        
    print(top_left_diff)
    print(top_right_diff, "\n")
    
    top_left_filtered = remove_outliers(top_left_diff)
    top_right_filtered = remove_outliers(top_right_diff)

    print(top_left_filtered)
    print(top_right_filtered)
    
    # Bottom Margin Detection (Highlight Closest Bboxes in Yellow)
    threshold = 30  # Define a threshold for closeness to the bottom line (adjustable)

    for bbox, text, prob in results:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]

        # Check if y2 is within threshold distance from y4
        if abs(y2 - y4) <= threshold:
            # Highlight close bbox in yellow and add to bottom_margin_bbox list
            color = (0, 255, 255)  # Yellow
            bottom_margin_bbox.append(bbox)
        else:
            continue  # Skip if not close enough to the bottom line

        # Draw the rectangle around the detected bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Bottom Margin Filtering
    bottom_left = []
    bottom_right = []

    def bottom_remove_outliers(data):
        if not data:
            return []
        median = statistics.median(data)
        mad = statistics.median([abs(x - median) for x in data])
        threshold = 3 * mad
        filtered_data = [x for x in data if abs(x - median) <= threshold]
        return filtered_data if filtered_data else []

    for bbox in bottom_margin_bbox[:]:
        x2, y2 = bbox[2]
        if x2 <=n3_verticle:
            bottom_left.append([x2,y2])
        elif x2> n3_verticle: 
            bottom_right.append([x2,y2])
        else:
            print(f"No bbox for bottom region at y={y}")
            pass
    bottom_left_diff =[]
    bottom_right_diff = []

    for x in bottom_left:
        bottom_left_diff.append(x[1] - y4)
    for x in bottom_right: 
        bottom_right_diff.append(x[1] - y4)
        
    print("Bottom Diff")
    print(bottom_left_diff)
    print(bottom_right_diff, "\n")
    
    bottom_left_filtered = bottom_remove_outliers(bottom_left_diff)
    bottom_right_filtered = bottom_remove_outliers(bottom_right_diff)

    print("Bottom Diff Filtered")
    print(bottom_left_filtered)
    print(bottom_right_filtered)
    

    # OpenCV Display with Yellow Highlighted Bottom Margin
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    plt.tight_layout()
    plt.show()

    # Print bottom margin bounding boxes for reference
    print("Bottom Margin Bounding Boxes (closest to line):", bottom_margin_bbox)

else:
    print("No text detected in the image.")