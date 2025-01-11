import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

image_path = r'D:\Margin-Detection\new images\Image_117.jpg'
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

    print("Left Margin:\nUnfiltered:")
    print("\nTop: ",top_diff)
    print("\nMid: ",mid_diff)
    print("\nBottom: ",bottom_diff)
    print("\nFiltered Data: \n")
    print("Top: ",filtered_top)
    print("\nMid: ",filtered_mid)
    print("\nBottom: ",filtered_bottom)
    def list_avg(lst):
        return int(80) if not lst else sum(lst) / len(lst)


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
        top_left_diff.append(int(x[1] - y_plot2))
    for x in top_right: 
        top_right_diff.append(int(x[1] - y_plot2))
        
    
    top_left_filtered = remove_outliers(top_left_diff)
    top_right_filtered = remove_outliers(top_right_diff)

    threshold = 33 
    for bbox, text, prob in results:
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]
        if abs(y2 - y4) <= threshold:
            color = (0, 255, 255)  
            bottom_margin_bbox.append(bbox)
        else:
            continue 
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    print("\nTop Margins:\nUnfiltered: \n")
    print("Left: ",top_left_diff)
    print("\n Right: ",top_right_diff)
    print("\nFiltered: \n")
    print("Left: ",top_left_filtered)
    print("\nRight: ",top_right_filtered)

# Bottom Margin Filtering
    bottom_left = []
    bottom_right = []

    def bottom_remove_outliers(data):
        if not data:
            return []
        median = statistics.median(data)
        mad = statistics.median([abs(x - median) for x in data])
        threshold = 2.7 * mad
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
        bottom_left_diff.append(int(x[1] - y4))
    for x in bottom_right: 
        bottom_right_diff.append(int(x[1] - y4))
        

    bottom_left_filtered = bottom_remove_outliers(bottom_left_diff)
    bottom_right_filtered = bottom_remove_outliers(bottom_right_diff)

    print("\nBottom Margin:\nUnfiltered: \n")
    print("Left: ",bottom_left_diff)
    print("\nRight: ",bottom_right_diff)
    print("\n Filtered: \n")
    print("Left: \n",bottom_left_filtered)
    print("\nRight: ",bottom_right_filtered)

    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    plt.tight_layout()
    plt.show()

   
else:
    print("No text detected in the image.")
