import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

image_path = r'images\Image_79.jpg'
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


    # for y_val2 in y_plot1: 
    #     cv2.circle(image, (int(n3_verticle), int(y_val2)), 2, (255,0,100), -1)
        
# Left Margin Filtering
    top = []
    mid = []
    bottom = []
    
    #margin seperation lines
    cv2.line(image, (0, int(n1_horizontal)), (width, int(n1_horizontal)), (0, 0, 0), 2)
    cv2.line(image, (0, int(n2_horizontal)), (width, int(n2_horizontal)), (0, 0, 0), 2)

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

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    plt.tight_layout()
    plt.show()

   
else:
    print("No text detected in the image.")
    
    
    
    
    
