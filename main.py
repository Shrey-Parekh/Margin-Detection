import easyocr
import cv2
import numpy as np
import statistics
import matplotlib.pyplot as plt


reader = easyocr.Reader(['en'])

image_path = r'C:\Users\Shrey Parekh\Documents\Margin-Detection\images\Image_7.jpg'
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
        if id == 0:
            x3, y3 = x1, y1
            first_y1 = y1
        y4 = max(y4, y2)
        x4 = max(x4, x2)

        if ((x3 - 250) <= x1 <= (x3 + 150)) and (last_y1 is None or abs(y1 - last_y1) > 30):
            first_word_midpoint.append((x1, y_midpoint))
        last_y1 = y1

        if (x3 - 250) <= x1 <= (x3 + 150):
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    y_plot1 = np.arange(y3, y4, 0.1)
    
    def get_split_indices(lst):
        length = len(lst)
        index1 = length // 3        
        index2 = 2 * length // 3    
        return index1, index2
    
    n1, n2 = get_split_indices(y_plot1)
    
    n1_list = y_plot1[n1]
    n2_list = y_plot1[n2]
            
    x_plot1 = x3
    for y_val in y_plot1:
        cv2.circle(image, (int(x_plot1), int(y_val)), 2, (0, 0, 255), -1)

    for x, y_midpoint in first_word_midpoint:
        if (x_plot1 - 250) <= x <= (x_plot1 + 250):
            cv2.line(image, (int(x_plot1), int(y_midpoint)), (int(x), int(y_midpoint)), (0, 0, 0), 3)
            x_diff.append(x - x_plot1)

    x_plot2 = np.arange(x3, x4, 0.1)
    y_plot2 = y3
    for x_val in x_plot2:
        cv2.circle(image, (int(x_val), int(y_plot2)), 2, (0, 0, 255), -1)
    
    for x_val2 in x_plot2:
        cv2.circle(image, (int(x_val2), int(y4)), 2, (0, 0, 255), -1)
        
    for x_val3 in x_plot2:
        cv2.circle(image, (int(x_val3), int(n1_list)), 2, (209, 26, 255), -1)
    for x_val4 in x_plot2:
        cv2.circle(image, (int(x_val4), int(n2_list)), 2, (209, 26, 255), -1)
        
    for y_val2 in y_plot1:
        cv2.circle(image, (int(x4), int(y_val2)), 2, (0, 0, 255), -1)

    midpoint_list = [list(t) for t in first_word_midpoint]

    top = []
    mid = []
    bottom = []

    for x,y in midpoint_list[1:]:
        if y <= n1_list:
            top.append([x, y])
        elif y <= n2_list and y > n1_list:
            mid.append([x, y])
        elif y > n2_list:
            bottom.append([x, y])
        else: 
            continue

    top_diff = []
    mid_diff = []
    bottom_diff = []
    
    for x in top:
        top_diff.append(x[0] - x_plot1)
        
    for x in mid:
        mid_diff.append(x[0] - x_plot1)
        
    for x in bottom:
        bottom_diff.append(x[0] - x_plot1)

    print("\nUn-filtered Data:\n")    
    print("top:\t", top_diff)
    print("mid:\t", mid_diff)
    print("bottom:\t", bottom_diff)

    def remove_outliers(data):
        median = statistics.median(data)
        mad = statistics.median([abs(x - median) for x in data])
        threshold = 2.35 * mad
        filtered_data = [x for x in data if abs(x - median) <= threshold]
        return filtered_data
    
    filtered_top = remove_outliers(top_diff)
    filtered_mid = remove_outliers(mid_diff)
    filtered_bottom = remove_outliers(bottom_diff)
    
    print("\nFiltered Data:\n")
    print(filtered_top)
    print(filtered_mid)
    print(filtered_bottom)
    
    
    
    def list_avg(lst):
        return sum(lst) / len(lst) if len(lst) > 0 else 0
    
    top_avg = list_avg(filtered_top)
    mid_avg = list_avg(filtered_mid)
    bottom_avg = list_avg(filtered_bottom)
    
    print("\nAverage of list: \n")
    print(top_avg, "\t", mid_avg, "\t", bottom_avg,"\n")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.axis('on')

    plt.tight_layout()
    plt.show()

else:
    print("No text detected in the image.")
