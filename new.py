import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

reader = easyocr.Reader(['en'])

image_path = r'C:\Users\Shrey Parekh\Documents\Margin detection\Margin-Detection\images\Image_4.jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Image not found at the path: {image_path}")

results = reader.readtext(image)
height, width, _ = image.shape

min_x, min_y = width, height
max_x, max_y = 0, 0
x3, y3 = None, None
y4 = 0
first_y1 = None

if results:
    for idx, (bbox, text, prob) in enumerate(results):
        x1, y1 = bbox[0]
        x2, y2 = bbox[2]

        if idx == 0:
            x3, y3 = x1, y1
            first_y1 = y1

        y4 = max(y4, y2)
        
        if idx == 0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    if first_y1 is not None:
        y = np.arange(first_y1, y4, 0.1)
        x = x3
        for y_val in y:
            cv2.circle(image, (int(x), int(y_val)), 1, (0, 0, 255), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

else:
    print("No text detected in the image.")
