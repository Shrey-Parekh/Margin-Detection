import csv
import easyocr
import cv2
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from main import *
from top_bottom import *
import os

# Initialize variables
slm = False
wlm, daflm, rlm, cclm, cvlm = False, False, False, False, False

# Function to check SLM condition
def check_slm():
    filtered_list = filtered_top + filtered_mid + filtered_bottom
    for value in filtered_list:
        if (x_plot1 - 6) <= value <= (x_plot1 + 6):
            return True
    return False

# Calculate SLM, WLM, DAFLM, RFLM, CCLM, CVLM
while True:
    slm = check_slm()
    avg_top = st.mean(filtered_top)
    avg_mid = st.mean(filtered_mid)
    avg_bottom = st.mean(filtered_bottom)
    if slm:
        break
    if avg_top < avg_mid < avg_bottom:
        daflm = True
        break
    elif avg_top > avg_mid > avg_bottom:
        rlm = True
        break
    elif avg_top < avg_mid > avg_bottom:
        cclm = True
        break
    elif avg_top > avg_mid < avg_bottom:
        cvlm = True
        break
    else:
        wlm = True
        break

# Initialize top and bottom margin variables
tns, tps, tls = False, False, False  # Top margin
bns, bps, bls = False, False, False  # Bottom margin
tda, tt, ts = False, False, False    # Top direction
bda, bt, bs = False, False, False    # Bottom direction

# Evaluate top margin gradient
while True:
    if top_line_gradient < -0.009:
        tda = True
        break
    elif top_line_gradient > 0.009:
        tt = True
        break
    else:
        ts = True
        break

# Evaluate bottom margin gradient
while True:
    if bottom_line_gradient < -0.009:
        bda = True
        break
    elif bottom_line_gradient > 0.009:
        bt = True
        break
    else:
        bs = True
        break

# Save results to CSV
csv_headers = ["SLM", "WLM", "DAFLM", "RFLM", "CCLM", "CVLM", 
               "TT", "TS", "TDA", "BT", "BS", "BDA"]
csv_values = [int(slm), int(wlm), int(daflm), int(rlm), int(cclm), int(cvlm), 
              int(tt), int(ts), int(tda), int(bt), int(bs), int(bda)]

csv_filename = "Auto_margins.csv"
file_exists = os.path.isfile(csv_filename)

# Write to the CSV file without extra blank lines
with open(csv_filename, mode="a", newline='') as file:
    writer = csv.writer(file, lineterminator='\n')
    if not file_exists:
        writer.writerow(csv_headers)
    writer.writerow(csv_values)

print(f"\nResults appended to {csv_filename}")
