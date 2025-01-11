import easyocr
import cv2
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
from main import *

slm = False
wlm, daflm, rlm, cclm, cvlm = False, False, False, False, False

def check_slm():
    filtered_list = filtered_top+filtered_mid+filtered_bottom
    for value in filtered_list:
        if (x_plot1 - 6)<=value<=(x_plot1 + 6):
            return True
    return False

while(True):
    slm = check_slm()
    avg_top = st.mean(filtered_top)
    avg_mid = st.mean(filtered_mid)
    avg_bottom = st.mean(filtered_bottom)
    if slm:
        break
    if(avg_top<avg_mid<avg_bottom):
        daflm = True
        break
    elif(avg_top>avg_mid>avg_bottom):
        rlm = True
        break
    elif(avg_top<avg_mid>avg_bottom):
        cclm = True
        break
    elif(avg_top>avg_mid<avg_bottom):
        cvlm = True
        break
    else:
        wlm = True
        break
print(filtered_top)
print(filtered_mid)
print(filtered_bottom)
print(f"SLM: {slm}\nWLM: {wlm}\nDAFLM: {daflm}\nRFLM: {rlm}\nCCLM: {cclm}\nCVLM: {cvlm}")

#TOP,Bottom MARGIN

tns = False #T no space
tps = False #T perfect space
tls = False #T large space
bns = False #B no space
bps = False #B Perfect space
bls = False #B large space





