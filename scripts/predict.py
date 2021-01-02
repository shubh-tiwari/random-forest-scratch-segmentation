"""Predicting and segmenting scratch from the image"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import calc_features

imgpath = r"C:\Users\shubh\Documents\steel_detection\metal_nut\test\scratch"
imgfile = os.path.join(imgpath,"018.png")

img = cv2.imread(imgfile, 0)

# Feature extraction
X_test = calc_features(imgfile)

# Loading model to predict region of interest
modelname = 'scratch_detect_model'
load_model = pickle.load(open(modelname,'rb'))
y_test = load_model.predict(X_test)

# Reshaping 1d pixel data into 2d image
y_img = np.reshape(y_test, (img.shape[0],img.shape[1]))

#plt.imshow(y_img, cmap='gray')
#plt.show()

# Writing image to output sample directory
outdir = os.path.join(os.getcwd(),"output_samples")
if not os.path.isdir(outdir):
    os.mkdir(outdir)
    
cv2.imwrite(os.path.join(outdir,"018_predicted_truth.png"), y_img)

kernel = np.ones((3,3),np.uint8)
opening1 = cv2.morphologyEx(y_img, cv2.MORPH_OPEN, kernel)

# Cleaning of the predicted output for small noises morphological opertaions
kernel = np.ones((3,1),np.uint8)
erosion = cv2.erode(opening1,kernel,iterations = 1)

kernel = np.ones((3,3),np.uint8)
opening2 = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

kernel = np.ones((35,35),np.uint8)
dilation = cv2.dilate(opening2,kernel,iterations = 1)

# Writing cleaned output in the output sample directory
cv2.imwrite(os.path.join(outdir,"018_final_truth.png"), dilation)

# Drawing rectangle over region of interest
contours, _ = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
for c in contours:
    rect = cv2.boundingRect(c)
    x,y,w,h = rect
    img2 = cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0), 5)

# Writing image with highlighted ROI in the output sample directory
cv2.imwrite(os.path.join(outdir,"018_output_scratch.png"), img2)