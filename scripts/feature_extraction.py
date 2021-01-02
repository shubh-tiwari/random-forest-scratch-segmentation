"""Feature extraction of image for training ML models"""

import os
import cv2
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from skimage.filters import sobel, scharr, roberts, prewitt
from skimage.feature import canny
from scipy import ndimage as nd

# Extract the features using different gabor kernels
def addGabor(df,img):
    num = 1
    kernels = []
    ksize = 9
    
    for theta in range(2):
        theta = theta/4.*np.pi
        for sigma in (1,3):
            for lamda in np.arange(0,np.pi,np.pi/4):
                for gamma in (0.05,0.5):
                    gabor_label = "gabor" + str(num)
                    kernel = cv2.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,0,cv2.CV_32F)
                    kernels.append(kernel)
                    fimg = cv2.filter2D(img,cv2.CV_8UC3,kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img
                    num += 1
                    
    return df

# Extract the features using different edge detector methods
def addEdges(df,gray):
    canny_edges = canny(gray,0.6)
    roberts_edges = roberts(gray)
    sobel_edges = sobel(gray)
    scharr_edges = scharr(gray)
    prewitt_edges = prewitt(gray)
    df['canny_edges'] = canny_edges.reshape(-1)
    df['roberts_edge'] = roberts_edges.reshape(-1)
    df['sobel_edge'] = sobel_edges.reshape(-1)
    df['scharr_edge'] = scharr_edges.reshape(-1)
    df['prewitt_edge'] = prewitt_edges.reshape(-1)
    
    return df
    
# Extract feutures using gaussian and median filters
def addFilter(df,gray):
    gaussian_3 = nd.gaussian_filter(gray,sigma=3)
    gaussian_7 = nd.gaussian_filter(gray,sigma=7)
    median_img = nd.median_filter(gray,size=3)
    df['gaussian_3'] = gaussian_3.reshape(-1)
    df['gaussian_3'] = gaussian_7.reshape(-1)
    df['gaussian_3'] = median_img.reshape(-1)
    
    return df

# Add ground truth of the given input image
def addTruthLabel(df,truthfile):
    gd = cv2.imread(truthfile,0)
    df['ground_truth'] = gd.reshape(-1)
    return df

# Add all the feautes in the dataframe
def calc_features(imgfile):
    df = pd.DataFrame([])
    img = cv2.imread(imgfile)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img1 = np.reshape(gray,(-1))
    
    df['pixels'] = img1
    df = addGabor(df,img1)
    df = addEdges(df,gray)
    df = addFilter(df,gray)
    
    return df

imgpath = r"C:\Users\shubh\Documents\steel_detection\metal_nut\test\scratch"
gdpath = r"C:\Users\shubh\Documents\steel_detection\metal_nut\ground_truth\scratch"

imglist = ["021.png", "022.png"]
gdlist = ["021_mask.png", "021_mask.png"]

imgfiles = [os.path.join(imgpath,file) for file in imglist]
gdfiles = [os.path.join(gdpath,file) for file in gdlist]

df = pd.DataFrame([])
for i in range(len(imgfiles)):
    if df.empty:
        df = calc_features(imgfiles[i])
        df = addTruthLabel(df,gdfiles[i])
    else:
        dftemp = calc_features(imgfiles[i])
        dftemp = addTruthLabel(dftemp,gdfiles[i])
        df = pd.concat([df,dftemp])

# print(df.head())
