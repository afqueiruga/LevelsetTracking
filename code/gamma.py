#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pylab as plt


#
# Set up and Parameters
#
cap = cv2.VideoCapture('small_block_movies/cylinder_horz.mov')
TNUM = 2
clipy = (100,300)
clipx = (200,500)


#
# Routine to rescale an image to grayscale.
#
maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )


kern_cross = np.zeros((13,13),np.uint8)
kern_cross[:,3]=1; kern_cross[3,:]=1;

#
# Routine to apply the set of filters we want to a frame.
#
def apply_filters(frame):
    # Extract region of interest
    clipped = frame[clipy[0]:clipy[1], clipx[0]:clipx[1]]
    # Convery to grayscale
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)

    # --OR-- Blur and threshhold
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret,thresh_shine = cv2.threshold(blur,178,255,cv2.THRESH_TRUNC)
    # Perform one erosion iteration to close shines and grow the lines
    kern_erode = np.ones((7,7),np.uint8)
    erosion = cv2.erode(thresh_shine,kern_erode,iterations = 1)

    ret,thresh_ero = cv2.threshold(erosion,140,255,cv2.THRESH_BINARY)

    
    # Perform a blackhat transform
    kern_blackhat = np.ones((13,13),np.uint8)
    erode_blackhat = cv2.morphologyEx(thresh_shine,cv2.MORPH_BLACKHAT,kern_blackhat,iterations = 1)
    # Rescale the range
    mero = maprange(erode_blackhat)
    # Threshhold it
    ret,thresh_mero = cv2.threshold(mero,25,255,cv2.THRESH_BINARY)

    dilate_thresh = cv2.dilate(thresh_mero,kern_cross,iterations = 1)
    
    dist_transform = cv2.distanceTransform(dilate_thresh,cv2.cv.CV_DIST_L1,5)
    mdt = maprange(dist_transform)

    # isolate = mdt & erode_thresh
    
    return [thresh_shine,erosion,thresh_ero,mdt]


#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_frame = cv2.flip(old_frame,-1)

old_fils = apply_filters(old_frame)

# Grab the resolution
resy,resx = old_fils[0].shape[0],old_fils[1].shape[1]
# Make a buffer to display a 2x2 grid of filters.
displayer = np.zeros((resy*2,resx*2,3),dtype=old_frame.dtype)


#
# Loop through the movie
#
while 1:
    # Pull the new frame and apply our filtering to it.
    ret,frame=cap.read()
    frame = cv2.flip(frame,-1)
    fils = apply_filters(frame)

    # 
    # Display the frames
    #
    # Convert to color
    for i in xrange(len(fils)):
        fils[i] = cv2.cvtColor(fils[i],cv2.COLOR_GRAY2BGR)
    # Copy onto the window frame
    displayer[0:resy,0:resx,:] = fils[0]
    displayer[resy:,0:resx,:] = fils[1]
    displayer[0:resy,resx:,:] = fils[2]
    displayer[resy:,resx:,:] = fils[3]
    # Display
    cv2.imshow('frame',displayer)
    # Do some event handling even though I just ^C it...
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


# Clean up
cv2.destroyAllWindows()
cap.release()
