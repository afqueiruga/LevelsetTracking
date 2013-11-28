#!/usr/bin/python

#
# New strategy to find the grid: Look for the squares, not the lines
# using segmentation techniques
#

import numpy as np
import cv2
import matplotlib.pylab as plt

import levelset as LS

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

w_cross = 11
kern_cross = np.zeros((w_cross,w_cross),np.uint8)
kern_cross[:,w_cross/2]=1; kern_cross[w_cross/2,:]=1;

#
# Routine to apply the set of filters we want to a frame.
#
def apply_filters(frame):
    # Extract region of interest
    clipped = frame[clipy[0]:clipy[1], clipx[0]:clipx[1]]
    # Convery to grayscale
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
    #
    # First, lets identify groups of LIGHT AREAS
    #
    # Do a de-noisening step
    blur1 = cv2.GaussianBlur(gray,(5,5),0)
    # ret,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
    thresh1 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY,15,1)
    # Close the noise
    kern_erode = np.ones((3,3),np.uint8)

    # erode_close = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kern_erode,iterations = 1)
    blur2 = cv2.GaussianBlur(thresh1,(13,13),0)
    ret,thresh2 = cv2.threshold(blur2,200,255,cv2.THRESH_BINARY)

    # Perform one erosion iteration to close shines and grow the lines
    erosion = cv2.erode(thresh2,kern_cross,iterations = 1)

    dist_transform = cv2.distanceTransform(thresh1,cv2.cv.CV_DIST_L2,5)
    lev = LS.levelset(dist_transform)
    mdt = maprange(lev)
    
    ero2 = erosion.copy()
    # ret, markers = cv2.connectedComponents(erosion)
    drawing = np.zeros_like(gray)
    contours,hierarchy = cv2.findContours(ero2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        color = 255 #np.random.randint(0,255,(3)).tolist()  # Select a random color
        cv2.drawContours(drawing,[cnt],0,color,2)

        
    return [thresh1,thresh2,erosion,mdt,contours,hierarchy]



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
    if frame==None:
        break
    frame = cv2.flip(frame,-1)
    fils = apply_filters(frame)

    # 
    # Display the frames
    #
    # Convert to color
    for i in xrange(4):
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
