#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pylab as plt


#
# Set up and Parameters
#
cap = cv2.VideoCapture('small_block_movies/cylinder_horz.mov')
TNUM = 2
clipy = (100,400)
clipx = (200,500)


#
# Routine to rescale an image to grayscale.
#
maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )


#
# Routine to apply the set of filters we want to a frame.
#
def apply_filters(frame):
    # Extract region of interest
    clipped = frame[clipy[0]:clipy[1], clipx[0]:clipx[1]]
    # Convery to grayscale
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
    # Perform a blackhat transform
    kernel = np.ones((13,13),np.uint8)
    erosion = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel,iterations = 1)    
    # Rescale the range
    mero = maprange(erosion)
    # Threshhold it
    ret,thresh = cv2.threshold(mero,20,255,cv2.THRESH_BINARY)
    return [gray,erosion,mero,thresh]


#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_frame = cv2.flip(old_frame,-1)

old_fils = apply_filters(old_frame)

# Grab the resolution
resy,resx = old_fils[0].shape[0],old_fils[1].shape[1]
# Make a buffer to display a 2x2 grid of filters.
displayer = np.zeros((resx*2,resy*2,3),dtype=old_frame.dtype)


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
    displayer[0:resx,0:resy,:] = fils[0]
    displayer[resx:,0:resy,:] = fils[1]
    displayer[0:resx,resy:,:] = fils[2]
    displayer[resx:,resy:,:] = fils[3]
    # Display
    cv2.imshow('frame',displayer)
    # Do some event handling even though I just ^C it...
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


# Clean up
cv2.destroyAllWindows()
cap.release()
