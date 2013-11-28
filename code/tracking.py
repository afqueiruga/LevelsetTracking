#!/usr/bin/python

"""
High speed tracking using the level set method.

ME220 Project
Alejandro F Queiruga
UCB November 2013
"""


import numpy as np
import cv2
import matplotlib.pylab as plt

import levelset as LS


#
# Set up and Parameters
#
cap = cv2.VideoCapture('baseball/fast_deep.mov')
TNUM = 2
clipy = (0,-1)
clipx = (0,-1)
flipit = False

#
# Routine to rescale an image to grayscale.
#
maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )


def apply_filters(frame):
    """
    Apply all the filters we want.
    """
    # Extract region of interest
    clipped = frame[clipy[0]:clipy[1], clipx[0]:clipx[1]]
    # Convery to grayscale
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
    # Do a de-noisening step
    blur1 = cv2.GaussianBlur(gray,(3,3),0)
    # ret,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
    thresh1 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.THRESH_BINARY,15,1)
    
    return [gray,thresh1,thresh1,thresh1]



#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_frame = cv2.flip(old_frame,-1) if flipit else old_frame

old_fils = apply_filters(old_frame)

# Grab the resolution
resy,resx = old_fils[0].shape[0],old_fils[1].shape[1]
# Make a buffer to display a 2x2 grid of filters.
displayer = np.zeros((resy*2,resx*2,3),dtype=old_frame.dtype)

video = cv2.VideoWriter('video.mjpeg',cv2.cv.CV_FOURCC('M','J','P','G'),
                        1,(2*resx,2*resy),1)

print video.isOpened()

#
# Loop through the movie
#
while 1:
    # Pull the new frame and apply our filtering to it.
    ret,frame=cap.read()
    if frame==None:
        break
    frame = cv2.flip(frame,-1)  if flipit else frame
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
    video.write(displayer)

# Clean up
cv2.destroyAllWindows()
cap.release()
video.release()
