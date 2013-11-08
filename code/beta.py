#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pylab as plt

#Open the file
cap = cv2.VideoCapture('small_block_movies/cylinder_vert.mov')

# Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def apply_filters(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flipped = cv2.flip(gray,-1)
    blur = cv2.GaussianBlur(flipped,(5,5),0)
    thresh = cv2.Canny(blur,5,50)
    return [gray,flipped,blur,thresh]

#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_fils = apply_filters(old_frame)
p0 = cv2.goodFeaturesToTrack(old_fils[1], mask=None, **feature_params)

# Make a layer that we can draw pathlines on
color=np.random.randint(0,255,(100,3))
tracing = np.zeros_like(old_frame)
plt.ion()
while 1:
    # Pull the new frame and apply our filtering to it.
    ret,frame=cap.read()
    fils = apply_filters(frame)
    frame=fils[0]
    # Calculate the optical flow and grab good points to track
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_fils[-1],fils[-1], p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Draw the pathlines
    fils[1] = cv2.cvtColor(fils[1],cv2.COLOR_GRAY2BGR)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(tracing, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(fils[1], (a,b),5,color[i].tolist(),-1)
    show = cv2.add(fils[1],tracing)
    # Show the images
    cv2.imshow('frame',fils[1])
    # plt.clf()
    # for i in xrange(4):
        # plt.subplot(2,2,i+1)
        # plt.imshow(fils[i])
        # plt.xticks([]),plt.yticks([])
    # plt.show()
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Save the state
    old_fils = fils
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
