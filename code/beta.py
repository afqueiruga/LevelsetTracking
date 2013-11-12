#!/usr/bin/python

import numpy as np
import cv2
import matplotlib.pylab as plt

#Open the file
cap = cv2.VideoCapture('small_block_movies/cylinder_horz.mov')

# Parameters for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.15,
                       minDistance = 30,
                       blockSize = 11 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )
TNUM = 2
def apply_filters(frame):
    clipped = frame[100:400,200:500]
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
    
    # cv2.Canny(blur,5,50)
    ret,thresh1 = cv2.threshold(gray,128,255,cv2.THRESH_TOZERO_INV)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(gray,kernel,iterations = 1)
    blur = cv2.GaussianBlur(erosion,(5,5),0)
    # thresh_refl = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                    cv2.THRESH_BINARY,11,2)

    kernel_cross = np.zeros((5,5),np.uint8)
    kernel_cross[:,2]=1; kernel_cross[2,:]=1;

    erosion_cross = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel_cross,iterations = 1)
    # erosion_cross = cv2.erode(gray,kernel_cross,iterations = 2)
    ehist = cv2.equalizeHist(gray)
    sobelx = cv2.Sobel(thresh1,cv2.CV_64F,2,0,ksize=7)
    sobely = cv2.Sobel(thresh1,cv2.CV_64F,0,2,ksize=7)

    # msobelx = maprange(sobelx)
    # msobely = maprange(sobely)
    # corners = msobelx & msobely
    
    # kernel1 = np.ones((11,11),np.uint8)
    # erosion1 = cv2.morphologyEx(erosion,cv2.MORPH_BLACKHAT,kernel1,iterations = 1)
    # kernel2 = np.ones((9,9),np.uint8)
    # erosion2 = cv2.morphologyEx(erosion,cv2.MORPH_BLACKHAT,kernel2,iterations = 1)
    kernel3 = np.ones((13,13),np.uint8)
    erosion3 = cv2.morphologyEx(erosion,cv2.MORPH_BLACKHAT,kernel3,iterations = 1)    

    mero3 = maprange(erosion3)


    # ret,thresh1 = cv2.threshold(erosion1,10,255,cv2.THRESH_BINARY)
    # ret,thresh2 = cv2.threshold(erosion2,10,255,cv2.THRESH_BINARY)
    ret,thresh3 = cv2.threshold(mero3,20,255,cv2.THRESH_BINARY)
    
    # lapl = cv2.Laplacian(erosion,cv2.CV_64F,ksize=5)
    # mapped = np.uint8( 255*(lapl-np.min(lapl))/(np.max(lapl)-np.min(lapl)) )
    # mapped = np.uint8( 255*(lapl.clip( 0.5*(np.max(lapl)+np.min(lapl)) ))/(np.max(lapl)) )
    # mapped = np.uint8( 255*((-1.0*lapl).clip( 0.5*(np.max(lapl)+np.min(lapl)) ))/(np.max(-1.0*lapl)) )

    # blur2 = cv2.GaussianBlur(mapped,(5,5),0)

    # thresh_refl2 = cv2.adaptiveThreshold(mapped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        # cv2.THRESH_BINARY,7,1)
    return [gray,thresh1,thresh3,mero3]

#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_frame = cv2.flip(old_frame,-1)


old_fils = apply_filters(old_frame)
p0 = cv2.goodFeaturesToTrack(old_fils[TNUM], mask=None, **feature_params)


hist = cv2.calcHist( [old_fils[0]],
                     channels=[0], 
                     mask=np.ones_like(old_fils[0]),
                     histSize=[16], 
                     ranges=[0,255] )
plt.plot(hist)
plt.show()

# Make a layer that we can draw pathlines on
color=np.random.randint(0,255,(100,3))
tracing = np.zeros_like(old_frame)
plt.ion()
resx,resy = old_fils[0].shape[0],old_fils[1].shape[1]
displayer = np.zeros((resx*2,resy*2,3),dtype=old_frame.dtype)
while 1:
    # Pull the new frame and apply our filtering to it.
    ret,frame=cap.read()
    frame = cv2.flip(frame,-1)

    fils = apply_filters(frame)
    # frame=fils[0]
    # Calculate the optical flow and grab good points to track
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_fils[TNUM],fils[TNUM], p0, None, **lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Draw the pathlines
    show = cv2.cvtColor(fils[TNUM],cv2.COLOR_GRAY2BGR)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(tracing, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(show, (a,b),5,color[i].tolist(),-1)
    # show = cv2.add(show,tracing)
    # Show the images
    displayer[0:resx,0:resy,:] = cv2.cvtColor(fils[0],cv2.COLOR_GRAY2BGR)
    displayer[resx:,0:resy,:] = show
    displayer[0:resx,resy:,:] = cv2.cvtColor(fils[2],cv2.COLOR_GRAY2BGR)
    displayer[resx:,resy:,:] = cv2.cvtColor(fils[3],cv2.COLOR_GRAY2BGR)
    
    cv2.imshow('frame',displayer)
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
    # old_fils = fils
    # p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
