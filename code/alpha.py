import numpy as np
import cv2

cap = cv2.VideoCapture('small_block_movies/cylinder_vert.mov')


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
# print old_frame

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# ret,old_thresh = cv2.threshold(old_gray,140,255,cv2.THRESH_TOZERO)
# old_thresh = cv2.adaptiveThreshold(old_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                                    cv2.THRESH_BINARY,5,1)
# old_thresh = cv2.Laplacian(old_gray,cv2.CV_64F)
old_blur = cv2.GaussianBlur(old_gray,(10,10),0)
old_thresh = cv2.Canny(old_blur,5,50)
p0 = cv2.goodFeaturesToTrack(old_thresh, mask = None, **feature_params)


mask = np.zeros_like(old_frame)
# print mask
while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(frame_gray,140,255,cv2.THRESH_TOZERO)
    # thresh = cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                cv2.THRESH_BINARY,5,2)
    # thresh = cv2.Laplacian(frame_gray,cv2.CV_64F)
    blur = cv2.GaussianBlur(frame_gray,(5,5),0)
    thresh = cv2.Canny(blur,5,50)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_thresh, thresh, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    threshcolor=cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # print i
        # print mask
        cv2.circle(threshcolor,(a,b),5,color[i].tolist(),-1)
        # print frame

    img = cv2.add(threshcolor,mask)
    img2=cv2.flip(img,-1)
    cv2.imshow('frame',img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
