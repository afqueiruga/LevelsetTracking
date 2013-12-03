#!/usr/bin/python

"""
High speed tracking using the level set method.

ME220 Project
Alejandro F Queiruga
UCB November 2013
"""

import copy
import cPickle as pickle

import numpy as np
import cv2
import matplotlib.pylab as plt

import levelset as LS
from detect_local_minima import detect_local_minima

#
# Set up and Parameters
#


# The corner parameters, CCW from top left
#cap = cv2.VideoCapture('baseball/fast_deep.mov')
#corners = [ [20,70], [80,70], [80,150], [20,150] ]
#clipy = (0,-1)
#clipx = (75,-1)
#NY,NX = 4,5
cap = cv2.VideoCapture('medium_block_movies/static_small_end.mov')
corners = [ [70,110], [180,110], [180,180], [70,180] ]
clipy = (80,-120)
clipx = (200,-140)
NY,NX = 6,5

points = [ [ (30,70) ] ]


TNUM = 2
flipit = False
writevid = True

# Search radius, i.e. max a point can move
radius = 5


#
# Routine to rescale an image to grayscale.
#
maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )

w_cross = 5
kern_cross = np.zeros((w_cross,w_cross),np.uint8)
kern_cross[:,w_cross/2]=1; kern_cross[w_cross/2,:]=1;


def apply_filters(frame):
    """
    Apply all the filters we want.
    """
    # Extract region of interest
    clipped = frame[clipy[0]:clipy[1], clipx[0]:clipx[1]]
    # Convery to grayscale
    gray = cv2.cvtColor(clipped, cv2.COLOR_BGR2GRAY)
    # Do a de-noisening step
    blur1 = cv2.GaussianBlur(gray,(5,5),0)
    # ret,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)
    # Apative threshhold
    thresh1 = cv2.adaptiveThreshold(blur1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                        cv2.THRESH_BINARY,35,2)
    
    # Perform one erosion iteration to close shines and grow the lines
    #erosion = cv2.erode(thresh1,kern_cross,iterations = 1)

    # Do a levelset filter
    #threshls,phi = LS.levelsetPhase(np.array(gray,dtype=np.double),[[-0.5,2.5,50]])
    phi1 = LS.makephi(thresh1)
    threshls,phi = LS.levelsetPhase(phi1,[[-1.0,0.0,30],[0.0,3.0,15]])
    phi2gray = maprange(phi)
    #thresh2 = cv2.adaptiveThreshold(phi2gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #                                    cv2.THRESH_BINARY,31,3)
    lvl2 = LS.makephi(threshls)
    
    lvl22gray = maprange(lvl2)
    return [gray,threshls,lvl22gray,thresh1, lvl2]


def find_new_minimum(lvl, pt, rad):
    """
    Look for a new minimum in radius from pt on the lvl.
    """
    limy = (max(int(pt[0]-rad),0),min(int(pt[0]+rad),lvl.shape[0]))
    limx = (max(int(pt[1]-rad),0),min(int(pt[1]+rad),lvl.shape[1]) )
    view = lvl[limy[0]:limy[1], limx[0]:limx[1]]

    loc = np.argmin(view)
    uri = np.unravel_index(loc,view.shape)
    return (limy[0]+uri[0],limx[0]+uri[1])


def update_points(lvl,points,rad):
    """
    Go through the list of points and find new locations.
    """
    for x in xrange(len(points)):
        for y in xrange(len(points[x])):
            points[x][y] = find_new_points_from_trials(lvl,points[x][y],rad)
    return points


def locate_initial_points(corners,ny,nx):
    """
    Find the first batch of points.
    """
    xtop = np.linspace(corners[0][0],corners[1][0],ny)
    ytop = np.linspace(corners[0][1],corners[1][1],ny)
    xbot = np.linspace(corners[3][0],corners[2][0],ny)
    ybot = np.linspace(corners[3][1],corners[2][1],ny)
    
    points = []
    for i in xrange(ny):
        xs = np.linspace(xtop[i],xbot[i],nx)
        ys = np.linspace(ytop[i],ybot[i],nx)
        points.append( zip(xs,ys) )
    return points


def draw_points(img, points):
    for x in xrange(len(points)):
        for y in xrange(len(points[x])-1):
            cv2.line(img, (int(points[x][y][1]),int(points[x][y][0])),
                     (int(points[x][y+1][1]),int(points[x][y+1][0])),
                     (0,0,255),1)
    for x in xrange(len(points)-1):
        for y in xrange(len(points[x])):
            cv2.line(img, (int(points[x][y][1]),int(points[x][y][0])),
                     (int(points[x+1][y][1]),int(points[x+1][y][0])),
                     (0,0,255),1)
    for x in xrange(len(points)):
        for y in xrange(len(points[x])):
            cv2.circle(img,(int(points[x][y][1]),int(points[x][y][0])),
                       2,(255,255,0),-1)
    

def find_possible_points(lvl, pt, rad):
    limy = (max(int(pt[0]-rad),0),min(int(pt[0]+rad),lvl.shape[0]))
    limx = (max(int(pt[1]-rad),0),min(int(pt[1]+rad),lvl.shape[1]) )
    view = lvl[limy[0]:limy[1], limx[0]:limx[1]]
    
    
    neighborret = detect_local_minima(view)
    vals = view[neighborret]
    mins=np.where(vals == vals.min())
    neighbors = np.array([neighborret[0][mins]+limy[0],
                          neighborret[1][mins]+limx[0],
                          vals[mins]]).T
    return neighbors

def find_new_points_from_trials(lvl, pt, rad):
    """
    Look for a new minimum in radius from pt on the lvl.
    """
    limy = (max(int(pt[0]-rad),0),min(int(pt[0]+rad),lvl.shape[0]))
    limx = (max(int(pt[1]-rad),0),min(int(pt[1]+rad),lvl.shape[1]) )
    view = lvl[limy[0]:limy[1], limx[0]:limx[1]]
    
    
    neighborret = detect_local_minima(view)
    vals = view[neighborret]
    mins=np.where(vals == vals.min())
    neighbors = np.array([neighborret[0][mins]+limy[0],
                          neighborret[1][mins]+limx[0],
                          vals[mins]]).T
    #print len(neighbors)
    closest = neighbors[0]
    #print pt
    #print neighbors
    mindist = (pt[0]-closest[0])**2 + (pt[1]-closest[1])**2
    for i in xrange(1,len(neighbors)):
        dist = (pt[0]-neighbors[i][0])**2 + (pt[1]-neighbors[i][1])**2
        if dist<mindist:
            closest = neighbors[i]
            mindist = dist
    #print closest
    return (closest[0],closest[1])


def find_all_new_points(lvl, points, rad):
    # Build list of all posiilities...
    trials = []
    for x in xrange(len(points)):
        trials.append([])
        for y in xrange(len(points[x])):
            trials[x].append(find_possible_points(lvl,points[x][y],rad))
    # Evaluate all the possible potentials...
    
    print trials
    pass

def draw_trials(img,lvl, pt,rad):
    """
    Look for a new minimum in radius from pt on the lvl.
    """
    limy = (max(int(pt[0]-rad),0),min(int(pt[0]+rad),lvl.shape[0]))
    limx = (max(int(pt[1]-rad),0),min(int(pt[1]+rad),lvl.shape[1]) )
    view = lvl[limy[0]:limy[1], limx[0]:limx[1]]

    
    neighborret = detect_local_minima(view)
    neighbors = np.array([neighborret[0],neighborret[1],view[neighborret]]).T
    #print neighbors
    #print len(neighbors)
    for p in neighbors:
        cv2.circle(img,(limx[0]+int(p[1]),limy[0]+int(p[0])),
                       2,(0,0,255),-1)

    #loc = np.argmin(view)
    #uri = np.unravel_index(loc,view.shape)
    #return (limy[0]+uri[0],limx[0]+uri[1])
#
# Take the first frame and initialize the tracking
#
ret,old_frame = cap.read()
old_frame = cv2.flip(old_frame,-1) if flipit else old_frame

old_fils = apply_filters(old_frame)

points = locate_initial_points(corners,NY,NX)
points = update_points(old_fils[4], points, radius)

# Grab the resolution
resy,resx = old_fils[0].shape[0],old_fils[1].shape[1]
# Make a buffer to display a 2x2 grid of filters.
displayer = np.zeros((resy*2,resx*2,3),dtype=old_frame.dtype)

if writevid:
    video = cv2.VideoWriter('video.mjpeg',cv2.cv.CV_FOURCC('M','J','P','G'),
                            1,(2*resx,2*resy),1)
    print video.isOpened()

pointhistory = [ copy.deepcopy(points) ]
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
    points = update_points(fils[4], points, radius)
    #find_all_new_points(fils[4],points,radius)
    pointhistory.append( copy.deepcopy(points) )
    # 
    # Display the frames
    #
    # Convert to color
    for i in xrange(4):
        fils[i] = cv2.cvtColor(fils[i],cv2.COLOR_GRAY2BGR)
    draw_points(fils[0],points)
    #draw_trials(fils[0],fils[4],points[0][0],radius)
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
    if writevid:
        video.write(displayer)


pickle.dump( pointhistory, open("pointhistory.p","wb"))
# Clean up
cv2.destroyAllWindows()
cap.release()
ux,uy = [],[]
for time in pointhistory:
    for line in time:
        for pt in line:
            ux.append( pt[1] )
            uy.append( pt[0] )

#ux = [ pt[1] for pt in line for line in time for time in pointhistory ]
#uy = [ pt[0] for pt in line for line in time for time in pointhistory ]
plt.plot( range(len(ux)),ux )
plt.show()
if writevid:
    video.release()

