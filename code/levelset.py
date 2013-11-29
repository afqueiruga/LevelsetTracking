"""
Routines to do level-set based transforms.
"""

import cv2
import numpy as np
from ctypes import *
import os

c_double_p = POINTER(c_double)
def as_pointer(x):
    return x.ctypes.data_as(c_double_p)
# Mac:
#os.system('gcc -std=c99 -shared -Wl,-install_name,levelset_clib.so -o levelset_clib.so -fPIC levelset_filters.c')
# Linux:
os.system('gcc -std=c99 -shared -Wl,-soname,levelset_clib -o levelset_clib.so -fPIC levelset_filters.c')

#print "FOOBARXXXXXXXXXXXXXXXXXXXXXXX"
clib = CDLL('./levelset_clib.so')

maprange = lambda x:np.uint8( 255*(1.0*x-np.min(x))/(1.0*np.max(x)-np.min(x)) )

def makephi(img):
    """
    Make a levelset from the contour of a binary image
    """
    dist_plus = cv2.distanceTransform(img,cv2.cv.CV_DIST_L2,5)[0]
    dist_minus = cv2.distanceTransform(~img,cv2.cv.CV_DIST_L2,5)[0]
    dist = np.array(dist_plus - dist_minus, dtype=np.double)
    return dist


def levelsetPhase(dist,phases=[[0.1,0.1,10]]):
    """
    Like levelsetBinary, but goes through a list in phases 
    to do multiple passes.
    """


    for phase in phases:
        levelsetC( dist, phase[0], phase[1],phase[2] )
    rebin = 255*np.array(dist>0,dtype=np.uint8)
    return rebin,dist


def levelsetC(p, A=1.0,eps=0.1,NITER=1):
    """
    Call the C code for the levelset filter.
    """
    dp = np.zeros_like(p)
    clib.curvature_filter(p.ctypes.data_as(c_double_p),
                          dp.ctypes.data_as(c_double_p),
                          c_int(p.shape[0]),c_int(p.shape[1]),c_int(p.shape[1]),
                          c_int(NITER),c_double(A),c_double(eps))
    return p


def levelsetBinary(img,A=1.0,eps=0.01,iterations=1):
    """
    Make the distance mapping first from a binary image.
    """
    dist_plus = cv2.distanceTransform(img,cv2.cv.CV_DIST_L2,5)
    dist_minus = cv2.distanceTransform(~img,cv2.cv.CV_DIST_L2,5)
    dist = dist_plus - dist_minus
    return levelset(dist,A,eps,iterations)


def levelset(img, A=1.0,eps=0.01, iterations=1):
    """
    Do a level set smoothing on 
    """
    resy,resx = img.shape
    H0 = 1.0
    H1 = 1.0
    out = img.copy()
    dphi = np.zeros_like(out)
    for it in xrange(iterations):
        dphi[:,:] = 0.0
        for m in xrange(1,resy-1):
            for n in xrange(1,resx-1):
                # Central derivatives
                Dc0 = -(out[m-1,n]-out[m+1,n])/(2.0*H0)
                Dc1 = -(out[m,n-1]-out[m,n+1])/(2.0*H0)
                # Upwind (plus) derivatives
                Dp0 = -(out[m,n]-out[m+1,n])/(2.0*H0)
                Dp1 = -(out[m,n]-out[m,n+1])/(2.0*H0)
                # Downwind (minus) derivatives
                Dm0 = -(out[m-1,n]-out[m,n])/(2.0*H0)
                Dm1 = -(out[m,n-1]-out[m,n])/(2.0*H0)

                # Compute the normal direction
                gpmag = np.sqrt(Dc0*Dc0+Dc1*Dc1)
                n0 = -Dc1/gpmag
                n1 = Dc0/gpmag

                # Upwind and downwind gradients
                gradP = np.sqrt( min(Dp0,0.0)**2.0 + max(Dm0,0.0)**2.0 +
                              min(Dp1,0.0)**2.0 + max(Dm1,0.0)**2.0 )
                gradM = np.sqrt( max(Dp0,0.0)**2.0 + min(Dm0,0.0)**2.0 +
                              max(Dp1,0.0)**2.0 + min(Dm1,0.0)**2.0 )

                # Second Derivatives
                DDc0 = ( out[m-1,n] - 2.0*out[m,n] + out[m+1,n] )/(H0*H0)
                DDc1 = ( out[m,n-1] - 2.0*out[m,n] + out[m,n+1] )/(H1*H1)
                Dc0Dc1 = ( out[m-1,n-1] - out[m+1,n-1] + out[m+1,n+1] 
                           - out[m-1,n+1] )/(4.0*H0*H1)

                # Curvature
                kappa = (DDc0*Dc1*Dc1 - 2.0*Dc0*Dc1*Dc0Dc1 + DDc1*Dc0*Dc0) / \
                  (gpmag**3.0 + 1.0e-5)
                # Flux
                F = A - eps*kappa

                dphi[m,n] = -( max(F,0.0)*gradP + min(F,0.0)*gradM )
        out[:,:] += dphi[:,:]
    return out
