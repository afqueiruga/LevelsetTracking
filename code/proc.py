"""
Import the pickled point history and display it and graph it.
"""

import cPickle as pickle

import numpy as np
import matplotlib.pylab as plt
from scipy import signal as sig

pointhistory = pickle.load( open("fast_block.p","rb") )

NTIME = len(pointhistory)
NY = len(pointhistory[0])
NX = len(pointhistory[0][0])
FPS = 1200.0

print NTIME, NY, NX

def plot_grid(points):
    for i in xrange(NY):
        plt.plot( [p[1] for p in points[i]],[p[0] for p in points[i]] )
    for i in xrange(NX):
        plt.plot( [p[i][1] for p in points],[p[i][0] for p in points] )

def extract_point(history, y,x):
    ux = np.array([ p[y][x][0] for p in history ],dtype=np.double)
    uy = np.array([ p[y][x][1] for p in history ],dtype=np.double)
    return ux, uy
def calculate_strains(ux0,uy0, ux1,uy1):
    lengths = np.sqrt( (ux1-ux0)**2+(uy1-uy0)**2 )
    strains = lengths/lengths[0]-1.0
    return strains

def animate():
    plt.show()
    for ph in pointhistory:
        print "yo"
        plt.clf()
        plt.xlim(20,160)
        plt.ylim(0,140)
        plot_grid(ph)
        plt.draw()
def motion_history(history, y,x):
    #fil = lambda x: np.convolve(x,np.ones(10)/10)
    fil = lambda x: sig.wiener(x,mysize=7)
    uy,ux = extract_point(history, y,x)
    #uy = np.convolve(uy, np.ones(10)/10)
    #ux = np.convolve(ux, np.ones(10)/10)
    uy = fil(uy)
    ux = fil(ux)
    print len(uy)
    vy = fil(np.diff(uy))/FPS
    vx = fil(np.diff(ux))/FPS

    ay = fil(np.diff(uy,2))/(FPS**2)
    ax = fil(np.diff(ux,2))/(FPS**2)
    print len(vx)
    print len(ax)
    plt.close('all')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')
    plt.plot(np.arange(len(uy))/FPS,ux)
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (px/s)')
    plt.plot(np.arange(len(vy))/FPS+0.5/FPS,vx)
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (px/s^2)')
    plt.plot(np.arange(len(ay))/FPS+1.0/FPS,ax)
    plt.show()
def strain_history(history, y,x):
    plt.close('all')
    uy0,ux0 = extract_point(history, y,x)
    uy1,ux1 = extract_point(history, y+1,x)
    uy2,ux2 = extract_point(history, y,x+1)
    time = np.arange(NTIME)/FPS
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')
    plt.plot(time,uy0,time,uy1)
    plt.show()
    plt.figure()
    
    uy0 = sig.wiener(uy0,mysize=7)
    ux0 = sig.wiener(ux0,mysize=7)
    uy1 = sig.wiener(uy1,mysize=7)
    ux1 = sig.wiener(ux1,mysize=7)
    uy2 = sig.wiener(uy2,mysize=7)
    ux2 = sig.wiener(ux2,mysize=7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Position (pixels)')
    plt.plot(time,uy0,time,uy1)
    plt.show()
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('% Strain')
    eyy = calculate_strains(ux0,uy0, ux1,uy1)
    exx = calculate_strains(ux0,uy0, ux2,uy2)
    plt.plot(time,100*exx, label='exx')
    plt.plot(time,100*eyy,label='eyy')
    plt.legend(loc=3)
    plt.show()
    
