"""
Import the pickled point history and display it and graph it.
"""

import cPickle as pickle

import numpy as np
import matplotlib.pylab as plt
from scipy import signal as sig

pointhistory = pickle.load( open("pointhistory.p","rb") )

NTIME = len(pointhistory)
NY = len(pointhistory[0])
NX = len(pointhistory[0][0])

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
        plt.xlim(80,200)
        plt.ylim(20,200)
        plot_grid(ph)
        plt.draw()

def strain_history(history, y,x):
    plt.close('all')
    uy0,ux0 = extract_point(history, y,x)
    uy1,ux1 = extract_point(history, y+1,x)
    uy2,ux2 = extract_point(history, y,x+1)
    plt.plot(range(NTIME),uy0,range(NTIME),uy1)
    plt.show()
    plt.figure()
    print uy0.shape
    uy0 = sig.wiener(uy0,mysize=7)
    ux0 = sig.wiener(ux0,mysize=7)
    uy1 = sig.wiener(uy1,mysize=7)
    ux1 = sig.wiener(ux1,mysize=7)
    uy2 = sig.wiener(uy2,mysize=7)
    ux2 = sig.wiener(ux2,mysize=7)
    print uy0
    print uy0.shape
    print uy1
    plt.plot(range(NTIME),uy0,range(NTIME),uy1)
    plt.show()
    plt.figure()
    eyy = calculate_strains(ux0,uy0, ux1,uy1)
    exx = calculate_strains(ux0,uy0, ux2,uy2)
    plt.plot(range(NTIME),exx, range(NTIME),eyy)
    plt.show()
    
