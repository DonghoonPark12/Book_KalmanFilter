'''
 Filename: 02_MovAvgFilter.py
 Created on: April, 3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat('./02_SonarAlt.mat')

xbuf = []
n = 0
firstRun = True

def GetSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z

def MovAvgFilter_batch(x):
    global n, xbuf, firstRun
    if firstRun:
        n = 10
        xbuf = x * np.ones(n)
        firstRun = False
    else:
        for i in range(n-1):
            xbuf[i] = xbuf[i+1]
        xbuf[n-1] = x
    avg = np.sum(xbuf) / n
    return avg

# def MovAvgFilter_recur(x):
#

Nsamples = 500
Xsaved = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

for k in range(0, Nsamples):
    xm = GetSonar(k)
    x = MovAvgFilter_batch(xm)

    Xsaved[k] = x
    Xmsaved[k] = xm

dt = 0.02
t = np.arange(0, Nsamples*dt, dt)

plt.plot(t, Xsaved, 'b-', label='Moving average')
plt.plot(t, Xmsaved, 'r.', label='Measured')
plt.legend(loc='upper left')
plt.ylabel('Altitude[m]')
plt.xlabel('Time [sec]')
plt.savefig('result/02_moving_average_filter.png')
plt.show()