'''
 Filename: 03_low_pass_filter.py
 Created on: April, 3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat('./02_SonarAlt.mat')

prevX1 = 0
firstRun1 = True
prevX2 = 0
firstRun2 = True

def GetSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z

def LPF1(x, alpha):
    global prevX1, firstRun1
    if firstRun1:
        prevX1 = x
        firstRun1 = False
    #alpha = 0.7
    xlpf = alpha * prevX1 + (1 - alpha) * x
    prevX1 = xlpf
    return xlpf

def LPF2(x, alpha):
    global prevX2, firstRun2
    if firstRun2:
        prevX2 = x
        firstRun2 = False
    #alpha = 0.7
    xlpf = alpha * prevX2 + (1 - alpha) * x
    prevX2 = xlpf
    return xlpf

Nsamples = 500
Xsaved1 = np.zeros(Nsamples)
Xsaved2 = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

for k in range(0, Nsamples):
    xm = GetSonar(k)
    x1 = LPF1(xm, 0.4)
    x2 = LPF2(xm, 0.9)

    Xsaved1[k] = x1
    Xsaved2[k] = x2
    Xmsaved[k] = xm

dt = 0.02
t = np.arange(0, Nsamples*dt, dt)

plt.plot(t, Xsaved1, 'b-', label='LPF(alpha = 0.4)')
plt.plot(t, Xsaved2, 'g-', label='LPF(alpha = 0.9)')
plt.plot(t, Xmsaved, 'r.', label='Measured')
plt.legend(loc='upper left')
plt.ylabel('Altitude[m]')
plt.xlabel('Time [sec]')
plt.savefig('result/03_low_pass_filter.png')
plt.show()