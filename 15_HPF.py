'''
 Filename: 15_HPF.py
 Created on: April,17, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat('./02_SonarAlt.mat')

prevX, prevU = 0, 0
dt, tau = 0, 0
firstRun = True

def GetSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z

def HPF(u):
    global firstRun
    global prevX, prevU, dt, tau
    if firstRun:
        prevX, prevU = 0, 0
        dt = 0.01
        tau = 0.0233
        firstRun = False

    alpha = tau / (tau + dt)
    xhpf = alpha * prevX + alpha * (u - prevU)
    prevX, prevU = xhpf, u
    return prevX

Nsamples = 500
Noise = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

for k in range(0, Nsamples):
    zm = GetSonar(k)
    x = HPF(zm)

    Noise[k] = x
    Xmsaved[k] = zm

dt = 0.02
t = np.arange(0, Nsamples*dt, dt)

plt.plot(t, Noise, 'b', label='HPF')
plt.plot(t, Xmsaved, 'r.', label='Measured')
plt.legend(loc='upper left')
plt.ylabel('Altitude[m]')
plt.xlabel('Time [sec]')
plt.savefig('result/15_high_pass_filter.png')
plt.show()

# plt.plot(t, Xmsaved - Noise, 'b', label='Measured - HPF')
# plt.plot(t, Xmsaved, 'r.', label='Measured')
# plt.legend(loc='upper left')
# plt.ylabel('Altitude[m]')
# plt.xlabel('Time [sec]')
# plt.savefig('result/15_high_pass_filter(2).png')
# plt.show()