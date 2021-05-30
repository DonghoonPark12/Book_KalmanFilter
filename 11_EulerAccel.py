'''
 Filename: 11_EulerAccel.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from math import asin, cos, pi, atan, sqrt
from scipy import io

prevPhi, prevTheta, prevPsi = None,None,None
input_mat = io.loadmat('./11_ArsAccel.mat')

def GetAccel(i):
    ax = input_mat['fx'][i][0]  # (41500, 1)
    ay = input_mat['fy'][i][0]  # (41500, 1)
    az = input_mat['fz'][i][0]  # (41500, 1)
    return ax, ay, az

def EulerAccel(ax, ay, az):
    g = 9.8
    theta = asin(ax / g)
    phi = asin(-ay / (g * cos(theta)))
    return phi, theta

def EulerAccel2(ax, ay, az):
    theta = atan(ax/sqrt(ay**2 + az**2))
    phi = atan(ay/az)
    return phi, theta

Nsamples = 41500
EulerSaved = np.zeros([Nsamples,2])
EulerSaved2 = np.zeros([Nsamples,2])
dt = 0.01

for k in range(Nsamples):
    ax, ay, az = GetAccel(k)
    phi, theta = EulerAccel(ax, ay, az)
    phi2, theta2 = EulerAccel2(ax, ay, az)

    EulerSaved[k] = [phi, theta]
    EulerSaved2[k] = [phi2, theta2]
'''
roll  -> phi
pitch -> theta
yaw   -> psi
'''
t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi

PhiSaved2 = EulerSaved2[:,0] * 180/pi
ThetaSaved2 = EulerSaved2[:,1] * 180/pi

plt.figure()
plt.plot(t, PhiSaved2)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/11_EulerAccel2_roll.png')

plt.figure()
plt.plot(t, ThetaSaved2)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/11_EulerAccel2_pitch.png')
plt.show()

