'''
 Filename: 11_EulerGyro.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, tan, pi
from scipy import io

prevPhi, prevTheta, prevPsi = None,None,None
input_mat = io.loadmat('./11_ArsGyro.mat')

def GetGyro(i):
    p = input_mat['wx'][i][0]  # (41500, 1)
    q = input_mat['wy'][i][0]  # (41500, 1)
    r = input_mat['wz'][i][0]  # (41500, 1)
    return p, q, r

def EulerGyro(p,q,r,dt):
    global prevPhi, prevTheta, prevPsi
    if prevPhi is None:
        prevPhi = 0
        prevTheta = 0
        prevPsi = 0
    sinPhi = sin(prevPhi)
    cosPhi = cos(prevPhi)
    cosTheta = cos(prevTheta)
    tanTheta = tan(prevTheta)

    phi = prevPhi + dt*(p + q*sinPhi*tanTheta + r*cosPhi*tanTheta)
    theta = prevTheta + dt*(q*cosPhi - r*sinPhi)
    psi = prevPsi + dt*(q*sinPhi/cosTheta + r*cosPhi/cosTheta)

    prevPsi = psi
    prevPhi = phi
    prevTheta = theta
    return phi, theta, psi

Nsamples = 41500
EulerSaved = np.zeros([Nsamples,3])
dt = 0.01

for k in range(Nsamples):
    p,q,r = GetGyro(k)
    phi, theta, psi = EulerGyro(p,q,r,dt)
    EulerSaved[k] = [phi, theta, psi]

'''
roll  -> phi
pitch -> theta
yaw   -> psi
'''
t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(t, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/11_EulerGyro_roll.png')

plt.figure()
plt.plot(t, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/11_EulerGyro_pitch.png')
plt.show()

