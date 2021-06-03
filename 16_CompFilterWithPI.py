'''
 Filename: 16_CompFilterWithPI.py
 Created on: April,17, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, tan, asin, atan2, pi
from scipy import io

p_hat, q_hat = None, None
prevPhi, prevTheta, prevPsi = None, None, None
prevP, prevdelPhi, prevQ, prevdelTheta = None, None, None, None

input_mat = io.loadmat('./11_ArsGyro.mat')
input_mat2 = io.loadmat('./11_ArsAccel.mat')

def GetGyro(i):
    p = input_mat['wx'][i][0]  # (41500, 1)
    q = input_mat['wy'][i][0]  # (41500, 1)
    r = input_mat['wz'][i][0]  # (41500, 1)
    return p, q, r

def GetAccel(i):
    ax = input_mat2['fx'][i][0]  # (41500, 1)
    ay = input_mat2['fy'][i][0]  # (41500, 1)
    az = input_mat2['fz'][i][0]  # (41500, 1)
    return ax, ay, az

def EulerAccel(ax, ay, az):
    g = 9.8
    theta = asin(ax / g)
    phi = asin(-ay / (g * cos(theta)))
    return phi, theta

def BodyToInertial(p, q, r, phi, theta):
    '''
        Body rate -> Eular angular rate
    '''
    sinPhi, cosPhi = sin(phi), cos(phi)
    cosTheta, tanTheta = cos(theta), tan(theta)

    dotPhi   = p + q * sinPhi * tanTheta + r * cosPhi * tanTheta
    dotTheta =     q * cosPhi            - r * sinPhi
    dotPsi   =     q * sinPhi / cosTheta + r * cosPhi / cosTheta
    return dotPhi, dotTheta, dotPsi

def PILawPhi(delPhi):
    global prevP, prevdelPhi
    if prevP is None:
        prevP = 0
        prevdelPhi = 0
    p_hat = prevP + 0.1415 * delPhi - 0.1414 * prevdelPhi
    prevP = p_hat
    prevdelPhi = delPhi
    return p_hat

def PILawTheta(delTheta):
    global prevQ, prevdelTheta
    if prevQ is None:
        prevQ = 0
        prevdelTheta = 0
    q_hat = prevQ + 0.1415 * delTheta - 0.1414 * prevdelTheta
    prevQ = q_hat
    prevdelTheta = delTheta
    return q_hat


def CompFilterWithPI(p, q, r, ax, ay, az, dt):
    global p_hat, q_hat
    global prevPhi, prevTheta, prevPsi
    if p_hat is None:
        p_hat, q_hat = 0, 0
        prevPhi, prevTheta, prevPsi = 0, 0, 0

    # Get roll, pitch from accelerometer
    phi_a, theta_a = EulerAccel(ax, ay, az)

    # Get 'Euler angle' from 'Gyro angular velocity'
    dotPhi, dotTheta, dotPsi = BodyToInertial(p, q, r, prevPhi, prevTheta)

    phi = prevPhi + dt * (dotPhi - p_hat)
    theta = prevTheta + dt * (dotTheta - q_hat)
    psi = prevPsi + dt * dotPsi

    p_hat = PILawPhi(phi - phi_a)
    q_hat = PILawTheta(theta - theta_a)

    prevPhi = phi
    prevTheta = theta
    prevPsi = psi

    return phi, theta, psi


Nsamples = 41500
EulerSaved = np.zeros([Nsamples,3])
dt = 0.01

for k in range(Nsamples):
    ax, ay, az = GetAccel(k)
    p, q, r = GetGyro(k)
    phi, theta, psi = CompFilterWithPI(p, q, r, ax, ay, az, dt)
    EulerSaved[k] = [phi, theta, psi]

t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(t, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/16_EulerCompFilter_roll.png')

plt.figure()
plt.plot(t, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/16_EulerCompFilter_pitch.png')
plt.show()