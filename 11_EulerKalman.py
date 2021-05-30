'''
 Filename: 11_EulerKalman.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import cos, sin, asin, atan2, pi
from scipy import io

H, Q, R = None, None, None
x, P = None, None
firstRun = True

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

def EulerToQuaternion(phi, theta, psi):
    sinPhi = sin(phi/2)
    cosPhi = cos(phi/2)
    sinTheta = sin(theta/2)
    cosTheta = cos(theta/2)
    sinPsi = sin(psi/2)
    cosPsi = cos(psi/2)
    z = np.array([cosPhi*cosTheta*cosPsi + sinPhi*sinTheta*sinPsi,
                  sinPhi*cosTheta*cosPsi - cosPhi*sinTheta*sinPsi,
                  cosPhi*sinTheta*cosPsi + sinPhi*cosTheta*sinPsi,
                  cosPhi*cosTheta*sinPsi - sinPhi*sinTheta*cosPsi])
    return z

def EulerKalman(A, z):
    global firstRun
    global Q, H, R
    global x, P
    if firstRun:
        H = np.eye(4)
        Q = 0.0001 * np.eye(4)
        R = 10 * np.eye(4)
        x = np.array([1, 0, 0, 0]).transpose()
        P = np.eye(4)
        firstRun = False
    else:
        Xp = A @ x # Xp : State Variable Prediction
        Pp = A @ P @ A.T + Q # Error Covariance Prediction

        K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

        x = Xp + K@(z - H@Xp) # Update State Variable Estimation
        P = Pp - K@H@Pp # Update Error Covariance Estimation

    phi   = atan2(2 * (x[2] * x[3] + x[0] * x[1]), 1 - 2*(x[1]**2 + x[2]**2))
    theta = -asin(2 *  (x[1] * x[3] - x[0] * x[2]))
    psi   = atan2(2 *  (x[1] * x[2] + x[0] * x[3]), 1-2*(x[2]**2 + x[3]**2))
    return phi, theta, psi

Nsamples = 41500
EulerSaved = np.zeros([Nsamples,3])
dt = 0.01

for k in range(Nsamples):
    p, q, r = GetGyro(k)
    A = np.eye(4) + dt * (1/2) * np.array([[0,-p,-q,-r],[p,0,r,-q],[q,-r,0,p],[r,q,-p,0]])
    ax, ay, az = GetAccel(k)
    phi, theta = EulerAccel(ax, ay, az)
    z = EulerToQuaternion(phi, theta, 0) #State variable as Quaternion form

    phi, theta, psi = EulerKalman(A, z)
    EulerSaved[k] = [phi, theta, psi]

t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(t, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/11_EulerKalman_roll.png')

plt.figure()
plt.plot(t, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/11_EulerKalman_pitch.png')
plt.show()

