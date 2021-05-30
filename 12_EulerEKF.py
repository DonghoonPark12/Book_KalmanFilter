'''
 Filename: 12_EulerEKF.py
 Created on: April,10, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import cos, sin, tan, asin, pi
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

def sec(theta):
    return 1/cos(theta)

def Ajacob(xhat, rates, dt):
    '''
    :param xhat: State Variables(phi, theta, psi)
    :param rates: angel speed(p,q,r)
    :param dt: variable to make discrete form
    '''
    A = np.zeros([3,3])
    phi = xhat[0]
    theta = xhat[1]

    p,q,r = rates[0], rates[1], rates[2]

    A[0][0] = q * cos(phi)*tan(theta) - r*sin(phi)*tan(theta)
    A[0][1] = q * sin(phi)*(sec(theta)**2) + r*cos(phi)*(sec(theta)**2)
    A[0][2] = 0

    A[1][0] = -q * sin(phi) - r * cos(phi)
    A[1][1] = 0
    A[1][2] = 0

    A[2][0] = q * cos(phi) * sec(theta) - r * sin(phi) * sec(theta)
    A[2][1] = q * sin(phi) * sec(theta)*tan(theta) + r*cos(phi)*sec(theta)*tan(theta)
    A[2][2] = 0

    A = np.eye(3) + A*dt

    return A

def fx(xhat, rates, dt):
    phi = xhat[0]
    theta = xhat[1]

    p,q,r = rates[0], rates[1], rates[2]

    xdot = np.zeros([3,1])
    xdot[0] = p + q * sin(phi) * tan(theta) + r * cos(phi)*tan(theta)
    xdot[1] = q * cos(phi) - r * sin(phi)
    xdot[2] = q * sin(phi)*sec(theta) + r * cos(phi) * sec(theta)

    xp = xhat.reshape(-1,1) + xdot*dt # xhat : (3,) --> (3,1)
    return xp

def EulerEKF(z, rates, dt):
    global firstRun
    global Q, H, R
    global x, P
    if firstRun:
        H = np.array([[1,0,0],[0,1,0]])
        Q = np.array([[0.0001,0,0],[0,0.0001,0],[0,0,0.1]])
        R = 10 * np.eye(2)
        x = np.array([0, 0, 0]).transpose()
        P = 10 * np.eye(3)
        firstRun = False
    else:
        A = Ajacob(x, rates, dt)
        Xp = fx(x, rates, dt) # Xp : State Variable Prediction
        Pp = A @ P @ A.T + Q # Error Covariance Prediction

        K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

        x = Xp + K@(z.reshape(-1,1) - H@Xp) # Update State Variable Estimation
        P = Pp - K@H@Pp # Update Error Covariance Estimation

    phi   = x[0]
    theta = x[1]
    psi   = x[2]
    return phi, theta, psi

Nsamples = 41500
EulerSaved = np.zeros([Nsamples,3])
dt = 0.01

for k in range(Nsamples):
    p, q, r = GetGyro(k)
    ax, ay, az = GetAccel(k)
    phi_a, theta_a = EulerAccel(ax, ay, az)

    phi, theta, psi = EulerEKF(np.array([phi_a, theta_a]).T, [p,q,r], dt)

    EulerSaved[k] = [phi, theta, psi]

t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(t, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/12_EulerEKF_roll.png')

plt.figure()
plt.plot(t, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/12_EulerEKF_pitch.png')
plt.show()
'''
plt.subplot(133)
plt.plot(t, PsiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Psi angle [deg]')
'''

