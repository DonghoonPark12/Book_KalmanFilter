'''
 Filename: 13_EulerUKF.py
 Created on: April,10, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt
from math import cos, sin, tan, asin, pi
from scipy import io

Q, R = None, None
x, P = None, None
N, M = None, None
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
    return 1. /cos(theta)

def SigmaPoints(xm, P, kappa):
    n = len(xm)
    Xi = np.zeros([n, 2*n + 1])
    W = np.zeros(2*n + 1)

    Xi[:,0] = xm
    W[0] = kappa / (n + kappa)

    U = cholesky((n+kappa)*P)

    for k in range(n):
        Xi[:, k + 1] = xm + U[:, k]
        Xi[:, n + k + 1] = xm - U[:, k]

        W[k + 1] = 1 / (2*(n + kappa))
        W[n+k + 1] = 1 / (2*(n + kappa))

    return Xi, W

def UT(Xi, W, noiseCov):
    n, kmax = len(Xi), len(Xi[0])
    xm = 0

    for k in range(kmax):
        xm += W[k] * Xi[:,k] # (7,) * (3,7)
    xcov = np.zeros([n, n])

    for k in range(kmax):
        l = (Xi[:,k] - xm).reshape(-1,1)   # (3,) -> (3,1)
        r = (Xi[:,k] - xm).reshape(-1,1).T # (3,) -> (1,3)
        xcov = xcov + W[k] * l @ r
    xcov = xcov + noiseCov

    return xm, xcov

def fx(xhat, rates, dt):
    phi = xhat[0]
    theta = xhat[1]

    p,q,r = rates[0], rates[1], rates[2]

    xdot = np.zeros(3)
    xdot[0] = p + q * sin(phi) * tan(theta) + r * cos(phi)*tan(theta)
    xdot[1] = q * cos(phi) - r * sin(phi)
    xdot[2] = q * sin(phi)*sec(theta) + r * cos(phi) * sec(theta)

    xp = xhat + xdot*dt # xhat : (3,) --> (3,1)
    return xp

def hx(x):
    return x[0], x[1]

def EulerUKF(z, rates, dt):
    global firstRun
    global Q, R, x, P
    global N, M
    if firstRun:
        Q = np.array([[0.0001,0,0],[0,0.0001,0],[0,0,1]])
        R = 10 * np.eye(2) #[21.04.20] 2 -> 3,
        x = np.array([0, 0, 0]).transpose()
        P = 1 * np.eye(3)
        N, M = 3, 2
        firstRun = False
    else:
        Xi, W = SigmaPoints(x, P, 0)

        # Convert data to system model data
        fXi = np.zeros([N, 2*N + 1])
        for k in range(2*N + 1):
            fXi[:,k] = fx(Xi[:,k], rates, dt)

        # Predict mean of f(x) and error covariance
        xp, Pp = UT(fXi, W, Q)

        hXi = np.zeros([M, 2*N + 1])
        for k in range(2*N + 1):
            hXi[:,k] = hx(fXi[:,k])

        # Predict mean of h(x) and error covariance
        zp, Pz = UT(hXi, W, R)

        # Calculate kalman gain
        Pxz = np.zeros([N,M])
        for k in range(2*N + 1):
            Pxz = Pxz + W[k] * (fXi[:,k] - xp).reshape(-1,1) @ (hXi[:,k] - zp).reshape(-1,1).T
        #Same as Pxz = W * (fXi - xp.reshape(-1,1)) @ (hXi - zp.reshape(-1,1)).T

        K = Pxz @ inv(Pz)

        x = xp + K@(z - zp)
        P = Pp - K@Pz@K.T

    phi   = x[0]
    theta = x[1]
    psi   = x[2]
    return phi, theta, psi

Nsamples = 41500
EulerSaved = np.zeros([Nsamples,3])
dt = 0.01

phi_a, theta_a, psi_a = 0, 0, 0
for k in range(Nsamples):
    p, q, r = GetGyro(k)
    ax, ay, az = GetAccel(k)
    phi_a, theta_a = EulerAccel(ax, ay, az)

    phi, theta, psi = EulerUKF(np.array([phi_a, theta_a]).T, [p,q,r], dt)

    EulerSaved[k] = [phi, theta, psi]

t = np.arange(0, Nsamples * dt ,dt)
PhiSaved = EulerSaved[:,0] * 180/pi
ThetaSaved = EulerSaved[:,1] * 180/pi
PsiSaved = EulerSaved[:,2] * 180/pi

plt.figure()
plt.plot(t, PhiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Roll angle [deg]')
plt.savefig('result/13_EulerUKF_roll.png')

plt.figure()
plt.plot(t, ThetaSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Pitch angle [deg]')
plt.savefig('result/13_EulerUKF_pitch.png')
plt.show()
'''
plt.subplot(133)
plt.plot(t, PsiSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Psi angle [deg]')
'''

