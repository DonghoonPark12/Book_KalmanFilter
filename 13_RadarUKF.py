'''
 Filename: 13_RadarUKF.py
 Created on: April,10, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt

Q, R = None, None
x, P = None, None
n, m = None, None
firstRun = True
posp = None

def SigmaPoints(xm, P, kappa):
    n = len(xm)
    Xi = np.zeros([n, 2*n + 1])
    W = np.zeros(2*n + 1)

    Xi[:,0] = xm
    W[0] = kappa / (n + kappa)

    U = cholesky((n+kappa)*P)

    for k in range(n):
        Xi[:, k + 1] = xm + U[:, k]
        W[k + 1] = 1 / (2*(n + kappa))

        Xi[:, n+k + 1] = xm - U[:, k]
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

'''
xm, Px, kappa = 5,9,2
Xi, W = SigmaPoints(xm, Px, kappa)
xAvg, xCov = UT(Xi, W)
'''

def fx(x, dt):
    A = np.eye(3) + dt * np.array([[0,1,0],[0,0,0],[0,0,0]])
    xp = A @ x # x : (3,) -> (3,1)
    return xp

def hx(x):
    x1, x3 = x[0], x[2]
    yp = np.sqrt(x1**2 + x3**2)
    return yp

def RadarUKF(z, dt):
    global firstRun
    global Q, R, x, P, n, m
    if firstRun:
        Q = np.array([[0.001,0,0],[0,0.001,0],[0,0,0.001]])
        R = 100
        x = np.array([0,90,1100]).T
        P = 100 * np.eye(3)
        n, m = 3, 1
        firstRun = False
    else:
        Xi, W = SigmaPoints(x, P, 0)

        # Convert data to system model data
        fXi = np.zeros([n, 2*n + 1])
        for k in range(2*n + 1):
            fXi[:,k] = fx(Xi[:,k], dt)

        # Predict mean of f(x) and error covariance
        xp, Pp = UT(fXi, W, Q)

        hXi = np.zeros([m, 2*n + 1])
        for k in range(2*n + 1):
            hXi[:,k] = hx(fXi[:,k])

        # Predict mean of h(x) and error covariance
        zp, Pz = UT(hXi, W, R)

        # Calculate kalman gain
        Pxz = np.zeros([n,m])
        for k in range(2*n + 1):
            Pxz = Pxz + W[k] * (fXi[:,k] - xp).reshape(-1,1) @ (hXi[:,k] - zp).reshape(-1,1).T

        K = Pxz @ inv(Pz)

        x = xp + K@(z - zp)
        P = Pp - K@Pz@K.T

    pos = x[0]
    vel = x[1]
    alt = x[2]
    return pos, vel, alt

def GetRadar(dt):
    global posp
    if posp == None:
        posp = 0

    vel = 100 + 5*np.random.randn()
    alt = 1000 + 10*np.random.randn()
    pos = posp + vel*dt
    v = 0 + pos*0.05*np.random.randn()
    r = np.sqrt(pos**2 + alt**2) + v
    posp = pos
    return r

dt = 0.05
t = np.arange(0, 20, dt)
Nsamples = len(t)
Xsaved = np.zeros([Nsamples,3])
Zsaved = np.zeros([Nsamples,1])
Estimated = np.zeros([Nsamples,1])

for k in range(Nsamples):
    r = GetRadar(dt)

    pos, vel, alt = RadarUKF(r, dt)

    Xsaved[k] = [pos, vel, alt]
    Zsaved[k] = r
    Estimated[k] = hx([pos, vel, alt])

PosSaved = Xsaved[:,0]
VelSaved = Xsaved[:,1]
AltSaved = Xsaved[:,2]

fig = plt.figure()
plt.subplot(313)
plt.plot(t, PosSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Position [m]')

plt.subplot(311)
plt.plot(t, VelSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Speed [m/s]')

plt.subplot(312)
plt.plot(t, AltSaved)
plt.xlabel('Time [Sec]')
plt.ylabel('Altitude [m]')
fig.savefig('result/13_RadarUKF.png')
plt.show()