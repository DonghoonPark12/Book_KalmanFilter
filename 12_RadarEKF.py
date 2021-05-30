'''
 Filename: 12_RadarEKF.py
 Created on: April,10, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

A, Q, R = None, None, None
x, P = None, None
firstRun = True
posp = None

def Hjacob(xp):
    H = np.zeros([1,3])
    x1 = xp[0]
    x3 = xp[2]
    H[:,0] = x1 / np.sqrt(x1**2 + x3**2)
    H[:,1] = 0
    H[:,2] = x3 / np.sqrt(x1**2 + x3**2)
    return H

def hx(xhat):
    x1 = xhat[0]
    x3 = xhat[2]

    zp = np.sqrt(x1**2 + x3**2)
    return zp

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

def RadarEKF(z, dt):
    global firstRun
    global A, Q, R, x, P
    if firstRun:
        A = np.eye(3) + dt * np.array([[0,1,0],[0,0,0],[0,0,0]])
        Q = np.array([[0,0,0],[0,0.001,0],[0,0,0.001]])
        R = 10
        x = np.array([0,90,1100]).transpose()
        P = 10 * np.eye(3)
        firstRun = False
    else:
        H = Hjacob(x)
        Xp = A @ x # Xp : State Variable Prediction
        Pp = A @ P @ A.T + Q # Error Covariance Prediction

        K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

        x = Xp + K@(np.array([z - hx(Xp)])) # Update State Variable Estimation
        P = Pp - K@H@Pp # Update Error Covariance Estimation

    pos = x[0]
    vel = x[1]
    alt = x[2]
    return pos, vel, alt

dt = 0.05
t = np.arange(0, 20, dt)
Nsamples = len(t)
Xsaved = np.zeros([Nsamples,3])
Zsaved = np.zeros([Nsamples,1])
Estimated = np.zeros([Nsamples,1])

for k in range(Nsamples):
    r = GetRadar(dt)

    pos, vel, alt = RadarEKF(r, dt)

    Xsaved[k] = [pos, vel, alt]
    Zsaved[k] = r
    Estimated[k] = hx([pos, vel, alt])

PosSaved = Xsaved[:,0]
VelSaved = Xsaved[:,1]
AltSaved = Xsaved[:,2]

t = np.arange(0, Nsamples * dt ,dt)

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
fig.savefig('result/12_RadarEKF.png')
#plt.show()

plt.figure()
plt.plot(t, Zsaved, 'r--', label='Measured')
plt.plot(t, Estimated, 'b-', label='Estimated')
plt.xlabel('Time [Sec]')
plt.ylabel('Radar distance [m]')
plt.legend(loc='upper left')
plt.savefig('result/12_RadarEKF(2).png')
plt.show()
