'''
 Filename: 09_DeDvKalman.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

np.random.seed(0)

firstRun = True
X, P = np.array([[0,0]]).transpose(), np.zeros((2,2)) # X : Previous State Variable Estimation, P : Error Covariance Estimation
A, H = np.array([[0,0], [0,0]]), np.array([[0,0]])
Q, R = np.array([[0,0], [0,0]]), 0

firstRun2 = True
X2, P2 = np.array([[0,0]]).transpose(), np.zeros((2,2)) # X : Previous State Variable Estimation, P : Error Covariance Estimation
A2, H2 = np.array([[0,0], [0,0]]), np.array([[0,0]])
Q2, R2 = np.array([[0,0], [0,0]]), 0

Posp, Velp = None, None

def GetPos():
    global Posp, Velp
    if Posp == None:
        Posp = 0
        Velp = 80
    dt = 0.1

    w = 0 + 10 * np.random.normal()
    v = 0 + 10 * np.random.normal()

    z = Posp + Velp * dt + v  # Position measurement

    Posp = z - v
    Velp = 80 + w
    return z, Posp, Velp

'''
    Estimate velocity through displacement
'''
def DvKalman(z):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 0.1
        A = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[1, 0], [0, 3]])
        R = np.array([10])

        X = np.array([0, 20]).transpose()
        P = 5 * np.eye(2)
        firstRun = False
    else:
        Xp = A @ X # Xp : State Variable Prediction
        Pp = A @ P @ A.T + Q # Error Covariance Prediction

        K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

        X = Xp + K@(z - H@Xp) # Update State Variable Estimation
        P = Pp - K@H@Pp # Update Error Covariance Estimation

    pos = X[0]
    vel = X[1]

    return pos, vel

def DeDvKalman(z):
    global firstRun2
    global A2, Q2, H2, R2
    global X2, P2
    if firstRun2:
        dt = 0.1
        A2 = np.array([[1, dt], [0, 1]])
        H2 = np.array([[1, 0]])
        Q2 = np.array([[1, 0], [0, 3]])
        R2 = np.array([10])

        X2 = np.array([0, 20]).transpose()
        P2 = 5 * np.eye(2)
        firstRun2 = False
    else:
        Xp = A2 @ X2 # Xp : State Variable Prediction
        Pp = A2 @ P2 @ A2.T + Q2 # Error Covariance Prediction

        #K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain
        K = 1/(np.array(Pp[0,0]) + R2) * np.array([Pp[0,0], Pp[1,0]]).transpose()
        K = K[:, np.newaxis] # maintain axis

        X2 = Xp + K@(z - H@Xp) # Update State Variable Estimation
        P2 = Pp - K@H2@Pp # Update Error Covariance Estimation

    pos = X2[0]
    vel = X2[1]

    return pos, vel

t = np.arange(0, 10, 0.1)
Nsamples = len(t)

Xsaved = np.zeros([Nsamples, 2])
DeXsaved = np.zeros([Nsamples, 2])

for i in range(Nsamples):
    Z, pos_true, vel_true = GetPos()
    pos, vel = DvKalman(Z)
    dpos, dvel = DeDvKalman(Z)

    Xsaved[i] = [pos, vel]
    DeXsaved[i] = [dpos, dvel]

plt.figure()
plt.plot(t, Xsaved[:,1], 'b*', label = 'Matrix')
plt.plot(t, DeXsaved[:,1], 'r-', label='Decomposed')
plt.legend(loc='upper left')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [sec]')
plt.savefig('result/09_DeDvKalman.png')
plt.show()