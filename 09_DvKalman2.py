'''
 Filename: 09_DvKalman2.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat('./02_SonarAlt.mat')

firstRun = True

def GetSonar(i):
    z = input_mat['sonarAlt'][0][i]  # (1, 1501)
    return z

def DvKalman(z):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 0.02
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

Nsamples = 500
t = np.arange(0, 10, 0.02)
Xsaved = np.zeros([Nsamples, 2])
Zsaved = np.zeros(Nsamples)

for k in range(0, Nsamples):
    z = GetSonar(k)
    z = (z - 40)/2
    pos, vel = DvKalman(z)

    Xsaved[k] = [pos, vel]
    Zsaved[k] = z

# plt.figure()
# plt.plot(t, Zsaved, 'b.', label = 'Measurements')
# plt.plot(t, Xsaved[:,0], 'r-', label='Kalman Filter')
# plt.legend(loc='upper left')
# plt.ylabel('Distance [m]')
# plt.xlabel('Time [sec]')
# plt.savefig('result/09_DvKalman2.png')

plt.figure()
plt.plot(t, Zsaved, 'b--', label = 'Distance')
plt.plot(t, Xsaved[:,1], 'r-', label='Velocity')
plt.legend(loc='upper left')
plt.ylabel('Velocity [m/s]')
plt.xlabel('Time [sec]')
plt.savefig('result/09_DvKalman2_eVel.png')
plt.show()