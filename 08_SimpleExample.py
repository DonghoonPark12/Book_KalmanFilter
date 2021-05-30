'''
 Filename: 08_SimpleExample.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt

firstRun = True
X, P = 0, 0  # X : Previous State Variable Estimation, P : Error Covariance Estimation
A, H, Q, R = 0, 0, 0, 0

def GetVolt():
    return 14.4 + np.random.normal(0, 4, 1)

def SimpleKalman(z):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        # A, Q = np.array([1]), np.array([0])
        # H, R = np.array([1]), np.array([4])
        #
        # X = np.array([14])
        # P = np.array([6])

        A, Q = 1,0
        H, R = 1,4

        X = 14
        P = 6
        firstRun = False

    Xp = A * X # Xp : State Variable Prediction
    Pp = A * P * A + Q # Error Covariance Prediction

    K = (Pp * H) / (H*Pp*H + R) # K : Kalman Gain

    X = Xp + K*(z - H*Xp) # Update State Variable Estimation
    P = Pp - K*H*Pp # Update Error Covariance Estimation
    return X, P, K

t = np.arange(0, 10, 0.2)
Nsamples = len(t)

X_esti = np.zeros([Nsamples, 3])
Z_saved = np.zeros(Nsamples)

for i in range(Nsamples):
    Z = GetVolt()
    Xe, Cov, Kg = SimpleKalman(Z)

    X_esti[i] = [Xe, Cov, Kg]
    Z_saved[i] = Z

# plt.plot(t, Z_saved, 'b*--', label='Measurements')
# plt.plot(t, X_esti[:,0], 'ro', label='Kalman Filter')
# plt.legend(loc='upper left')
# plt.ylabel('Volt [V]')
# plt.xlabel('Time [sec]')
#plt.savefig('result/08_SimpleExample.png')

plt.figure()
plt.plot(t, X_esti[:,1], 'o-')
plt.ylabel('Error Covariance')
plt.xlabel('Time [sec]')
plt.savefig('result/08_ErrorCovariance.png')


plt.figure()
plt.plot(t, X_esti[:,2], 'o-')
plt.ylabel('Kalman Gain')
plt.xlabel('Time [sec]')
plt.savefig('result/08_KalmanGain.png')
plt.show()