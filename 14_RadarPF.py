'''
 Filename: 13_RadarUKF.py
 Created on: April,10, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = None
firstRun = True
Npt = None
pt, wt = None, None
posp = None

def fx(x, dt):
    A = np.eye(3) + dt * np.array([[0,1,0],[0,0,0],[0,0,0]])
    xp = A @ x
    return xp

def hx(x):
    x1, x3 = x[0], x[2]
    yp = np.sqrt(x1**2 + x3**2)
    return yp

def RadarPF(z, dt):
    global firstRun
    global Npt, x, pt, wt
    if firstRun:
        x = np.array([0,90,1100]).T
        Npt = 1000
        pt = x.reshape(-1,1) + 0.1 * x.reshape(-1,1) * np.random.randn(1, Npt) # x : (3,) -> (3,1)
        wt = np.ones((1,Npt)) * (1 / Npt)
        firstRun = False
    else:
        pt = fx(pt, dt) + np.random.randn(3,1)

        wt = wt * norm.pdf(z, hx(pt), 10)

        wt = wt / np.sum(wt)

        x = pt @ np.array(wt).T

        ''' Sequential Importance Resampling '''
        ind = np.random.choice(Npt, Npt, p=wt[0], replace=True)
        pt = pt[:, ind]
        wt = np.ones((1,Npt)) * (1/Npt)

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

for k in range(Nsamples):
    r = GetRadar(dt)
    pos, vel, alt = RadarPF(r, dt)
    Xsaved[k] = [pos, vel, alt]
    Zsaved[k] = r

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
fig.savefig('result/14_RadarPF.png')
plt.show()