'''
 Filename: 01_AvgFilter.py
 Created on: April, 3, 2021
 Author: dhpark
'''
import numpy as np
import matplotlib.pyplot as plt

prevAvg = 0
k = 1

def AvgFilter(x):
    global k, prevAvg
    alpha = (k-1) / k
    avg = alpha * prevAvg + (1 - alpha)*x
    prevAvg = avg
    k += 1
    return avg

def GetVolt():
    return 14.4 + np.random.normal(0, 4, 1)

t = np.arange(0, 10, 0.2)
Nsamples = len(t)

Avgsaved = np.zeros(Nsamples)
Xmsaved = np.zeros(Nsamples)

for i in range(Nsamples):
    xm = GetVolt()
    avg = AvgFilter(xm)

    Avgsaved[i] = avg
    Xmsaved[i] = xm

plt.plot(t, Xmsaved, 'b*--', label='Measured')
plt.plot(t, Avgsaved, 'ro', label='Average')
plt.legend(loc='upper left')
plt.ylabel('Volt [V]')
plt.xlabel('Time [sec]')
plt.savefig('result/01_average_filter.png')
plt.show()