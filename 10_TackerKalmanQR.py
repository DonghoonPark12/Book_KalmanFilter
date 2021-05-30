'''
 Filename: 10_TackerKalmanQR.py
 Created on: April,3, 2021
 Author: dhpark
'''
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity

firstRun = True
X, P, A, H, Q, R = 0, 0, 0, 0, 0, 0
firstRun2 = True
X2, P2, A2, H2, Q2, R2 = 0, 0, 0, 0, 0, 0

def GetBallPos(iimg=0):
    """Return measured position of ball by comparing with background image file.
        - References:
        (1) Data Science School:
            https://datascienceschool.net/view-notebook/f9f8983941254a34bf0fee42c66c5539
        (2) Image Diff Calculation:
            https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python
    """
    # Read images.
    imageA = cv2.imread('./10_Img/bg.jpg')
    imageB = cv2.imread('./10_Img/{}.jpg'.format(iimg+1))

    # Convert the images to grayscale.
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images,
    # ensuring that the difference image is returned.
    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8')

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    xc = int(M['m10'] / M['m00'])  # center of x as true position.
    yc = int(M['m01'] / M['m00'])  # center of y as true position.

    v = np.random.normal(0, 15)  # v: measurement noise of position.

    xpos_meas = xc + v  # x_pos_meas: measured position in x (observable).
    ypos_meas = yc + v  # y_pos_meas: measured position in y (observable).

    return np.array([xpos_meas, ypos_meas])

def TrackKalman(xm, ym):
    global firstRun
    global A, Q, H, R
    global X, P
    if firstRun:
        dt = 1
        A = np.array([[1,dt,0,0], [0,1,0,0], [0,0,1,dt],[0,0,0,1]])
        H = np.array([[1,0,0,0],[0,0,1,0]])
        Q = np.eye(4)
        R = np.array([[50, 0],[0, 50]])

        X = np.array([0,0,0,0]).transpose()
        P = 100 * np.eye(4)
        firstRun = False

    Xp = A @ X # Xp : State Variable Prediction
    Pp = A @ P @ A.T + Q # Error Covariance Prediction

    K = (Pp @ H.T) @ inv(H@Pp@H.T + R) # K : Kalman Gain

    z = np.array([xm, ym]).transpose()
    X = Xp + K@(z - H@Xp) # Update State Variable Estimation
    P = Pp - K@H@Pp # Update Error Covariance Estimation

    xh = X[0]
    yh = X[2]

    return xh, yh

def TrackKalmanQR(xm, ym):
    global firstRun2
    global A2, Q2, H2, R2
    global X2, P2
    if firstRun2:
        dt = 1
        A2 = np.array([[1,dt,0,0], [0,1,0,0], [0,0,1,dt],[0,0,0,1]])
        H2 = np.array([[1,0,0,0],[0,0,1,0]])
        Q2 = 0.01 * np.eye(4)
        R2 = np.array([[50, 0],[0, 50]])

        X2 = np.array([0,0,0,0]).transpose()
        P2 = 100 * np.eye(4)
        firstRun2 = False

    Xp = A2 @ X2 # Xp : State Variable Prediction
    Pp = A2 @ P2 @ A2.T + 0.01 * Q2 # Error Covariance Prediction

    K = (Pp @ H2.T) @ inv(H2@Pp@H2.T + R2) # K : Kalman Gain

    z = np.array([xm, ym]).transpose()
    X2 = Xp + K@(z - H2@Xp) # Update State Variable Estimation
    P2 = Pp - K@H2@Pp # Update Error Covariance Estimation

    xh = X2[0]
    yh = X2[2]

    return xh, yh

NoOfImg = 24
#Xmsaved = np.zeros([NoOfImg,2])
Xhsaved = np.zeros([NoOfImg,2])
Xqrsaved = np.zeros([NoOfImg,2])

for k in range(NoOfImg):
    xm, ym = GetBallPos(k)
    xh, yh = TrackKalman(xm, ym)
    xqr, yqr = TrackKalmanQR(xm, ym)

    #Xmsaved[k] = [xm, ym]
    Xhsaved[k] = [xh, yh]
    Xqrsaved[k] = [xqr, yqr]

plt.figure()
#plt.plot(Xmsaved[:,0], Xmsaved[:,1], '*', label='Measured')
plt.plot(Xhsaved[:,0], Xhsaved[:,1], 's', label='Q')
plt.plot(Xqrsaved[:,0], Xqrsaved[:,1], 'o', label='1/100 Q')
plt.legend(loc='upper left')
plt.ylabel('Vertical [pixel]')
plt.xlabel('Horizontal [pixel]')
plt.ylim([0, 250])
plt.xlim([0, 350])
plt.plot([0, 350], [250, 0], '-')
plt.gca().invert_yaxis()
plt.savefig('result/10_TrackerKalmanQR.png')
plt.show()