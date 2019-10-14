from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import imutils
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

sys.path.insert(0, '/home/vdorbala/ICRA/pylsd')
import cv2

# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from pylsd.lsd import lsd

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def callback(x):
    pass

def select_lines(image, x1, x2, y1, y2, colour):
    lower = colour[0]
    upper = colour[1]

    # blue = np.linspace(lower[0], upper[0], 1)
    # green = np.linspace(lower[1], upper[1], 1)
    # red = np.linspace(lower[2], upper[2], 1)

    if lower[0] < image[y1,x1,0] < upper[0] and lower[0] < image[y2,x2,0] < upper[0]:
        if lower[1] < image[y1,x1,1] < upper[1] and lower[1] < image[y2,x2,1] < upper[1]:
            if lower[2] < image[y1,x1,2] < upper[2] and lower[2] < image[y2,x2,2] < upper[2]:
                return True


def main():

    dirpath = os.getcwd()

    for subdir, dirs, files in os.walk(dirpath + '/Rosbag/bag1_corrected'):
        files.sort()
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):

                print(file)
                # read an image
                image = cv2.imread(filepath)
                cv2.waitKey(0) 
                # Grayscale 
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # gray = cv2.GaussianBlur(gray, (3, 3), 0)
                  
                cv2.imshow("Original Image ", image)
                cv2.waitKey(0)

                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
                # fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

                kernel = np.ones((3,3), np.uint8)

                frame = image.copy()

                # fgmask = fgbg.apply(frame)
                # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # cv2.imshow('frame',fgmask)

                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()
                median = cv2.medianBlur(frame,5)

                # hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

                # Tolerance levels
                tb, tg, tr = 5, 5, 5
                # # BGR!
                # HSV 22, 50, 67; 42 45 100
                lower = np.array([21 - tb, 85 - tg, 0 - tr])
                upper = np.array([92 + tb, 173 + tg, 255 + tr])

                # thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]

                mask = cv2.inRange(median, lower, upper)
                res = cv2.bitwise_and(frame, frame, mask = mask)

                blank = np.zeros([np.shape(res)[0],np.shape(res)[1],3])

                # res = cv2.medianBlur(res,5)
                # res = cv2.erode(res, kernel, iterations=1)
                # res = cv2.dilate(res, kernel, iterations=5)

                # cv2.imshow('frame',frame)
                # cv2.imshow('mask',mask)
                cv2.imshow('res',res)
                
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

                lines = lsd(gray)

                selectarr = []


                for i in xrange(lines.shape[0]):

                    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
                    pt2 = (int(lines[i, 2]), int(lines[i, 3]))

                    x1 = int(lines[i,0])
                    x2 = int(lines[i,2])
                    y1 = int(lines[i,1])
                    y2 = int(lines[i,3])
                    w =  int(lines[i,4])

                    # Theta is the angle the line makes with the horizontal axis.
                    theta = np.arctan2((y2-y1),(x2-x1))
                    if theta >= np.pi:
                        theta = theta - np.pi
                    # l_mag is the length of the line detected.
                    l_mag = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

                    # if(select_lines(frame, x1, x2, y1, y2, [lower, upper]) != True):
                    #     continue

                    # For Horizontal Lines:
                    # if abs(theta) < np.pi/20  and abs(theta) > 0 and abs(l_mag)>40 and abs(l_mag) < 180: 
                    #     cv2.line(res,(x1,y1),(x2,y2),(0,255,0),2)

                    # # For Vertical Lines:
                    # if abs(theta) < np.pi/2 + np.pi/36 and abs(theta) >= np.pi/2 - np.pi/36 and abs(l_mag)>10 and abs(l_mag) < 30: 
                    #     cv2.line(res,(x1,y1),(x2,y2),(0,0,255),2)

                    # For all the lines!
                    if abs(l_mag)>15:
                        if abs(theta) < np.pi/20  and abs(theta) > 0:
                            x = x2 - x1
                            cv2.line(blank,(x1-x,y1),(x2+x,y2),(0,255,0),2)
                            print('Line number is ' + str(i))

                    #     selectarr.append(lines[i])
                    # cv2.line(res,(x1,y1),(x2,y2),(0,255,0),2)


                # blank = cv2.dilate(blank, kernel, iterations=5)
                cv2.imshow('image', blank)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

                # lines = lsd(cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY))

                # for i in xrange(lines.shape[0]):

                #     pt1 = (int(lines[i, 0]), int(lines[i, 1]))
                #     pt2 = (int(lines[i, 2]), int(lines[i, 3]))

                #     x1 = int(lines[i,0])
                #     x2 = int(lines[i,2])
                #     y1 = int(lines[i,1])
                #     y2 = int(lines[i,3])
                #     w =  int(lines[i,4])

                #     # Theta is the angle the line makes with the horizontal axis.
                #     theta = np.arctan2((y2-y1),(x2-x1))
                #     if theta >= np.pi:
                #         theta = theta - np.pi
                #     # l_mag is the length of the line detected.
                #     l_mag = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

                #     # if(select_lines(frame, x1, x2, y1, y2, [lower, upper]) != True):
                #     #     continue

                #     # For Horizontal Lines:
                #     # if abs(theta) < np.pi/20  and abs(theta) > 0 and abs(l_mag)>40 and abs(l_mag) < 180: 
                #     #     cv2.line(res,(x1,y1),(x2,y2),(0,255,0),2)

                #     # # For Vertical Lines:
                #     # if abs(theta) < np.pi/2 + np.pi/36 and abs(theta) >= np.pi/2 - np.pi/36 and abs(l_mag)>10 and abs(l_mag) < 30: 
                #     #     cv2.line(res,(x1,y1),(x2,y2),(0,0,255),2)

                #     # For all the lines!
                #     if abs(l_mag)>15:
                #         cv2.line(blank,(x1,y1),(x2,y2),(0,0,255),2)
                #         # print('Line number is ' + str(i))

                #     #     selectarr.append(lines[i])
                #     # cv2.line(res,(x1,y1),(x2,y2),(0,255,0),2)


                # cv2.imshow('image', blank)

                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()

                
                # Shi-Tomasi                


                # corners = cv2.goodFeaturesToTrack(gray,10,0.1,10)
                # corners = np.int0(corners)


                # for i in corners:
                #     x,y = i.ravel()
                #     cv2.circle(res,(x,y),3,255,-1)

                # cv2.imshow('Shi-Tomasi', res)
                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()


                # Harris


                # dst = cv2.cornerHarris(gray,2,3,0.04)

                # #result is dilated for marking the corners, not important
                # dst = cv2.dilate(dst,None)

                # # Threshold for an optimal value, it may vary depending on the image.
                # res[dst>0.01*dst.max()]=[0,0,255]

                # cv2.imshow('dst',res)
                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()


                # Following code for dynamically updating values and observing the outcome.

                # cv2.namedWindow('image')

                # x1 = 0
                # x2 = 10
                # y1 = 4
                # y2 = 1

                # cv2.createTrackbar('x1','image', x1, 360, callback)
                # cv2.createTrackbar('x2','image', x2, 360, callback)
                # cv2.createTrackbar('y1','image', y1, 360, callback)
                # cv2.createTrackbar('y2','image', y2, 360, callback)



                # while True:
                #     frame = res.copy()
                #     edged = cv2.Canny(gray, 210, 260)

                #     x1 = cv2.getTrackbarPos('thresh','image')
                #     x2 = cv2.getTrackbarPos('minLineLength','image')
                #     y1 = cv2.getTrackbarPos('maxLineGap','image')
                #     y2 = cv2.getTrackbarPos('resolution','image')

                #     # print(minLineLength)
                    
                #     # cv2.imshow("Edge", edged)
                    
                #     # cv2.waitKey(0)

                #     i = 0

                #     theta = np.arctan2((y2-y1),(x2-x1))

                #     print(theta)

                #     lines = cv2.HoughLinesP(edged,10,np.pi/180, 20, 20, 20)
                #     # print(len(lines))

                #     for i in range(len(lines) - 1):
                #         for x1,y1,x2,y2 in lines[i]:
                #             cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

                #     cv2.imshow('image', frame)
                #     k = cv2.waitKey(1000) & 0xFF
                #     if k == 113 or k == 27:
                #         break
                #     # cv2.waitKey(0)

if __name__ == '__main__':
    main()
