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

                frame = image.copy()

                # fgmask = fgbg.apply(frame)
                # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # cv2.imshow('frame',fgmask)

                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()

                # Find Canny edges 

                # Tolerance levels
                tb, tg, tr = 30, 400, 400
                # # BGR!
                lower = np.array([98 - tb,180 - tg,83 - tr])
                upper = np.array([70 + tb,175 + tg,180 + tr])

                # thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]
                # thresh = cv2.erode(thresh, None, iterations=2)
                # thresh = cv2.dilate(thresh, None, iterations=2)

                median = cv2.medianBlur(frame,5)

                mask = cv2.inRange(median, lower, upper)
                res = cv2.bitwise_and(frame, frame, mask = mask)

                # cv2.imshow('frame',frame)
                # cv2.imshow('mask',mask)
                cv2.imshow('res',res)
                
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

                # thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]
                # thresh = cv2.erode(thresh, None, iterations=2)
                # thresh = cv2.dilate(thresh, None, iterations=2)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

                lines = lsd(gray)

                for i in xrange(lines.shape[0]):

                    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
                    pt2 = (int(lines[i, 2]), int(lines[i, 3]))

                    x1 = int(lines[i,0])
                    x2 = int(lines[i,2])
                    y1 = int(lines[i,1])
                    y2 = int(lines[i,3])
                    w =  int(lines[i, 4])

                    # Theta is the angle the line makes with the horizontal axis.
                    theta = np.arctan2((y2-y1),(x2-x1))
                    # l_mag is the length of the line detected.
                    l_mag = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

                    # For Horizontal Lines:
                    # if theta < np.pi/30 and theta > -np.pi/30 and abs(l_mag)>75 and abs(l_mag) < 180: 
                    #     cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    # # For Vertical Lines:
                    # if theta < np.pi*26/20 and theta > -np.pi*26/20 and abs(l_mag)>70 and abs(l_mag) < 110: 
                        # cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    # For all the lines!
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.imshow('image', frame)
                k = cv2.waitKey(1000) & 0xFF
                if k == 113 or k == 27:
                    break
                # cv2.waitKey(0)

                
                # # # Shi-Tomasi                


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

                # Finding Contours 
                # Use a copy of the image e.g. edged.copy() 
                # since findContours alters the image 

                # contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                #     cv2.CHAIN_APPROX_SIMPLE)

                # contours = imutils.grab_contours(contours)
                # print(contours)
                # c = max(contours, key=cv2.contourArea)

                # # image, contours, hierarchy = cv2.findContours(edged,  
                # #     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                
                # extLeft = tuple(c[c[:, :, 0].argmin()][0])
                # extRight = tuple(c[c[:, :, 0].argmax()][0])
                # extTop = tuple(c[c[:, :, 1].argmin()][0])
                # extBot = tuple(c[c[:, :, 1].argmax()][0])

                # cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
                # # cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
                # # cv2.circle(image, extRight, 8, (0, 255, 0), -1)
                # # cv2.circle(image, extTop, 8, (255, 0, 0), -1)
                # # cv2.circle(image, extBot, 8, (255, 255, 0), -1)
                 
                # # show the output image
                # cv2.imshow("Image", image)
                # cv2.waitKey(0)


                # cv2.imshow('Canny Edges After Contouring', edged) 
                # cv2.waitKey(0) 
                  
                # print("Number of Contours found = " + str(len(contours))) 
                  
                # # print("Hierarchy is {}".format(np.shape(contours[0])))
                # # Draw all contours 
                # # -1 signifies drawing all contours

                # # r, g, b = cv2.split(image)

                # # for i, c in enumerate(contours):
                # #     area = cv2.contourArea(c)
                # #     if area > 100 and area < 300:
                # #         # if b<98 and b >70
                # #         cv2.drawContours(image, contours, i, (255, 255, 255), 3)      
                # #         cv2.imshow('Contours', image) 
                # #         cv2.waitKey(0) 
                # # #         cv2.destroyAllWindows()
                # cv2.namedWindow('image')

                # threshold = 0
                # minLineLength = 10
                # maxLineGap = 4
                # limit = 1

                # cv2.createTrackbar('thresh','image', threshold, 100, callback)
                # cv2.createTrackbar('minLineLength','image', minLineLength, 100, callback)
                # cv2.createTrackbar('maxLineGap','image', maxLineGap, 100, callback)
                # cv2.createTrackbar('resolution','image', limit, 100, callback)



                # while True:
                #     frame = res.copy()
                #     edged = cv2.Canny(gray, 210, 260)

                #     threshold = cv2.getTrackbarPos('thresh','image')
                #     minLineLength = cv2.getTrackbarPos('minLineLength','image')
                #     maxLineGap = cv2.getTrackbarPos('maxLineGap','image')
                #     limit = cv2.getTrackbarPos('resolution','image')

                #     print(minLineLength)
                    
                #     # cv2.imshow("Edge", edged)
                    
                #     # cv2.waitKey(0)

                #     i = 0

                #     lines = cv2.HoughLinesP(edged,limit,np.pi/180, threshold, minLineLength, maxLineGap)
                #     print(len(lines))

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