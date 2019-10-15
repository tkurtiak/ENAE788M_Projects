#!/usr/bin/env python2
# vertex find


import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans, whiten, kmeans2
import imutils
#Image Select
import Tkinter, tkFileDialog


root = Tkinter.Tk()
root.withdraw()
imgname = tkFileDialog.askopenfilename()

# imgname='Images/Yellow/IMG_20191007_114115484_HDR.jpg'
# Selected training img
#imgname = 'Images/InitislSet.png'

# Save/Load File name for RGB values
#filename = 'MaroonWindow.npy'



print('trying to load image')
# load image
img = cv2.imread(imgname)
imgOG = cv2.imread(imgname)
img_fin=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img_fin=cv2.cvtColor(img_fin,cv2.COLOR_GRAY2RGB)
print('loaded image')


#start with a blur
kernel = np.ones((3,3), np.uint8)
frame = img.copy()
median = cv2.medianBlur(frame,5)

# hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

# Tolerance levels
tb, tg, tr = 5, 5, 5
# # BGR!
# HSV 22, 50, 67; 42 45 100
lower = np.array([21 - tb, 85 - tg, 0 - tr])
upper = np.array([92 + tb, 173 + tg, 255 + tr])

# tb, tg, tr = 50, 50, 50
# lower = np.array([85 - tb, 0 - tg, 21 - tr])
# upper = np.array([173 + tb, 255 + tg, 92 + tr])
# thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]

#use a basic color thresholding (conservative, don't get a lot of background)
mask = cv2.inRange(median, lower, upper)
#this for later
blank=0*np.ones(np.shape(frame))
#get a mask out from color thresholding
res = cv2.bitwise_and(frame, frame, mask = mask)
cv2.imshow('base mask',res)


#make a pure white mask called gray:
gray=res.copy()
gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

#erode out the noise
gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=1)
#fine dilation
gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=2)
#thicc dialation
gray = cv2.dilate(gray,np.ones((5,5), np.uint8),iterations=3)
cv2.imshow('dialate3',gray)



#use this thick dialated image to find contours: 
temp_gray=gray.copy()
#reformat for contouring
thresh = cv2.threshold(temp_gray, 60, 255, cv2.THRESH_BINARY)[1]
#contours:
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#find maximum contour:
maxA=0.
for c in cnts:
	A = cv2.contourArea(c)
	if A>maxA:
		maxA=A
		max_contour=c
 
# draw the max contour filled in
contour_img=cv2.drawContours(blank.copy(), [max_contour], -1, (255, 255, 255), -1)
cv2.imshow('biggest contour',contour_img)

#make it bigger to get the edges right: (MAY NOT NEED TO)
contour_img=cv2.dilate(contour_img,np.ones((5,5), np.uint8),iterations=4)
#make a mask out of it
contour_mask = cv2.inRange(contour_img, np.array([1,1,1]), np.array([255,255,255]))

#apply the mask to the original thickened mask
new_gray = cv2.bitwise_and(gray, gray, mask = contour_mask)
cv2.imshow('hollow fat mask',new_gray)

#make a mask out of that
hollow_mask= cv2.inRange(new_gray, 200, 270)
#apply to original image
colored_masked= cv2.bitwise_and(frame, frame, mask = hollow_mask)
cv2.imshow('applied hollow fat mask',colored_masked)


#do second liberal color masking here!




cv2.waitKey(0)




