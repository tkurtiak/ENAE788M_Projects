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





def findCorners(lines):
	#make into a Nx2 rather than Nx1x2
	lines=np.squeeze(lines)
	#yoink out the lines that are horizontal or vertical and place them in seperate arrays
	hor_lines= lines[np.where( (abs(lines[:,1]- np.pi/2) < (np.pi/6))  )]
	vert_lines= lines[np.where( (abs(lines[:,1]) < (np.pi/6)) | (abs(lines[:,1]- np.pi) < (np.pi/6)) )]


	#find intersections of vertical and horizontal lines only, not all lines			
	for i in range(hor_lines.shape[0]):
		for j in range(vert_lines.shape[0]):
			r1=hor_lines[i,0]
			th1=hor_lines[i,1]
			r2=vert_lines[j,0]
			th2=vert_lines[j,1]
			#only th2 can be close to 0 because they are vertical lines

			if th2==0:
				xi=r2
			else:
				xi= ((r2/np.sin(th2))-(r1/np.sin(th1))) / ((np.cos(th2)/np.sin(th2))- (np.cos(th1)/np.sin(th1)))
			yi= (-np.cos(th1)/np.sin(th1))*xi + (r1/np.sin(th1))
			if i==0 and j==0:
				intersections=np.array([[xi,yi]])
			else:
				intersections=np.append(intersections,np.array([[xi,yi]]),axis=0)



	extremea=np.array([np.amax(intersections,axis=0) , np.amin(intersections,axis=0)])
	mid=((extremea[0,:]-extremea[1,:])/2)+extremea[1,:]

	print('inter')
	print(intersections)
	print('extremea')
	print(extremea)
	print('mid')
	print(mid)

	#labels= -1* np.ones((intersections.shape[0],1))
	#print((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]))
	# labels=np.where((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]) , 1 , -1)
	# labels=np.where(intersections[:,0] > mid[0] & intersections[:,1] < mid[1] , 2 , labels )
	# labels=np.where(intersections[:,0] < mid[0] & intersections[:,1] > mid[1] , 3 , labels )
	# labels=np.where(intersections[:,0] < mid[0] & intersections[:,1] < mid[1] , 4 , labels )

	label = 1*((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]))+ 2*((intersections[:,0] < mid[0]) & (intersections[:,1] > mid[1])) + 3*((intersections[:,0] > mid[0]) & (intersections[:,1] < mid[1])) + 4* ((intersections[:,0] < mid[0]) & (intersections[:,1] < mid[1]))
	label=label-1
	print(label)

	centers= np.array( [np.mean(np.squeeze(intersections[np.where(label==0),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==1),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==2),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==3),:]),axis=0)])
	#print(np.squeeze(intersections[np.where(label==0),:]))
	center= np.mean(centers,axis=0)
	print(center)

	# #put these intersections into 4 groups
	# centers,label=kmeans2(intersections,4,iter=10,minit='points')
	# # print(centers)
	# print(label)
	# #here is the true center: not weighted by # of lines made for an edge
	# center=np.mean(centers,axis=0)
	# print(center)


	#stores the max/min (col 0, col1 ) distances in each cluster (row) 
	dist_store=np.array([[0.,999999.],[0.,999999.],[0.,999999.],[0.,999999.]]) 
	inner_corners=np.zeros((4,2))
	outer_corners=np.zeros((4,2))
	#find the minimum and maximum dist in each corner, gives inner and outer corners
	for i in range(intersections.shape[0]):
		dist= np.linalg.norm((intersections[i,:]-center))

		if dist> dist_store[label[i],0]: #if dist is greater that the max (col 0) in its group (label[i])
			dist_store[label[i],0]=dist
			outer_corners[label[i],:]=intersections[i,:]
		if dist< dist_store[label[i],1]: #if dist is less that the min (col 1) in its group (label[i])
			dist_store[label[i],1]=dist
			inner_corners[label[i],:]=intersections[i,:]
	return center, inner_corners, outer_corners




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

mask = cv2.inRange(median, lower, upper)
# print('mask')
# print(mask.shape)
# print(mask)
blank=0*np.ones(np.shape(frame))
res = cv2.bitwise_and(frame, frame, mask = mask)
cv2.imshow('mask',res)
#img=res.copy()




#https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
# Convert the img to grayscale 
gray=res.copy()
#print(img.shape)


gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]

gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=1)
cv2.imshow('eroded1',gray)
gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=2)
cv2.imshow('dialate2',gray)
gray = cv2.dilate(gray,np.ones((5,5), np.uint8),iterations=3)
cv2.imshow('dialate3',gray)
gray = cv2.erode(gray,np.ones((5,5), np.uint8),iterations=3)
cv2.imshow('eroded4',gray)


#temp_gray=cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
temp_gray=gray.copy()
thresh = cv2.threshold(temp_gray, 60, 255, cv2.THRESH_BINARY)[1]

#cv2.imshow('thresh',thresh)
#cv2.waitKey(0)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

maxA=0.

for c in cnts:
	# compute the center of the contour
	A = cv2.contourArea(c)
	if A>maxA:
		maxA=A
		max_contour=c
 
# draw the contour and center of the shape on the image
contour_img=cv2.drawContours(blank.copy(), [max_contour], -1, (255, 255, 255), -1)
#cv2.imshow('biggest contour',contour_img)

contour_img=cv2.dilate(contour_img,np.ones((5,5), np.uint8),iterations=4)
contour_mask = cv2.inRange(contour_img, np.array([1,1,1]), np.array([255,255,255]))

new_gray = cv2.bitwise_and(gray, gray, mask = contour_mask)
#cv2.imshow('eroded proper masked',new_gray)


# Apply edge detection method on the image 
edges = cv2.Canny(new_gray,300,900,apertureSize = 3) 
#edges = cv2.dilate(edges,np.ones((3,3), np.uint8),iterations=1)


cv2.imshow('edges',edges)
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 50) 

# print(lines.shape)
# print(lines)

for i in range(lines.shape[0]): 
	r=lines[i,0,0]
	theta=lines[i,0,1]
	a = np.cos(theta) 
	b = np.sin(theta) 
	x0 = a*r 
	y0 = b*r 
	x1 = int(x0 + 1000*(-b))  
	y1 = int(y0 + 1000*(a)) 
	x2 = int(x0 - 1000*(-b)) 
	y2 = int(y0 - 1000*(a)) 
	cv2.line(imgOG,(x1,y1), (x2,y2), (255,0,255),1)

	if np.abs(theta)<(np.pi/6) or np.abs(theta-np.pi)<(np.pi/6):
		#vertical line
		cv2.line(imgOG,(x1,y1), (x2,y2), (0,0,255),1)

	if np.abs(theta- np.pi/2)<(np.pi/6):
		#horizontal line
		cv2.line(imgOG,(x1,y1), (x2,y2), (0,255,0),1)

cv2.imshow('lines',imgOG)
# plt.imshow(lines)
# plt.show()
# plt.ion()
#cv2.waitKey(0)
center, inner_corners, outer_corners = findCorners(lines)



img_fin[int(center[1]),int(center[0])]= (0,0,255)
for i in range(inner_corners.shape[0]):
	img_fin[int(inner_corners[i,1]),int(inner_corners[i,0])]= (255,0,0)
	img_fin[int(outer_corners[i,1]),int(outer_corners[i,0])]= (0,255,0)
	 
#plt.ion()
cv2.imshow('fin',img_fin)
cv2.waitKey(0)




