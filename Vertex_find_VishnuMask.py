#!/usr/bin/env python2
# vertex find


import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans, whiten, kmeans2
#Image Select
import Tkinter, tkFileDialog





def findCorners(lines):
	#make into a Nx2 rather than Nx1x2
	lines=np.squeeze(lines)
	#yoink out the lines that are horizontal or vertical and place them in seperate arrays
	hor_lines= lines[np.where(abs(lines[:,1]- np.pi/2) < (np.pi/6))]
	vert_lines= lines[np.where(abs(lines[:,1]) < (np.pi/6))]


	#find intersections of vertical and horizontal lines only, not all lines			
	for i in range(hor_lines.shape[0]):
		for j in range(vert_lines.shape[0]):
			r1=hor_lines[i,0]
			th1=hor_lines[i,1]
			r2=vert_lines[j,0]
			th2=vert_lines[j,1]
			xi= ((r2/np.sin(th2))-(r1/np.sin(th1))) / ((np.cos(th2)/np.sin(th2))- (np.cos(th1)/np.sin(th1)))
			yi= (-np.cos(th1)/np.sin(th1))*xi + (r1/np.sin(th1))
			if i==0 and j==0:
				intersections=np.array([[xi,yi]])
			else:
				intersections=np.append(intersections,np.array([[xi,yi]]),axis=0)


	#print(intersections)
	extremea=np.array([np.amax(intersections,axis=0) , np.amin(intersections,axis=0)])
	mid=extremea[0,:]-extremea[1,:]
	#print(mid)

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
print('loaded image')



kernel = np.ones((3,3), np.uint8)
frame = img.copy()
median = cv2.medianBlur(frame,5)

# hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)

# Tolerance levels
#tb, tg, tr = 5, 5, 5
# # BGR!
# HSV 22, 50, 67; 42 45 100
# lower = np.array([21 - tb, 85 - tg, 0 - tr])
# upper = np.array([92 + tb, 173 + tg, 255 + tr])

tb, tg, tr = 50, 50, 50
lower = np.array([85 - tb, 0 - tg, 21 - tr])
upper = np.array([173 + tb, 255 + tg, 92 + tr])
# thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]

mask = cv2.inRange(median, lower, upper)
blank=255*np.ones(np.shape(frame))
res = cv2.bitwise_and(frame, frame, mask = mask)
cv2.imshow('mask',res)
img=res.copy()




#https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
# Convert the img to grayscale 
gray=img.copy()
print(img.shape)


#gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) 

gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=1)
gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=4)




cv2.imshow('eroded',gray)
# Apply edge detection method on the image 
edges = cv2.Canny(gray,300,900,apertureSize = 3) 



cv2.imshow('edges',edges)
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 40) 

# print(lines.shape)
# print(lines)

for i in range(lines.shape[0]): 
	r=lines[i,0,0]
	theta=lines[i,0,1]

	
	# Stores the value of cos(theta) in a 
	a = np.cos(theta) 

	# Stores the value of sin(theta) in b 
	b = np.sin(theta) 
	
	# x0 stores the value rcos(theta) 
	x0 = a*r 
	
	# y0 stores the value rsin(theta) 
	y0 = b*r 
	
	# x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
	x1 = int(x0 + 1000*(-b)) 
	
	# y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
	y1 = int(y0 + 1000*(a)) 

	# x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
	x2 = int(x0 - 1000*(-b)) 
	
	# y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
	y2 = int(y0 - 1000*(a)) 

	cv2.line(img,(x1,y1), (x2,y2), (255,0,255),1)
	#splits up lines into verticals and horizontals
	#print(theta)
	if np.abs(theta)<(np.pi/6) or np.abs(theta-np.pi)<(np.pi/6):
		#vertical line
		cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1)

	if np.abs(theta- np.pi/2)<(np.pi/6):
		#horizontal line
		cv2.line(img,(x1,y1), (x2,y2), (0,255,0),1)

cv2.imshow('lines',img)
# plt.imshow(lines)
# plt.show()
# plt.ion()
cv2.waitKey(0)
center, inner_corners, outer_corners = findCorners(lines)


imgOG[int(center[1]),int(center[0])]= (0,0,255)
for i in range(inner_corners.shape[0]):
	imgOG[int(inner_corners[i,1]),int(inner_corners[i,0])]= (255,0,0)
	imgOG[int(outer_corners[i,1]),int(outer_corners[i,0])]= (0,255,0)
	 
#plt.ion()
cv2.imshow('inter',imgOG)
cv2.waitKey(0)




