#!/usr/bin/env python2
# vertex find
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
#from scipy.cluster.vq import kmeans, whiten, kmeans2
import imutils
#Image Select
#import Tkinter, tkFileDialog
#from time import time





def MaskandApplyCorners2(img):

	center, inner_corners, outer_corners = np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
	# load image
	imgOG = img.copy()
	# img_fin=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# img_fin=cv2.cvtColor(img_fin,cv2.COLOR_GRAY2RGB)
	kernel = np.ones((3,3), np.uint8)
	frame = img.copy()
	median = cv2.medianBlur(frame,5)
	median = cv2.medianBlur(median,5)
	#cv2.imshow('blurred',median)
	# Tolerance levels
	tb, tg, tr = 5, 10, 10
	# # BGR!
	# HSV 22, 50, 67; 42 45 100
	lower = np.array([45 - tb, 47 - tg, 82 - tr])
	upper = np.array([50 + tb, 95 + tg, 135 + tr])


	mask = cv2.inRange(median, lower, upper)
	blank=0*np.ones(np.shape(frame))
	res = cv2.bitwise_and(frame, frame, mask = mask)

	#cv2.imshow('mask',res)


	gray=res.copy()
	gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
	gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
	gray = cv2.dilate(gray,np.ones((5,5), np.uint8),iterations=1)
	gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=2)
	gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=4)
	#cv2.imshow('dialate2',gray)



	temp_gray=gray.copy()
	thresh = cv2.threshold(temp_gray, 60, 255, cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	edges=0*gray.copy()

	countours_exist=None

	maxA=0.

	max_contour=None
	for c in cnts:
		# compute the center of the contour
		A = cv2.contourArea(c)

		countours_exist=1
		if A>maxA:
			maxA=A
			max_contour=c


	

	if countours_exist==1:
		p_min= .15* cv2.arcLength(max_contour,True)
		equi_radius = .5*np.sqrt(4*maxA/np.pi)
		M = cv2.moments(max_contour)
		cx0 = int(M['m10']/M['m00'])
		cy0 = int(M['m01']/M['m00'])


		#print('p_min is: ')
		#print(p_min)
		for c in cnts:
			# compute the center of the contour
			perimeter = cv2.arcLength(c,True)

			if perimeter>p_min:
				
				M = cv2.moments(c)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])

				if np.linalg.norm(np.array([cx-cx0,cy-cy0]))< equi_radius:

		 			cv2.drawContours(edges, [c], -1, (255), 1)
		

	#cv2.imshow('edges',edges)
	

	if countours_exist is None:
		return center, inner_corners, outer_corners
	else:
		# draw the contour and center of the shape on the image
		
		min_line_scale=.3
		if p_min<60:
			min_line_scale=.5
		lines = cv2.HoughLines(edges,1,np.pi/180, int(.3*p_min)) 

		if lines.shape[0]<2:
			return center, inner_corners, outer_corners
		else:
			
			center, inner_corners, outer_corners = findCorners(lines)
			if center is None or np.isnan(center).any() or np.isnan(inner_corners).any() or np.isnan(outer_corners).any():
				return np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
			else:
				return center, inner_corners, outer_corners
				

				
def drawCorners( center, inner_corners, outer_corners,imgOG)

	cv2.circle(imgOG,(int(center[0]),int(center[1])),3,(0,0,255),-1)
	for i in range(inner_corners.shape[0]):
		cv2.circle(imgOG,(int(inner_corners[i,0]),int(inner_corners[i,1])),3,(255,0,0),-1)
		cv2.circle(imgOG,(int(outer_corners[i,0]),int(outer_corners[i,1])),3,(0,255,0),-1)		 	

	cv2.imshow(imgOG)


def findCorners(lines):
	#make into a Nx2 rather than Nx1x2
	lines=np.squeeze(lines)
	#yoink out the lines that are horizontal or vertical and place them in seperate arrays
	hor_lines= lines[np.where( (abs(lines[:,1]- np.pi/2) < (np.pi/6))  )]
	vert_lines= lines[np.where( (abs(lines[:,1]) < (np.pi/6)) | (abs(lines[:,1]- np.pi) < (np.pi/6)) )]

	intersections=None
	center, inner_corners, outer_corners = np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
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


	if intersections is None:
		return center, inner_corners, outer_corners
	else:
		extremea=np.array([np.amax(intersections,axis=0) , np.amin(intersections,axis=0)])
		mid=((extremea[0,:]-extremea[1,:])/2)+extremea[1,:]

		# print('inter')
		# print(intersections)
		# print('extremea')
		# print(extremea)
		# print('mid')
		# print(mid)

		#labels= -1* np.ones((intersections.shape[0],1))
		#print((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]))
		# labels=np.where((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]) , 1 , -1)
		# labels=np.where(intersections[:,0] > mid[0] & intersections[:,1] < mid[1] , 2 , labels )
		# labels=np.where(intersections[:,0] < mid[0] & intersections[:,1] > mid[1] , 3 , labels )
		# labels=np.where(intersections[:,0] < mid[0] & intersections[:,1] < mid[1] , 4 , labels )

		label = 1*((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]))+ 2*((intersections[:,0] < mid[0]) & (intersections[:,1] > mid[1])) + 3*((intersections[:,0] > mid[0]) & (intersections[:,1] < mid[1])) + 4* ((intersections[:,0] < mid[0]) & (intersections[:,1] < mid[1]))
		label=label-1
		# print(label)

		centers= np.array( [np.mean(np.squeeze(intersections[np.where(label==0),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==1),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==2),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==3),:]),axis=0)])
		#print(np.squeeze(intersections[np.where(label==0),:]))
		center= np.mean(centers,axis=0)
		# print(center)

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


def draw_lines(lines,frame):
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
		cv2.line(frame,(x1,y1), (x2,y2), (255,0,255),1)

		if np.abs(theta)<(np.pi/6) or np.abs(theta-np.pi)<(np.pi/6):
			#vertical line
			cv2.line(frame,(x1,y1), (x2,y2), (0,0,255),1)

		if np.abs(theta- np.pi/2)<(np.pi/6):
			#horizontal line
			cv2.line(frame,(x1,y1), (x2,y2), (0,255,0),1)
	
	cv2.imshow('lines',frame)




#center, inner_corners, outer_corners= MaskandApplyCorners2(img)



cv2.waitKey(0)

# When everything done, release the capture
cv2.destroyAllWindows()