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
import Tkinter, tkFileDialog
from time import time




def applyCorners2Mask(mask,img):

	center, inner_corners, outer_corners = np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
	# # load image
	# imgOG = img.copy()
	# frame = img.copy()
	# median = cv2.medianBlur(frame,5)
	# median = cv2.medianBlur(median,5)
	# #cv2.imshow('blurred',median)
	# # Tolerance levels
	# tb, tg, tr = 5, 10, 10
	# # # BGR!
	# # HSV 22, 50, 67; 42 45 100
	# lower = np.array([45 - tb, 47 - tg, 82 - tr])
	# upper = np.array([50 + tb, 95 + tg, 135 + tr])


	# mask = cv2.inRange(median, lower, upper)
	# blank=0*np.ones(np.shape(frame))
	frame=img.copy()
	res = cv2.bitwise_and(frame, frame, mask = mask)

	#cv2.imshow('mask',res)


	gray=res.copy()
	gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
	gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
	gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=4)
	gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=3)
	#gray = cv2.dilate(gray,np.ones((5,5), np.uint8),iterations=3)
	cv2.imshow('dialate2',gray)



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

		# print('p_min is: ')
		# print(p_min)
		for c in cnts:
			# compute the center of the contour
			perimeter = cv2.arcLength(c,True)

			if perimeter>p_min:
				
				M = cv2.moments(c)
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])

				if np.linalg.norm(np.array([cx-cx0,cy-cy0]))< equi_radius:

		 			cv2.drawContours(edges, [c], -1, (255), 1)
		

	cv2.imshow('edges',edges)
	

	if countours_exist is None:
		return center, inner_corners, outer_corners
	else:
		# draw the contour and center of the shape on the image
		
		min_line_scale=.3
		if p_min<60:
			min_line_scale=.5
		#lines = cv2.HoughLines(edges,1,np.pi/180, int(.3*p_min)) 

		# start_t=time()
		lines = cv2.HoughLinesP (edges, 1, np.pi/180, int(.3*p_min), minLineLength = 10, maxLineGap = 3)
		# print("--- %s s HoughLinesP ---" % (time() - start_t))
		# cv2.imshow('lines drawn',draw_lines_P(lines,imgOG))
		#print(lines.shape)


		if lines is None:
			return center, inner_corners, outer_corners
		if lines.shape[0]<2:
			return center, inner_corners, outer_corners
		else:
			
			# start_t=time()
			center, inner_corners, outer_corners = findCornersP(lines)
			# print("--- %s s find corners P ---" % (time() - start_t))

			# for i in range(intersections.shape[0]):
			# 	cv2.circle(imgOG,(int(intersections[i,0]),int(intersections[i,1])),2,(255,0,255),-1)
			# cv2.imshow('intersections',imgOG)

			if center is None or np.isnan(center).any() or np.isnan(inner_corners).any() or np.isnan(outer_corners).any():
				return np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
			else:
				return center, inner_corners, outer_corners
				
def drawCorners( center, inner_corners, outer_corners,imgOG):

	cv2.circle(imgOG,(int(center[0,0]),int(center[0,1])),3,(0,0,255),-1)
	for i in range(inner_corners.shape[0]):
		cv2.circle(imgOG,(int(inner_corners[i,0]),int(inner_corners[i,1])),3,(255,0,0),-1)
		cv2.circle(imgOG,(int(outer_corners[i,0]),int(outer_corners[i,1])),3,(0,255,0),-1)		 	

	return imgOG
	

def findCornersP(lines):
	#make into a Nx2 rather than Nx1x2
	lines=np.squeeze(lines)
	# print(lines.shape)
	# print('lines')
	# print(lines)
	#[x1 y1 x2 y2]
	#yoink out the lines that are horizontal or vertical and place them in seperate arrays

	#+.001 is a hack to prevent div by 0
	vert_lines= lines[np.where( ( abs(lines[:,2]-lines[:,0])/(abs(lines[:,3]-lines[:,1])+.001) < .55 ) )] #vertical
	hor_lines= lines[np.where( (abs(lines[:,3]-lines[:,1])/(abs(lines[:,2]-lines[:,0])+.001) < .55) )]

	# print('hor lines')
	# print(hor_lines)
	# print('vert lines')
	# print(vert_lines)

	intersections=None
	#find intersections of vertical and horizontal lines only, not all lines            
	for i in range(hor_lines.shape[0]):
		for j in range(vert_lines.shape[0]):

			#point on hor line
			x1=hor_lines[i,0]
			y1=hor_lines[i,1]
			#point on vert line
			x2=vert_lines[j,0]
			y2=vert_lines[j,1]

			m1= float(hor_lines[i,3]-y1)/(hor_lines[i,2]-x1)#slope of hor line

			if (vert_lines[j,2]-x2)==0:
				#perfectly vertical line, so avoid div by 0 error
				xi=x2
				yi= m1*(xi-x1)+y1
			else:
				m2= float(vert_lines[j,3]-y2)/(vert_lines[j,2]-x2)
				
				#intersection of 2 lines in point-slope form
				xi= (m2*x2 - m1*x1 + y1 - y2)/(m2-m1)
				yi= m1*(xi-x1)+y1

				# print(m1)
				# print(m2)

			if i==0 and j==0:
				intersections=np.array([[xi,yi]])
			else:
				intersections=np.append(intersections,np.array([[xi,yi]]),axis=0)
				


	if intersections is None:
		return np.zeros((1,2)),np.zeros((4,2)),np.zeros((4,2))
	else:
		extremea=np.array([np.amax(intersections,axis=0) , np.amin(intersections,axis=0)])
		mid=((extremea[0,:]-extremea[1,:])/2)+extremea[1,:]

		#extremea: max_x, max_y; min_x,min_y
		# print('inter')
		# print(intersections)
		# print('extremea')
		# print(extremea)
		# print('mid')
		# print(mid)

		#throw out points in the middle of maxima
		#horizontally
		intersections= np.delete(intersections, np.where( (.3<= (intersections[:,0]-extremea[1,0])/(extremea[0,0]-extremea[1,0])) & ( (intersections[:,0]-extremea[1,0])/(extremea[0,0]-extremea[1,0]) <= .7 )), axis=0)
		#vertically
		#at an angle the close vertical side will be very large in pixels, while the far can be quite small, so range is bigger for these intersections
		intersections= np.delete(intersections, np.where( (.4<= (intersections[:,1]-extremea[1,1])/(extremea[0,1]-extremea[1,1])) & ( (intersections[:,1]-extremea[1,1])/(extremea[0,1]-extremea[1,1]) <= .6 )), axis=0)

		# print('inter after delete')
		# print(intersections)

		label = 1*((intersections[:,0] > mid[0]) & (intersections[:,1] > mid[1]))+ 2*((intersections[:,0] < mid[0]) & (intersections[:,1] > mid[1])) + 3*((intersections[:,0] > mid[0]) & (intersections[:,1] < mid[1])) + 4* ((intersections[:,0] < mid[0]) & (intersections[:,1] < mid[1]))
		label=label-1
		# print(label)

		centers= np.array( [np.mean(np.squeeze(intersections[np.where(label==0),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==1),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==2),:]),axis=0),np.mean(np.squeeze(intersections[np.where(label==3),:]),axis=0)])
		#print(np.squeeze(intersections[np.where(label==0),:]))
		center_temp= np.mean(centers,axis=0)

		center= np.array([[center_temp[0],center_temp[1]]])
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
	
def draw_lines_P(lines,frame):
	lines=np.squeeze(lines)

	print(lines)
	for i in range(lines.shape[0]):
		
		cv2.line(frame,(lines[i,0],lines[i,1]),(lines[i,2],lines[i,3]),(0,255,0),2)
	
	return frame

def makeMask(img,icovariance,mu):
	thresh  = .12

	frame = img.copy()
	median = cv2.medianBlur(frame,5)
	median = cv2.medianBlur(median,5)

	# cv2.imshow('blurred',median)
	tb, tg, tr = 15, 15, 20
	lower = np.array([42 - tb, 47 - tg, 82 - tr])
	upper = np.array([50 + tb, 100 + tg, 135 + tr])

	mask = cv2.inRange(median, lower, upper)
	blank=0*np.ones(np.shape(frame))
	res = cv2.bitwise_and(median, median, mask = mask)
	cv2.imshow('thresholded',res)


	# start_t=time()

	# this code inspired by BRZ
	img_flat= np.reshape(res, (res.shape[0]*res.shape[1] , 3))
	indx= np.where(img_flat[:,0] > 1)
	diff= np.matrix(img_flat-mu)
	p_flat=np.zeros((diff.shape[0],1))
	p_flat[indx]= np.exp( -.5* np.sum( np.multiply( diff[indx] * icovariance, diff[indx]),axis=1))
	p= np.reshape(p_flat, (res.shape[0],res.shape[1]))
	#end inspired by BRZ
	#note, we hd it working but slowly, they suggested this nice reshape vector operation for speed

	mask = cv2.inRange(p,thresh,1)
	return mask
	#res2 = cv2.bitwise_and(frame,frame,mask = mask)

	#print("--- %s s gaussian calc ---" % (time() - start_t))

	#cv2.imshow('SGauss thresholded',res2)


root = Tkinter.Tk()
root.withdraw()
imgname = tkFileDialog.askopenfilename()

print('trying to load image')
print(imgname)
img = cv2.imread(imgname)
print('loaded image')


filename = 'Final_GoodData.npy'
BGR_set = np.load(filename)[0:3]
covariance = np.cov(BGR_set)
icovariance = np.linalg.inv(covariance)
mu = np.mean(BGR_set,axis = 1)

start_t=time()
mask= makeMask(img,icovariance,mu)

cv2.imshow('mask',cv2.bitwise_and(img, img, mask = mask))

center, inner_corners, outer_corners= applyCorners2Mask(mask,img)
fin_img=drawCorners(center, inner_corners, outer_corners,img)
print("--- %s full operation ---" % (time() - start_t))

cv2.imshow('fin img',fin_img)




cv2.waitKey(0)

# When everything done, release the capture
cv2.destroyAllWindows()