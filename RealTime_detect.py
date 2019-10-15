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
from m021v4l2 import Capture640x480
from time import time




def findCorners(lines):
	#make into a Nx2 rather than Nx1x2
	lines=np.squeeze(lines)
	#yoink out the lines that are horizontal or vertical and place them in seperate arrays
	hor_lines= lines[np.where( (abs(lines[:,1]- np.pi/2) < (np.pi/6))  )]
	vert_lines= lines[np.where( (abs(lines[:,1]) < (np.pi/6)) | (abs(lines[:,1]- np.pi) < (np.pi/6)) )]

	intersections=None
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
		return None, None, None
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



def MaskandApplyCorners(img):


	# load image
	imgOG = img.copy()
	# img_fin=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	# img_fin=cv2.cvtColor(img_fin,cv2.COLOR_GRAY2RGB)
	kernel = np.ones((3,3), np.uint8)
	frame = img.copy()
	median = cv2.medianBlur(frame,5)

	# Tolerance levels
	tb, tg, tr = 15, 15, 15
	# # BGR!
	# HSV 22, 50, 67; 42 45 100
	lower = np.array([21 - tb, 85 - tg, 0 - tr])
	upper = np.array([92 + tb, 173 + tg, 255 + tr])

	# tb, tg, tr = 50, 50, 50
	# lower = np.array([85 - tb, 0 - tg, 21 - tr])
	# upper = np.array([173 + tb, 255 + tg, 92 + tr])
	# thresh = cv2.threshold(gray, 10, 100, cv2.THRESH_BINARY)[1]

	mask = cv2.inRange(median, lower, upper)
	blank=0*np.ones(np.shape(frame))
	res = cv2.bitwise_and(frame, frame, mask = mask)

	gray=res.copy()
	gray = cv2.cvtColor(gray,cv2.COLOR_RGB2GRAY) 
	gray = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
	gray = cv2.erode(gray,np.ones((3,3), np.uint8),iterations=1)
	gray = cv2.dilate(gray,np.ones((3,3), np.uint8),iterations=4)

	temp_gray=gray.copy()
	thresh = cv2.threshold(temp_gray, 60, 255, cv2.THRESH_BINARY)[1]

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	maxA=0.

	max_contour=None
	for c in cnts:
		# compute the center of the contour
		A = cv2.contourArea(c)
		if A>maxA:
			maxA=A
			max_contour=c
	

	if max_contour is None:
		return imgOG
	else:
		# draw the contour and center of the shape on the image
		contour_img=cv2.drawContours(blank.copy(), [max_contour], -1, (255, 255, 255), -1)
		#cv2.imshow('biggest contour',contour_img)

		contour_img=cv2.dilate(contour_img,np.ones((5,5), np.uint8),iterations=4)
		contour_mask = cv2.inRange(contour_img, np.array([1,1,1]), np.array([255,255,255]))

		new_gray = cv2.bitwise_and(gray, gray, mask = contour_mask)
		# #cv2.imshow('eroded proper masked',new_gray)

		# Apply edge detection method on the image 
		edges = cv2.Canny(new_gray,300,900,apertureSize = 3) 

		#cv2.imshow('edges',edges)
		# This returns an array of r and theta values 

		lines=None
		lines = cv2.HoughLines(edges,1,np.pi/180, 40) 

		if lines is None:
			return imgOG
		else:



			center, inner_corners, outer_corners = findCorners(lines)

			if center is None or np.isnan(center).any() or np.isnan(inner_corners).any() or np.isnan(outer_corners).any():
				return imgOG
			else:

				cv2.circle(imgOG,(int(center[0]),int(center[1])),3,(0,0,255),-1)
				for i in range(inner_corners.shape[0]):
					cv2.circle(imgOG,(int(inner_corners[i,0]),int(inner_corners[i,1])),3,(255,0,0),-1)
					cv2.circle(imgOG,(int(outer_corners[i,0]),int(outer_corners[i,1])),3,(0,255,0),-1)

				return imgOG

def colour_correct(img):

	width, height = img.shape[:2]
	# convert image to RGB color for matplotlib
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# find average per row
	# np.average() takes in an axis argument which finds the average across that axis. 
	average_color_per_row = np.average(img, axis=0)

	# find average across average per row
	average_color = np.average(average_color_per_row, axis=0)

	# convert back to uint8
	average_color = np.uint8(average_color)

	if average_color[0] < 50 or average_color[1] < 50 or average_color[2] < 50:
	    contrast = 50
	else:
	    contrast = 65
	# take color compliment of the average value 
	print(contrast)
	average_color[0] = 255-average_color[0]
	average_color[1] = 255-average_color[1]
	average_color[2] = 255-average_color[2]

	# create height x width pixel array with average color value
	average_color_img = np.array([[average_color]*height]*width, np.uint8)

	# add the color compliment to the original image, each with 50% weights
	dst = cv2.addWeighted(img, 0.5, average_color_img, 0.5, 0)

	# Increase contrast of the resultant image #90 for bag1 64 for bag2
	f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
	alpha_c = f
	gamma_c = 127*(1-f)
	dst = cv2.addWeighted(dst, alpha_c, dst, 0, gamma_c)

	dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

	return dst

cap = Capture640x480()

start = time()

while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # frame = colour_correct(frame)
    # Display the resulting frame
    frame_mod=MaskandApplyCorners(frame)

    cv2.imshow('LI-USB30-M021',frame_mod)
    if cv2.waitKey(1) & 0xFF == 27:
        break

count = cap.getCount()
elapsed = time() - start
# print('%d frames in %3.2f seconds = %3.2f fps' % (count, elapsed, count/elapsed))

# When everything done, release the capture
cv2.destroyAllWindows()

