#!/usr/bin/env python

# starter image subscriber code from: http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from matplotlib import pyplot as plt
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import color_correct as cc

class image_converter:
	def __init__(self):
		# Load GMM
		#GMMin = 'Final_Yellow_GMM_BGR_3.npz' # Read in GMM parameters
		GMMin = 'Final_GoodData_GMM_BGR_3.npz'

		thresh = .16#1e-4#1e-5
		npzfile = np.load(GMMin)
		k = npzfile['arr_0']
		mean = npzfile['arr_1']
		cov = npzfile['arr_2']
		pi = npzfile['arr_3']
		GMMargs = (thresh,k,mean,cov,pi)

		# Initialize Single Gauss Model
		filename = 'Final_GoodData.npy'
		#filename = 'MaroonWindow_2.npy'
		thresh_single  = .16#1e-19#1e-4 # threshold for Maroon
		BGR_set = np.load(filename)[:3]
		HSV_set = np.load(filename)[3:]
		#cov_single = np.cov(HSV_set)
		#icov_single = np.linalg.inv(cov_single)
		#mu_single = np.mean(HSV_set,axis = 1)
		cov_single = np.cov(BGR_set)
		icov_single = np.linalg.inv(cov_single)
		mu_single = np.mean(BGR_set,axis = 1)
		
		single_Gauss = (thresh_single,mu_single,cov_single,icov_single)

		# Maunal HSV Limits
		variance = 20
		Hlims = [mu_single[0]-variance,mu_single[0]+variance]
		Slims = [mu_single[1]-variance,mu_single[1]+variance]
		Vlims = [mu_single[2]-variance,mu_single[2]+variance]
		manual_args = (Hlims,Slims,Vlims)

		self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=5)
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/image_raw",Image,self.callback,(GMMargs,single_Gauss,manual_args))

	def callback(self,data,args):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)

		# GMM arguments
		temp = args[0]
		thresh = temp[0]
		k= temp[1]
		mean = temp[2]
		cov= temp[3]
		pi = temp[4]

		# Single Gauss args
		temp = args[1]
		thresh_single = temp[0]
		mu_single = temp[1]
		cov_single = temp[2]
		icov_single = temp[3]

		# Manual args
		temp = args[2]
		Hlims = temp[0]
		Slims = temp[1]
		Vlims = temp[1]

		# Color Correct
		cv_image_cc = cv_image#cc.color_correct(cv_image)
		cv_image_cc_blur = cv_image_cc#cv2.medianBlur(cv_image_cc,5)

		gray = cv2.cvtColor(cv_image_cc_blur,cv2.COLOR_RGB2GRAY) 
		mask = (gray>150).astype('uint8')*255
		mask2 = (gray<50).astype('uint8')*255
		cv_image_cc_blur_sat = cv2.bitwise_not(cv_image_cc_blur,cv_image_cc_blur,mask = mask)
		#cv_image_cc_blur_sat = cv2.bitwise_not(cv_image_cc_blur_sat1,cv_image_cc_blur_sat1,mask = mask2)
		#cv_image_cc = cv_image
		# SCALE DOWN IMAGE SIZE to 20%
		scale = .2
		width = int(cv_image_cc.shape[1] * scale)
		height = int(cv_image_cc.shape[0] * scale)
		dim = (width, height)
		# resize image
		cv_image_small = cv2.resize(cv_image_cc_blur_sat, dim, interpolation = cv2.INTER_AREA)
		print('reduced Image')
		# Convert image to HSV
		#cv_image_small_HSV = cv2.cvtColor(cv_image_small, cv2.COLOR_BGR2HSV)
		# Run GMM
		img_thresh_GMM = GMM(cv_image_small,thresh,k,mean,cov,pi)
		# Run Single Gauss
		img_thresh_Single = SingleGauss(cv_image_small,thresh_single,mu_single,cov_single,icov_single)
		#img_thresh_Single = SingleGauss(cv_image_small,thresh_single,mu_single,cov_single,icov_single)
		# Run Manual Threshold
		img_thresh_manual = ManualThresh(cv_image_small,Hlims,Slims,Vlims)

		cv2.imshow("Raw Image", cv_image_small)
		cv2.imshow("GMM Thresholded", img_thresh_GMM)
		cv2.moveWindow("GMM Thresholded",0,200)
		cv2.imshow("Single Gauss Thresholded", img_thresh_Single)
		cv2.moveWindow("Single Gauss Thresholded",000,350)
		cv2.imshow("Manual Thresholded", img_thresh_manual)
		cv2.moveWindow("Manual Thresholded",0,550)
		cv2.waitKey(1)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image_cc, "bgr8"))
		except CvBridgeError as e:
			print(e)

def GMM(img,thresh,k,mean,cov,pi):
	img_thresh = np.zeros(img.shape) # initialize black image
	print('Running GMM')
	icov = np.linalg.inv(cov)
	#p0 = 1/(((2*3.14159)**3*np.linalg.det(cov)))**(0.5)
	#pmax = np.zeros(k)
	#for j in range(0,k):
	#	pmax[j] =  p0[j]*np.exp(-.5*np.linalg.multi_dot([[0,0,0],icov[j],[0,0,0]]))

	# Loop through pixels and compare each value to the threshold.
	x=0
	y=0
	
	po =  np.zeros(img.shape[:2])
	pmax = np.sum(pi[:])
	for col in img:
		for pixel in col:
			p = np.zeros(k)
			for j in range(0,k):
				if (pixel == np.zeros(3)).all():
					p[j] = 0
				else:
					temp = pixel-mean[j]
					#p[j] = pi[j]*p0[j]*np.exp(-.5*np.linalg.multi_dot([temp,icov[j],temp]))/pmax[j]
					p[j] = pi[j]*np.exp(-.5*np.linalg.multi_dot([temp,icov[j],temp]))
			po[y,x] = np.sum(p)
			#print(post)
			#if post <= thresh:
				# set the pixel to neon pink to stand out
				#img_thresh[y,x] = [199,110,255]
				# use this for HSL instead
				#img_thresh[y,x] = [0, 255, 255]
			x = x+1
		x = 0
		y = y+1
	#plt.ion() # This prevents the program from hanging at the end
	#plt.subplot(2,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	#plt.subplot(2,1,2),plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
	#plt.show()
	pnorm = po/pmax#(po/np.max(po))
	#pnorm = (po/np.mean(po))
	#pnorm = (po/np.median(po))
	mask = (pnorm>=thresh).astype('uint8')*255
	#res = cv2.bitwise_and(img_thresh,img_thresh,mask = mask)
	return mask

def SingleGauss(img,thresh,mean,cov,icov):
	img_thresh = np.zeros(img.shape) # initialize black image
	print('Running Single Gauss')	
	#p0 = 1/(((2*3.14159)**3*np.linalg.det(cov)))**(0.5)
	#pmax =  p0*np.exp(.5*np.linalg.multi_dot([[0,0,0],icov,[0,0,0]]))


	# Loop through pixels and compare each value to the threshold.
	x=0
	y=0
	p =  np.zeros(img.shape[:2])
	for col in img:
		for pixel in col:
			if (pixel == np.zeros(3)).all():
				p[y,x] = 0
			else:
				temp = pixel-mean
				#p[y,x] = p0*np.exp(-.5*np.linalg.multi_dot([temp,icov,temp]))/pmax
				p[y,x] = np.exp(-.5*np.linalg.multi_dot([temp,icov,temp]))
			#post = p
			#if post <= thresh:
			#	img_thresh[y,x] = [199,110,255]
			x = x+1
		x = 0
		y = y+1
	pnorm = p#(p/np.max(p))
	mask = (pnorm>=thresh).astype('uint8')*255
	#res = cv2.bitwise_and(img_thresh,img_thresh,mask = mask)
	return mask

def ManualThresh(img,Hlims,Slims,Vlims):
	img_thresh = np.zeros(img.shape) # initialize black image
	# Loop through pixels and compare each value to the threshold.
	x=0
	y=0
	for col in img:
		for pixel in col:
			if Hlims[0] <= pixel[0] <= Hlims[1]:
				if Slims[0] <= pixel[1] <= Slims[1]:
					if Vlims[0] <= pixel[2] <= Vlims[1]:
						
						img_thresh[y,x] = [255,255,255]
			x = x+1
		x = 0
		y = y+1


	return img_thresh


def main(args):
	ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)


