import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal
import color_correct as cc
# Image Select
import Tkinter, tkFileDialog

root = Tkinter.Tk()
root.withdraw()
imgname = tkFileDialog.askopenfilename()
# Selected training img
#imgname = 'Images/InitislSet.png'

# Save/Load File name for RGB values
#filename = 'MaroonWindow.npy'
filename = 'Final_GoodData.npy'
#filename = 'test.npy'

# load image
img = cv2.imread(imgname)

# SCALE DOWN IMAGE SIZE
scale = .5
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dim = (width, height)
# resize image
img_thresh = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#img_thresh = cc.color_correct(img_thresh)

#img_thresh = cv2.medianBlur(img_thresh,5)

gray = cv2.cvtColor(img_thresh,cv2.COLOR_RGB2GRAY) 
mask = (gray>200).astype('uint8')*255
img_thresh = cv2.bitwise_not(img_thresh,img_thresh,mask = mask)


# load current training dataset
BGR_set = np.load(filename)[0:3]
HSV_set = np.load(filename)[3:]

# Calculate single gausian
# USE BGR COLORS
covariance = np.cov(BGR_set)
icovariance = np.linalg.inv(covariance)
mu = np.mean(BGR_set,axis = 1)

### USE HSV COLORS
# covariance = np.cov(HSV_set)
# icovariance = np.linalg.inv(covariance)
# mu = np.mean(HSV_set,axis = 1)
thresh  = .18#1e-19#35 # threshold for Yellow in HSV is 1e-62
# Convert image to HSV
#img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2HSV)
#
# resize or subsample image to improve run speed

#prior = .5 #2 options, either it is the color or it is not the color


#p = 1/((2*3.14159)^3*np.norm(covariance))*np.exp(.5*(X-mean).T*icovariance*(X-mean))
#p0 = 1/(((2*3.14159)**3*np.linalg.det(covariance)))**(0.5)
# What is the highest probability if the exact mean is used, ie 000, no error
#pmax =  p0*np.exp(-.5*np.linalg.multi_dot([[0,0,0],icovariance,[0,0,0]]))
#pmax =  multivariate_normal.pdf(mu, mean=mu, cov=covariance)


# Loop through pixels and compare each value to the threshold.
x=0
y=0
p = np.zeros(img_thresh.shape[:2])
for col in img_thresh:
	for pixel in col:
		if (pixel == np.zeros(3)).all():
			p[y,x] = 0
		else:
			
			p[y,x] = np.exp(-.5*np.linalg.multi_dot([pixel-mu,icovariance,pixel-mu]))
		#temp = pixel-mu
		#p = p0*np.exp(-.5*np.linalg.multi_dot([temp,icovariance,temp]))/pmax
		#p[y,x] = np.exp(-.5*np.linalg.multi_dot([-pixel+mu,icovariance,-pixel+mu]))
		#p[y,x] = np.exp(-.5*np.linalg.multi_dot([pixel-mu,icovariance,pixel-mu]))
		
		#z = multivariate_normal.pdf(pixel, mean=mu, cov=covariance)/pmax
		#print(p)
		#print(z)
		#post = np.linalg.norm((p*prior)/2)
		#post = p
		#print(post)
		#if p[y,x]<thresh: #<= thresh:
			# set the pixel to neon pink to stand out
			#img_thresh[y,x] = [199,110,255]
			# use this for HSL instead
		#	img_thresh[y,x] = [0, 255, 255]
		x = x+1
	x = 0
	y = y+1
#pnorm = (p/np.max(p))
pnorm = p#(p/np.mean(p))
mask = (pnorm>=thresh).astype('uint8')*255
res = cv2.bitwise_and(img_thresh,img_thresh,mask = mask)

plt.ion() # This prevents the program from hanging at the end
plt.subplot(3,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(3,1,2),plt.imshow(mask)
#plt.subplot(3,1,2),plt.imshow(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))
plt.subplot(3,1,3),plt.imshow(np.log10(pnorm))
plt.show()

#cv2.imshow(res)



# What do the probability of actual points look like?
#p_set = np.zeros(HSV_set.shape[1])
#for i in range(HSV_set.shape[1]):
#	e = HSV_set[:,i]-mu
#	p_set[i] = np.exp(-.5*np.linalg.multi_dot([e,icovariance,e]))


