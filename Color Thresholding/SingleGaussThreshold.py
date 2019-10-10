import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from scipy.stats import multivariate_normal

# Image Select
import Tkinter, tkFileDialog

root = Tkinter.Tk()
root.withdraw()
imgname = tkFileDialog.askopenfilename()
# Selected training img
#imgname = 'Images/InitislSet.png'

# Save/Load File name for RGB values
#filename = 'MaroonWindow.npy'
filename = 'YellowWindow.npy'

# load image
img = cv2.imread(imgname)

# SCALE DOWN IMAGE SIZE
scale = .1
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dim = (width, height)
# resize image
img_thresh = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# load current training dataset
BGR_set = np.load(filename)

# Calculate cov
covariance = np.cov(BGR_set)
icovariance = np.linalg.inv(covariance)
mu = np.mean(BGR_set,axis = 1)

# resize or subsample image to improve run speed

prior = .5 #2 options, either it is the color or it is not the color


#p = 1/((2*3.14159)^3*np.norm(covariance))*np.exp(.5*(X-mean).T*icovariance*(X-mean))
p0 = 1/(((2*3.14159)**3*np.linalg.det(covariance)))**(0.5)
# What is the highest probability if the exact mean is used, ie 000, no error
pmax =  p0*np.exp(.5*np.linalg.multi_dot([[0,0,0],icovariance,[0,0,0]]))
#pmax =  multivariate_normal.pdf(mu, mean=mu, cov=covariance)


# Loop through pixels and compare each value to the threshold.
x=0
y=0

for col in img_thresh:
	for pixel in col:
		temp = pixel-mu
		p = p0*np.exp(-.5*np.linalg.multi_dot([temp,icovariance,temp]))/pmax
		#z = multivariate_normal.pdf(pixel, mean=mu, cov=covariance)/pmax
		#print(p)
		#print(z)
		#post = np.linalg.norm((p*prior)/2)
		post = p
		#print(post)
		if post <= 0.15:
			# set the pixel to neon pink to stand out
			img_thresh[y,x] = [199,110,255]
		x = x+1
	x = 0
	y = y+1

plt.ion() # This prevents the program from hanging at the end
plt.subplot(2,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(2,1,2),plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
plt.show()

