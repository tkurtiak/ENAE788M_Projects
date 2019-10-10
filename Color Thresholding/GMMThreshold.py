# GMM Threshold


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
scale = .05
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dim = (width, height)
# resize image
img_thresh = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
GMMin = 'Yellow_GMM_4.npz'

# load GMM here
npzfile = np.load(GMMin)
k = npzfile['arr_0']
mean = npzfile['arr_1']
cov = npzfile['arr_2']
pi = npzfile['arr_3']


#k = 4
#mean = [[  36.66281864,   64.81118029,  103.27519976],
#       [ 245.93997884,  248.94475704,  249.13393796],
#       [ 126.49990718,  130.40771154,  132.83349729],
#       [  73.55020833,  110.64312562,  151.12236608]]
#cov = [[[   85.38992059,   147.91432211,   186.16698995],
#        [  147.91432211,   317.39895903,   424.28360083],
#        [  186.16698995,   424.28360083,   616.27459291]],
#
#       [[   31.87338667,  -176.39579349,  -223.73078396],
#        [ -176.39579349,  2666.48228886,  3330.99506833],
#        [ -223.73078396,  3330.99506833,  4165.56331643]],
#
#       [[ 6933.15016094,  6436.88518114,  6457.23927872],
#        [ 6436.88518114,  6855.80134145,  7019.91654485],
#        [ 6457.23927872,  7019.91654485,  7249.51946886]],
#
#       [[ 1491.03755167,  1107.63133422,   860.85064193],
#        [ 1107.63133422,   842.16102434,   650.80438933],
#        [  860.85064193,   650.80438933,   515.09929906]]]
#pi = [  0.33606731,   0.20328249,  16.27144447,   0.18920573]

icov = np.linalg.inv(cov)
p0 = 1/(((2*3.14159)**3*np.linalg.det(cov)))**(0.5)
pmax = np.zeros(k)
for j in range(0,k):
	pmax[j] =  p0[j]*np.exp(.5*np.linalg.multi_dot([[0,0,0],icov[j],[0,0,0]]))



# Loop through pixels and compare each value to the threshold.
x=0
y=0
p = np.zeros(k)

for col in img_thresh:
	for pixel in col:
		for j in range(0,k):
			temp = pixel-mean[j]
			p[j] = pi[j]*p0[j]*np.exp(-.5*np.linalg.multi_dot([temp,icov[j],temp]))/pmax[j]
		post = np.sum(p)
		#print(post)
		if post <= 0.000001:
			# set the pixel to neon pink to stand out
			img_thresh[y,x] = [199,110,255]
		x = x+1
	x = 0
	y = y+1

plt.ion() # This prevents the program from hanging at the end
plt.subplot(2,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(2,1,2),plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
plt.show()

