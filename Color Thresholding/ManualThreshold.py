import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 


# Selected training img
imgname = 'Images/InitislSet.png'
# load image
img = cv2.imread(imgname)
img_thresh = cv2.imread(imgname)
# Initialize thresholds as desired values
#B_thresh = [30,60]
#G_thresh = [38,80]
#R_thresh = [40,75]

B_thresh = [40,80]
G_thresh = [15,70]
R_thresh = [35,70]


# Loop through pixels and compare each value to the threshold.
x=0
y=0

for col in img:
	
	for pixel in col:
		
		#print(pixel.shape)
		b = pixel[0]
		if B_thresh[0] <= b <= B_thresh[1]:
			g = pixel[1]
			if G_thresh[0] <= g <= G_thresh[1]:
				r = pixel[2]
				if R_thresh[0] <= r <= R_thresh[1]:
					# set the pixel to neon pink to stand out
					img_thresh[y,x] = [199,110,255]
		x = x+1
	x = 0
	y = y+1


plt.subplot(2,1,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(2,1,2),plt.imshow(cv2.cvtColor(img_thresh, cv2.COLOR_BGR2RGB))
plt.show()