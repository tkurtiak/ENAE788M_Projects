import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

dirpath = os.getcwd()
print(dirpath)	
# read an image
img = cv2.imread('rgb_nemo.jpg')
print(img)
# detemine its size
width, height = img.shape[:2]
# convert image to RGB color for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
# show image with matplotlib
img_plot = plt.imshow(img)
# plt.show()c

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# find average per row
# np.average() takes in an axis argument which finds the average across that axis. 
average_color_per_row = np.average(img, axis=0)

# find average across average per row
average_color = np.average(average_color_per_row, axis=0)

# convert back to uint8
average_color = np.uint8(average_color)

# take color compliment of the average value 
average_color[0] = 255-average_color[0]
average_color[1] = 255-average_color[1]
average_color[2] = 255-average_color[2]

# create height x width pixel array with average color value
average_color_img = np.array([[average_color]*height]*width, np.uint8)

# add the color compliment to the original image, each with 50% weights
dst = cv2.addWeighted(img, 0.5, average_color_img, 0.5, 0)

# Increse contrast of the resultant image #90 for bag1 64 for bag2
contrast = 90
f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
alpha_c = f
gamma_c = 127*(1-f)
dst = cv2.addWeighted(dst, alpha_c, dst, 0, gamma_c)

plt.imshow(dst)
# plt.show()

#### adaptive threshold ####
# img1 = cv2.medianBlur(cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY),5)

# ret,th1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img1, th1, th2, th3]

# for i in xrange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


img_rgb = img
img_lab = cv2.cvtColor(dst, cv2.COLOR_RGB2LAB)
img_hsv = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(img_hsv)
r, g, b = cv2.split(img_rgb)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
# print(np.shape(img_rgb))
pixel_colors = img.reshape((np.shape(img_rgb)[0]*np.shape(img_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
print(np.shape(pixel_colors))


axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()