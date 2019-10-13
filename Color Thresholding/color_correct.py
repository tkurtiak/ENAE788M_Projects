import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt

def color_correct(img):
	# read an image
	#img = cv2.imread('test.jpg')
	# detemine its size
	width, height = img.shape[:2]

	# convert image to RGB color for matplotlib
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# plt.imshow(img)
	# plt.show()
	# show image with matplotlib
	#img_plot = plt.imshow(img)

	# convert image to grayscale
	#gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	#plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
	#plt.show()

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
	dst = cv2.addWeighted(img,0.5,average_color_img,0.5,0)

	# Increse contrast of the resultant image
	contrast = 64
	f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
	alpha_c = f
	gamma_c = 127*(1-f)
	dst = cv2.addWeighted(dst, alpha_c, dst, 0, gamma_c)

	#plt.imshow(dst)
	#plt.show()
	dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
	#dst = cv2.cvtColor(dst, cv2.COLOR_HSV2BGR)
	return dst
