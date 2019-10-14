import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import glob
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


dirpath = os.getcwd()

for subdir, dirs, files in os.walk(dirpath + '/Rosbag/bag2_images'):
    files.sort()
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".jpg") or filepath.endswith(".pgm") or filepath.endswith(".png") or filepath.endswith(".ppm"):
            fig = plt.figure()
            print(file)
            # read an image
            img = cv2.imread(filepath)
            # detemine its size
            width, height = img.shape[:2]
            # convert image to RGB color for matplotlib
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(211)
            plt.imshow(img)
            plt.title("Original_Image")
            # plt.show()
            # show image with matplotlib
            img_plot = plt.imshow(img)
            # plt.show()

            # convert image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            gray_img = cv2.equalizeHist(gray_img)
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
                contrast = 30
            # take color compliment of the average value 
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

            plt.subplot(212)
            plt.imshow(dst)
            plt.title("Corrected_Image")

            cv2.imwrite(str(dirpath) + "/Rosbag/bag2_corrected/" + str(file), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
            # plt.show()