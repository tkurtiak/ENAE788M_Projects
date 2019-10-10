import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 


## Selected training img
# imgname = 'Images/InitislSet.png'
# Image Select
import Tkinter, tkFileDialog

root = Tkinter.Tk()
root.withdraw()
imgname = tkFileDialog.askopenfilename()

# Save/Load File name for RGB values
filename = 'MaroonWindow.npy'
filename = 'YellowWindow.py'

# load image
img_raw = cv2.imread(imgname)
# SCALE DOWN IMAGE SIZE
scale = .1
width = int(img_raw.shape[1] * scale)
height = int(img_raw.shape[0] * scale)
dim = (width, height)
# resize image
img = cv2.resize(img_raw, dim, interpolation = cv2.INTER_AREA)


# load current training dataset
try:  # If it exists, load
	temp = np.load(filename)
	B = temp[0]
	G = temp[1]
	R = temp[2]
except:	# Else initialize to zero
	print('no starting data found, creating my own.  MUAHAHAHA')
	B =np.array([])
	G =np.array([])
	R =np.array([])

# Select rectangular regions of interest.  
# Be careful not to select bad data or this will make our life hell
samples = cv2.selectROIs('Window',img)
cv2.destroyAllWindows()



for sample in samples:
	# sample has a starting point, top left corner of box, and a width, and a height
	yS = slice(sample[0],sample[0]+sample[2],1)
	xS = slice(sample[1],sample[1]+sample[3],1)
	# extract sample space
	sampleImg = img[xS,yS,:]


	# Convert the RGB values to 1D arrays for processing

	B = np.append(B,np.ndarray.flatten(sampleImg[0],order='C'))
	G = np.append(G,np.ndarray.flatten(sampleImg[1],order='C'))
	R = np.append(R,np.ndarray.flatten(sampleImg[2],order='C'))


# Save newly added training points to dataset
np.save(filename,np.vstack((B,G,R)))
# Make a 3D ScaTTER Of RGB values
fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
i=0
for i in range(0,R.shape[0]-1):
	ax.scatter(R[i], G[i], B[i],color='k')


ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')
plt.ion() # This prevents the program from hanging at the end
plt.show()