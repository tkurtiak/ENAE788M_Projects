# ENAE788M_Projects
Master Repo for ENAE788M Project files


Color Thresholding Folder:
ROSbags can be found on google drive: https://drive.google.com/drive/folders/1dBvmxCwH1vNJoI5euONhH42natqm0I_8?usp=sharing

ROS_Threshold.py runs GMM, sIngle Gauss, and Manual Thresholding by subscribing to ROS topic for images.  This contains the most updated and accurate GMM and Single Gauss.  GMM model is stored under a ".npz" file called "Yellow_GMM_HSV_5.npz".  RGB,HSV sample data collected from sample images is stored in "YellowWindow.npy"

Train_Thresholds.py is a script used to select Regions of OInterest on training photos.  It adds selected pixels to the array of sample pixels which are used in generating the Single Gaussian and GMM.  If plotting is enabled, this script will also plot all of the sample points in rGB space and HSV space.  

GMM.py creates a GMM with a given dimention k.  This may take a while to converge.  Please change the value in the script, k, and the save file name or else you will overwrite a previous GMM.  Note that increasing GMM dimentions, (larger k) will result in longer compute time both in optimizing the GMM and running it later.  
