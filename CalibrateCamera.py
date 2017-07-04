# Calibrate the camera

# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

# Define function to return camera calibration
def get_calibration(images):
	# Define object points and image points array
	objpoints = []
	imgpoints = []

	# Prepare object points array
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Loop through example images and add to imgpoints and objpoints arrays 
	gray = []
	for fname in images:
		img = mpimg.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)

	# Use open CV calibrate camera function to get camera calibration
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return ret, mtx, dist, rvecs, tvecs
