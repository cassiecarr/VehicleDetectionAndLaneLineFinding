# Vehicle Detection and Find Lane Lines

# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math 
import moviepy
from moviepy.editor import VideoFileClip
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import pickle
from functools import partial

# Import Vehicle Detection Functions
from VehicleDetectionUtils import *

# # **UNCOMMENT WHEN CALIBRATING CAMERA**
# # Calibrate the camera for finding lane lines functions
# from CalibrateCamera import *
# images = glob.glob('test_images/camera_cal/calibration*.jpg')
# ret, mtx, dist, rvecs, tvecs = get_calibration(images)

# # Save the camera calibration
# pickle.dump([ret, mtx, dist, rvecs, tvecs], open( "camera_cal.p", "wb" ))
# # ***

# Load the camera calibration
ret, mtx, dist, rvecs, tvecs = pickle.load(open( "camera_cal.p", "rb" ))


# Define the parameters for extract_features to determine spatial, histogram, and hog features
color_space = 'YCrCb' # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = False # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 620] # Min and max in y to search in slide_window()

# # **UNCOMMENT WHEN TRAINING DATASET**
# # Read in cars and notcars
# cars = []
# noncars = []
# images = glob.glob('test_images/non-vehicles/**/*.png')
# for image in images:
#     noncars.append(image)
#     noncars.append(image)

# images = glob.glob('test_images/vehicles/**/*.png')
# for image in images:
#     cars.append(image)
#     cars.append(image)

# # Even the dataset between cars and not cars
# noncars = shuffle(noncars)
# cars = shuffle(cars)
# cars_difference = abs(len(noncars) - len(cars))
# max_number_loaded = max(len(noncars), len(cars))

# if len(cars) > len(noncars):
# 	noncars = noncars + noncars[0:cars_difference]
# else:
# 	cars = cars + cars[0:cars_difference]

# # Extract features
# car_features = extract_features(cars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)
# notcar_features = extract_features(noncars, color_space=color_space, 
#                         spatial_size=spatial_size, hist_bins=hist_bins, 
#                         orient=orient, pix_per_cell=pix_per_cell, 
#                         cell_per_block=cell_per_block, 
#                         hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                         hist_feat=hist_feat, hog_feat=hog_feat)

# # Shuffle the data
# car_features = shuffle(car_features)
# notcar_features = shuffle(notcar_features)

# # Define the features vector
# X = np.vstack((car_features, notcar_features)).astype(np.float64)  

# # Fit a per-column scaler
# X_scaler = StandardScaler().fit(X)
# # Apply the scaler to X
# scaled_X = X_scaler.transform(X)

# # Define the labels vector
# y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# # Split up data into randomized training and test sets
# rand_state = np.random.randint(0, 100)
# X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

# # Use a linear SVC, train, and print accuracy 
# svc = LinearSVC(C = 0.5)
# svc.fit(X_train, y_train)

# print('Train Accuracy of SVC = ', round(svc.score(X_train, y_train), 4))
# print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# # Save the trained model
# pickle.dump(svc, open( "svc.p", "wb" ))
# pickle.dump(X_scaler, open( "scaler.p", "wb" ))
# # ***

# Load the trained model
svc = pickle.load(open( "svc.p", "rb" ))
X_scaler = pickle.load(open( "scaler.p", "rb" ))

# Import Threshold and Find Lane Line functions
import Threshold
import FindLaneLineUtils

# Import class for storing previous frame results
from ObjectTracker import *
results = Object()

# Function to return color processed image with vehicle detection boxes and lane lines overlaid
def process_image(mtx, dist, image):

	# Define an image to draw on
	draw_image = np.copy(image)

	# **LANE LINE DETECTION**

	# Undistort the image
	dist = cv2.undistort(image, mtx, dist, None, mtx)

	# Apply thresholding to the image
	combined_binary = Threshold.white_yellow_threshold(dist)

	# Mask lane line area
	masked_img = FindLaneLineUtils.mask_lane_line_area(combined_binary)

	# Warp the image to an overhead view
	warped_img, src, dest = FindLaneLineUtils.warp_image(masked_img)

	# Identify the lane line pixels
	combined = FindLaneLineUtils.get_lane_line_pixels(warped_img)

	# Get polynomials for lane lines
	left_fit, right_fit = FindLaneLineUtils.get_polynomials(combined)

	# **VEHICLE DETECTION**

	# Define xy overlap and use slide_windows function to develop search windows for vehicles
	xy_overlap = (0.8, 0.8)
	windows_1 = slide_window(image, x_start_stop=[None, None], y_start_stop=[440, 620], 
                    		xy_window=(128, 128), xy_overlap=xy_overlap)
	windows_2 = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 496], 
                    		xy_window=(96, 96), xy_overlap=xy_overlap)
	windows_3 = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 528], 
                    		xy_window=(64, 64), xy_overlap=xy_overlap)
	windows = windows_1 + windows_2 + windows_3

	# Alter images read since images are jpg, where training data is png
	sized_image = image.astype(np.float32)/255

	# Use hot_windows function to "turn on" windows are assigned as vehicle by the classifier
	hot_windows = search_windows(sized_image, windows, svc, X_scaler, color_space=color_space, 
	                        spatial_size=spatial_size, hist_bins=hist_bins, 
	                        orient=orient, pix_per_cell=pix_per_cell, 
	                        cell_per_block=cell_per_block, 
	                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
	                        hist_feat=hist_feat, hog_feat=hog_feat)                      
	
	# Intialize all_hot_windows with this frame's hot windows
	all_hot_windows = hot_windows
	
	# Define window image, draw_boxes draws all_hot_windows onto the draw image
	window_img = draw_boxes(draw_image, all_hot_windows, color=(0, 0, 255), thick=6)     

	# Add heat to each box in box list
	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	heat = add_heat(heat, all_hot_windows)

	# Apply a threshold to the heatmap
	threshold = (np.array(all_hot_windows).shape[0]) // 10 + 1
	# threshold = 1

	heat = apply_threshold(heat, threshold)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)
	# mpimg.imsave((os.path.join("output_images/heat_images/", filename)), heatmap)               

	# Find final boxes from heatmap using label function
	# save the boxes drawn as an array
	labels = label(heat)
	output_boxes = get_vehicle_boxes(np.copy(image), labels)

    # **MONITOR OVER LAST 10 FRAMES**

	# Update results
	results.update(output_boxes, left_fit, right_fit)

	# Get the average over the last 10 frames
	avg_boxes = results.get_boxes()
	avg_left_fit = results.get_poly_left()
	avg_right_fit = results.get_poly_right()

	# **COMBINED DETECTION & DRAW**

	# Draw the lane area on a blank image
	poly_img = FindLaneLineUtils.plot_polynomial_overlay(combined, avg_left_fit, avg_right_fit)

	# Un-warp the image back to it's normal view with the lane line area overlain
	lane_line_overlay_img = FindLaneLineUtils.overlay_with_inverse_transform(draw_image, poly_img, src, dest)

	# Add the vehicle detection boxes to to the lane line overlay image
	box_image = draw_labeled_bboxes(np.copy(image), avg_boxes)
	result = cv2.addWeighted(box_image, 0.5, lane_line_overlay_img, 0.5, 0)

	return result

# Bind the process image and calibration data
bound_process_image = partial(process_image, mtx, dist)

# # **UNCOMMENT WHEN TESTING ON OUTPUT IMAGES**
# # Process test images with process image function
# for filename in os.listdir("test_images/"):
#     if filename.endswith(".jpg"): 
#         # Identify the image
#         image = mpimg.imread(os.path.join("test_images/", filename))
#         output = bound_process_image(image)

#         # Save the file as overlay
#         mpimg.imsave((os.path.join("output_images/", filename)),output)
# # ***

# Process video with process image function
output = 'output.mp4'
clip = VideoFileClip("project_video.mp4")
sub_clip = clip.subclip(18, 21)
output_clip = clip.fl_image(bound_process_image) 
output_clip.write_videofile(output, audio=False)

