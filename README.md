# Vehicle and Lane Line Detection in Video
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./output_images/test5.jpg

The Project
---
This project was developed from two projects completed during Term 1 of the [Udacity Self-Driving Car Nanodegree](http://www.udacity.com/drive). It develops a software pipeline to detect vehicles and lane lines in a video.

The goals / steps of this project are the following:

For lane line detection:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms to create a thresholded binary image
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries as a video stream

For vehicle detection:
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected

Here is an example of the pipeline output:
![alt text][image1]

Project Files
---
* [VehicleDetectionLaneLines.py](VehicleDetectionLaneLines.py) contains the main pipeline for training the dataset and processing the video to overlay the found vehicle windows and lane boundries.
* [VehicleDetectionUtils.py](VehicleDetectionUtils.py) contains the functions used to extract features from images to train and test the dataset, in addition, functions to determine the windows and search windows for vehicles.
* [FindLaneLineUtils.py](FindLaneLineUtils.py) contains the functions used to warp the image to an overhead view, find lane line pixels, and determine polynomials that represent the lane line boundries.
* [Threshold.py](Threshold.py) contains function for thresholding the image for lane line detection.
* [CalibrateCamera.py](CalibrateCamera.py) contains the function for calibrating the camera.
* [ObjectTracker.py](ObjectTracker.py) contains class used to track and average the vehicle detection boxes and lane lines over time to more accurately display the results. 

Results
---
Here's a [link to my video result](https://youtu.be/s4M1O6Nw_wI)


Additional Links
---
Here are links to the labeled data used in this project for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.  Samples from the [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) were also used to augment the training data. 

The images used in the Camera Calibration can also be found [here](https://github.com/cassiecarr/CarND-AdvancedLaneLines-P4-1/tree/master/camera_cal). 
