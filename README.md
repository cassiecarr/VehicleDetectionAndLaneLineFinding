# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
This project develops a software pipeline to detect vehicles in a video.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Project Files
---
* [VehicleDetection.py](VehicleDetection.py) contains the main pipeline for training the dataset and processing the video to overlay the found vehicle windows.
* [VehicleDetectionUtils.py](VehicleDetectionUtils.py) contains the functions used to extract features from images to train and test the dataset, in addition, functions to determine the windows and search windows for vehicles.
* [writeup.md](writeup.md) explains the results.


Results
---
Here's a [link to my video result](https://youtu.be/tqVbcKeqQzQ)


Additional Links
---
Here are links to the labeled data used in this project for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train the classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.  Samples from the [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) were also used to augment the training data.  
