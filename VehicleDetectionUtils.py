import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    histogram_image = np.copy(img)
    channel1_hist = np.histogram(histogram_image[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(histogram_image[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(histogram_image[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image_original = mpimg.imread(file)
        # Add a vertically flipped image to the trianing set
        image_flipped = np.fliplr(image_original)
        # Only include 3 color channels
        images = [image_original[:,:,:3], image_flipped[:,:,:3]]

        for image in images:
        # Apply color conversion if other than 'RGB'
          file_features = []
          if color_space != 'RGB':
              if color_space == 'HSV':
                  feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
              elif color_space == 'LUV':
                  feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
              elif color_space == 'HLS':
                  feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
              elif color_space == 'YUV':
                  feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
              elif color_space == 'YCrCb':
                  feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
          else: feature_image = np.copy(image)      

          if spatial_feat == True:
            # Apply spacial features
              spatial_features = bin_spatial(feature_image[:,:,2], size=spatial_size)
              file_features.append(spatial_features)
          if hist_feat == True:
              # Apply color_hist()
              hist_features = color_hist(feature_image, nbins=hist_bins)
              file_features.append(hist_features)
          if hog_feat == True:
              # Call get_hog_features() with vis=False, feature_vec=True
              if hog_channel == 'ALL':
                  hog_features = []
                  for channel in range(feature_image.shape[2]):
                      hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                          orient, pix_per_cell, cell_per_block, 
                                          vis=False, feature_vec=True))
                  hog_features = np.ravel(hog_features)        
              else:
                  hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                              pix_per_cell, cell_per_block, vis=False, feature_vec=True)
              # Append the new feature vector to the features list
              file_features.append(hog_features)
          features.append(np.concatenate(file_features))
      # Return list of feature vectors
    return features
    
# Function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y) and returns a window list 
# for those parameters
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Function to extract features from a single image
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    # Define an empty list to receive features
    img_features = []
    image = np.copy(img)
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)      
    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image[:,:,2], size=spatial_size)
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    # Return concatenated array of features
    return np.concatenate(img_features)

# Function that takes an image and the list of windows to be searched and 
# returns the windows that vehicles were found 
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []
    # Iterate over all windows in the list
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        # Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1).astype(np.float64))
        # Predict using classifier
        prediction = clf.predict(test_features)
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # Return windows for positive detections
    return on_windows
    

# Function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Function to create heatmap of overlapping boxes
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

# Function to apply threshold to heatmap based on set theshold
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

# Get box array, combines boxes that are close together into one box
def get_vehicle_boxes(img, labels):
    # Define empty array for boxes of each label
    bboxes = []
    # Define an empty array for all box coordinates that are drawn 
    output_boxes = []
    # Loop though the labels
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)

    # If there is more than one box, search around boxes and combine them if they are closer
    # than the buffer zone 
    if len(bboxes) > 1:
      bbox_number = 0
      while bbox_number < (len(bboxes) - 1):
          box1_x1 = bboxes[bbox_number][0][0]
          box2_x1 = bboxes[bbox_number+1][0][0]
          box1_x2 = bboxes[bbox_number][1][0]
          box2_x2 = bboxes[bbox_number+1][1][0]
          box1_y1 = bboxes[bbox_number][0][1]
          box2_y1 = bboxes[bbox_number+1][0][1]
          box1_y2 = bboxes[bbox_number][1][1]
          box2_y2 = bboxes[bbox_number+1][1][1]
          left_distance = abs(box1_x1 - box2_x1)
          right_distance = abs(box1_x2 - box2_x2)
          buffer_zone = 100
          if (left_distance < buffer_zone) or (right_distance < buffer_zone):
              bbox = ((min(box1_x1, box2_x1), min(box1_y1, box2_y1)), (max(box1_x2, box2_x2), max(box1_y2, box2_y2)))
              bbox_number += 1
          else:
              bbox = bboxes[bbox_number]
          # Draw the box on the image
          if (abs(bbox[0][0] - bbox[1][0]) > 50) and (abs(bbox[0][1] - bbox[1][1]) > 50):
              output_boxes.append(bbox)
              if(bbox_number == (len(bboxes) - 2)):
                output_boxes.append(bboxes[bbox_number+1])
          bbox_number += 1
    # If there is only one box, draw the box
    elif len(bboxes) == 1:
      if (abs(bboxes[0][0][0] - bboxes[0][1][0]) > 50) and (abs(bboxes[0][0][1] - bboxes[0][1][1]) > 50):
          output_boxes.append(bboxes[0])
    return output_boxes


# Function to draw boxes on image
def draw_labeled_bboxes(img, bboxes):
    box_image = np.copy(img)
    for bbox in bboxes:
      cv2.rectangle(box_image, tuple(bbox[0]), tuple(bbox[1]), (0,0,255), 6)
    return box_image
