# Import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

def mask_lane_line_area(img):
    # Create masked edge image using cv2.fillPoly()
    mask = np.zeros_like(img)
    ignore_mask_color = 255
    # Define four sided polygon to mask
    imshape = img.shape
    left_bottom = (np.int(imshape[1]*0.05),imshape[0])
    left_top = (np.int(0.4*imshape[1]), np.int(0.65*imshape[0]))
    right_top = (np.int(0.6*imshape[1]), np.int(0.65*imshape[0]))
    right_bottom = (np.int(imshape[1]-imshape[1]*0.05),imshape[0])
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    # Mask the image with defined polygon
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def warp_image(img):
    # Define the 4 corners of a polygon surrounding the lane lines
    contour_image = img.copy()
    lower_left = [190, 720]
    lower_right = [1100, 720]
    upper_left = [575, 468]
    upper_right = [715, 468]
    cv2.line(contour_image,(lower_left[0],lower_left[1]), (upper_left[0],upper_left[1]),(0,255,0),3)
    cv2.line(contour_image,(lower_right[0],lower_right[1]), (upper_right[0],upper_right[1]),(0,255,0),3)

    # Use perspective transform to define a top down view of the image
    src = [lower_left, lower_right, upper_left, upper_right]
    dest = [[320, 720], [960, 720], [320, 0], [960, 0]]
    src = np.array(src, np.float32)
    dest = np.array(dest, np.float32)
    M = cv2.getPerspectiveTransform(src, dest)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, src, dest


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(warped.shape[0]/2):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(warped.shape[0]/2):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def get_lane_line_pixels(img):
    # Find lane lines using window search
    # Set window settings and find centroids
    window_width = 50 
    window_height = 120 # Break image into 9 vertical layers since image height is 720
    margin = 80 # Define how much to slide left and right for searching
    window_centroids = find_window_centroids(img, window_width, window_height, margin)
    # If any window centers found
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Use window_mask to draw window areas
            l_mask = window_mask(window_width,window_height,img,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,img,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results 
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        img[img == 1] = 255
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Identify only the pixels that are contained within the windows found
    warped_binary = np.zeros_like(img)
    warped_binary[(img > 0)] = 255
    template_binary = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    template_binary[(template_binary > 0)] = 255
    combined = np.zeros_like(template_binary)
    combined[(warped_binary == 255) & (template_binary == 255)] = 255

    return combined

def get_polynomials(img):
    # Identify the indices of the left and right pixels
    img_size = (img.shape[1], img.shape[0])
    half = np.int(img_size[0]/2)
    nonzero = np.argwhere(img > 1)
    nonzero_left = nonzero[(nonzero[:, 1] < half)]
    nonzero_right = nonzero[(nonzero[:, 1] > half)]
    x_left = nonzero_left[:, 1]
    y_left = nonzero_left[:, 0]
    x_right = nonzero_right[:, 1]
    y_right = nonzero_right[:, 0]

    # Use polyfit to determine the line curve the identified left and right pixels
    left_fit = np.polyfit(y_left, x_left, 2)
    right_fit = np.polyfit(y_right, x_right, 2)

    return left_fit, right_fit

def plot_polynomial_overlay(img, left_fit, right_fit):
    # Identify right and left points for plot
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define an image to draw the lane line polynomials on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Draw the lane line polynomials onto blank image
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp

def overlay_with_inverse_transform(img, poly_img, src, dest):
    # Warp the blank image with lane lines back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dest, src)
    newwarp = cv2.warpPerspective(poly_img, Minv, (poly_img.shape[1], poly_img.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.5, 0)

    return result




