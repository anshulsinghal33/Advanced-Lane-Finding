# Advanced Lane Finding
---

**The goals / steps of this project are the following:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: output_images/calibration_imgs_vis.png
[image2]: output_images/corrected_cal_img_vis.png
[image3]: output_images/test_imgs_vis.png
[image4]: output_images/undistorted_imgs/undistort_vis_test4.png
[image5]: output_images/sobel_filtered/sobel_x_thresh_vis_test3.png
[image6]: output_images/sobel_filtered/sobel_y_thresh_vis_test3.png
[image7]: output_images/sobel_filtered/sobel_mag_thresh_vis_test3.png
[image8]: output_images/sobel_filtered/sobel_dir_thresh_vis_straight_lines2.png
[image9]: output_images/sobel_filtered/sobel_x-y-mag-dir_thresh_vis_test3.png
[image10]: output_images/color_filtered/hls_vis.png
[image11]: output_images/color_filtered/hls_channel_vis.png
[image12]: output_images/color_filtered/s_thresh_vis.png
[image13]: output_images/color_filtered/lab_vis.png
[image14]: output_images/color_filtered/lab_channel_vis.png
[image15]: output_images/color_filtered/l_thresh_vis.png
[image16]: output_images/color_filtered/lab_vis.png
[image17]: output_images/color_filtered/lab_channel_vis.png
[image18]: output_images/color_filtered/r&g_thresh_vis.png
[image19]: output_images/color_filtered/s-l-r&g_thresh_vis.png
[image20]: output_images/color_sobel_combined_vis.png
[image21]: output_images/pre_processed_test_imgs_vis.png
[image22]: output_images/roi_test_imgs_vis.png
[image23]: output_images/roi_masked_imgs_vis.png
[image24]: output_images/warpped_imgs_vis.png
[image25]: output_images/sliding_window_search_vis.png
[image26]: output_images/histogram_vis.png
[image27]: output_images/neighbouring_search_vis.png
[image28]: output_images/sample_lane_detect_vis.png

---
## 1. Files Submitted

The required files can be referenced at :
* [Jupyter Notebook with Code](advanced_lane_finding.ipynb)
* [HTML output of the code](advanced_lane_finding.html)
* [Helpers.py](helpers.py)
* [Project Output Video](project_video_out.mp4)

---

## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


**The code for this step is contained in the code cell titled as under of the [IPython notebook](advanced_lane_finding).**


* **Loading and Visualizing Calibration Images**
![alt text][image1]


* **Functions to Calibrate Camera and Undistort Images**

    `def calibrate()` 
    `def undistort()`


* **Camera Calibration**

    The OpenCV functions `findChessboardCorners` and `calibrateCamera` are the backbone of the image calibration. A number of images of a chessboard, taken from different angles with the same camera, comprise the input. I started by preparing arrays of object points, which will be the (x, y, z) coordinates of the chessboard corners in the world, corresponding to the location (essentially indices) of internal corners of a chessboard, and image points. These pixel locations of the internal chessboard corners determined by `findChessboardCorners`. 

    Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

    I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 


* **Testing Calibration**

    These can then be used by the OpenCV `cv2.undistort()` function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). The image below depicts the results of applying `cv2.undistort()` to one of the chessboard images stored in the folder called [camera_cal](camera_cal):

![alt text][image2]


---
## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I applied the distortion correction to one of the test images like this one:
* [**Undistorting the Images**](output_images/undistorted_imgs)

![alt text][image4]

>_The effect of undistortion is subtle, but can be perceived from the difference in shape of the hood of the car at the bottom corners of the image and how the white car is pulled towards the side of the edge of the image as if it was flattened out._

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

**The aim is to find the lanes from the undistorted image. To do this I have used the following techniques:**

* [**Sobel Filtering**](output_images/sobel_filtered)
    * Sobel X
    * Sobel Y
    * Gradient Magnitude
    * Gradient Direction
        * **Combined Sobel**
    

* [**Colorspace Exploration**](output_images/color_filtered)
    * HLS Colorspace
    * LAB Colorspace
    * RGB Colorspace
        * **Combined Color**


* **Combined Sobel and Color**

### Sobel Filtering

**Calculating Directional Gradients & Thresholding**
```python
def abs_sobel_thresh():
    # Applying x or y gradient with the OpenCV Sobel() function
    # and taking the absolute value
    # Rescaling back to 8 bit integer
    # Creating a copy and apply the threshold
    return binary_output
```
[**Thresholding Sobel x**](output_images/sobel_filtered)
```python
# Thresholding Parameters
sobel_x_low = 30
sobel_x_high = 110
```
![alt text][image5]


[**Thresholding Sobel y**](output_images/sobel_filtered)
```python
# Thresholding Parameters
sobel_y_low = 30
sobel_y_high = 110
```
![alt text][image6]


**Calculating Gradient Magnitude and Thresholding**
```python
def mag_thresh():
    # Taking both Sobel x and y gradients
    # Calculating the gradient magnitude
    # Rescaling back to 8 bit integer
    # Creating a binary image of ones where threshold is met, zeros otherwise
    return binary_output
```

[**Thresholding Sobel x and y Magnitude**](output_images/sobel_filtered)
```python
# Thresholding Parameters
sobel_kernel = 15
sobel_mag_low = 30
sobel_mag_high = 110
```
![alt text][image7]


**Calculating Gradient Direction and Thresholding**
```python
def dir_threshold():
    # Calculating the x and y gradients
    # Taking the absolute values of the gradient direction, 
    # and applying a threshold 
    # Rescaling back to 8 bit integer
    # Creating a binary image of ones where threshold is met, zeros otherwise
    return binary_output
```

[**Thresholding Gradient Direction**](output_images/sobel_filtered)
```python
# Thresholding Parameters
sobel_kernel = 15
sobel_dir_low = 0.7
sobel_dir_high = 1.3
```
![alt text][image8]


**Combining the Gradient, Magnitude and Direction Thresholds**
![alt text][image9]


### Colorspace Exploration

**HLS Colorspace**
```python
hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
```
![alt text][image10]
![alt text][image11]

>### Inference: 
_**It can be observed that the `yellow lane lines` are most evident in the `S-channel` of the `HLS Colorspace` and hence will be the preferred channel for thresholding to obtain a Binary Image**_

[**Thresholding S-Channel**](output_images/color_filtered)
```python
# Thresholding Parameters
s_thresh = (175,255)
```
![alt text][image12]


**LAB Colorspace**
```python
lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
```
![alt text][image13]
![alt text][image14]

>### Inference: 
_**It can be observed that the `white lane lines` are most evident in the `L-channel` of the `LAB Colorspace` and hence will be the preferred channel for thresholding to obtain a Binary Image**_

[**Thresholding S-Channel**](output_images/color_filtered)
```python
# Thresholding Parameters
l_thresh = (210,255)
```
![alt text][image15]


**RGB Colorspace**
![alt text][image16]
![alt text][image17]

>### Inference: 
_**It can be observed that the `lane lines` are most evident in the `R & G - channels` of the `RGB Colorspace` and hence will be the preferred channel for thresholding to obtain a Binary Image. Also, this channel will specially help in lane detection where the S and L-channels could fail due to shadows.**_

[**Thresholding R & G - Channel**](output_images/color_filtered)
```python
# Thresholding Parameters
r_g_thresh = 200
r = rgb_img[:,:,0] #Selecting R-Channel
g = rgb_img[:,:,1] #Selecting G-Channel
```
![alt text][image18]


**Combining S-Channel, L-Channel and R & G - Channel Thresholds**

![alt text][image19]


### Combining Sobel and Color Thresholds
![alt text][image20]


### Image Pre-Processing Pipeline
```python
def pre_process(img:'Undistorted RGB Image'):
    # Sobel Filtering
    # Colorspace Filtering
    # Combining Sobel and Colorspace Binary Images
    return combined
```
![alt text][image21]


---
## 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


### Region of Interest

The combination of Sobel and Colospace filtering does an amazing job at detecting lane lines but it also detects several other objects and features in the surroundings which would as noise in the pipeline to be processed further. An easy way to filter this noise would be to define an area on the image where the probability of occurence of lane lines is the highest and mask out the image beyond this region of interest. 
Hence the following steps will be followed:

#### **Defining ROI**
![alt text][image22]


#### **Applying ROI mask**
![alt text][image23]


### Perspective Transform
In this step I will define a function **`warp()`** which transforms the undistorted image to a "birds eye view" of the road which focuses only on the lane lines and displays them in such a way that they appear to be relatively parallel to eachother. This will make it easier later on to fit polynomials to the lane lines and measure the curvature.

```python
def warp(image:'Undistorted Image):
    # Given src and dst points, calculate the perspective transform matrix
    # Warp the image using OpenCV warpPerspective()
    # Return the resulting image and matrix
    return warped
```
**Mapping Coordinates used:**

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 700      | 380, 720      | 
| 580, 450      | 380, 10       |
| 730, 450      | 950, 10       |
| 1160, 700     | 950, 720      |

![alt text][image24]

---
## 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

### Finding Lanes

At this point I was able to use the combined binary image to isolate lane line pixels and fit a polynomial to each of the lane lines. The space in between the identified lane lines is filled in to highlight the driveable area in the lane. The position of the vehicle was measured by taking the average of the x intercepts of each line.

The next step was to fit a polynomial to each lane line, which was done by:

* Identifying peaks in a histogram of the image to determine location of lane lines.
* Identifying all non zero pixels around histogram peaks using the numpy function numpy.nonzero().
* Fitting a polynomial to each lane using the numpy function numpy.polyfit().

The `sliding_window_search()` and `swNeighbourSearch()` help us to do the above mentioned steps. The image below demonstrates how this process works:
![alt text][image25]

The image below depicts the histogram generated by `sliding_window_search()`; the resulting base points for the left and right lanes - the two peaks nearest the center - are clearly visible:
![alt text][image26]

The `swNeighbourSearch()` function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:
![alt text][image27]


---

## 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

#### Finding Curvature `findCurvature()`

After fitting the polynomials I was able to calculate the position of the vehicle with respect to center with the following calculations:

* Calculated the average of the x intercepts from each of the two polynomials position = (rightx_int+leftx_int)/2.

* Calculated the distance from center by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis distance_from_center = abs(image_width/2 - position)

* If the horizontal position of the car was greater than image_width/2 than the car was considered to be left of center, otherwise right of center.

## 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

#### Visual Display of the Lane Boundaries and Numerical Estimation
* **`draw_data()`**
* **`drawLane()`**

![alt text][image28]

---

## Pipeline (video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

#### `findLanes()`
**Here's a [link to my video result](project_video_out.mp4)**

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The video pipeline developed in this project did a fairly robust job of detecting the lane lines in the video provided for the project, which shows a road in basically ideal conditions, with fairly distinct lane lines, and on a clear day, although it did lose the lane lines slightly momentarily when there was heavy shadow over the road from a tree.

The problems I encountered were almost exclusively due to lighting conditions, shadows, discoloration, etc. specially when the white lines didn't contrast with the rest of the image enough. This would definitely be an issue in snow or in a situation where, for example, a bright white car were driving among dull white lane lines. Producing a pipeline from which lane lines can reliably be identified was of utmost importance.

I've considered a few possible approaches for making my algorithm more robust. These include more dynamic thresholding (perhaps considering separate threshold parameters for different horizontal slices of the image, or dynamically selecting threshold parameters based on the resulting number of activated pixels), designating a confidence level for fits and rejecting new fits that deviate beyond a certain amount (this is already implemented in a relatively unsophisticated way) or rejecting the right fit (for example) if the confidence in the left fit is high and right fit deviates too much (enforcing roughly parallel fits). I hope to revisit some of these strategies in the future.

I have not yet tested the pipeline on additional video streams which could challenge the pipeline with varying lighting and weather conditions, road quality, faded lane lines, and different types of driving like lane shifts, passing, and exiting a highway. For further research I plan to record some additional video streams of my own driving in various conditions and continue to refine my pipeline to work in more varied environments.