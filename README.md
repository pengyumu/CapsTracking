Here, we introduce a method for tracking the positions of pen caps with different colors using linear filtering techniques. The method leverages color-based detection to distinguish between pen caps of varying colors and implements a 2D Kalman filter to track their trajectories over time. We introduce this method as follows:

### Color-based Detection

We employed predefined color ranges to distinguish between pen caps of different colors, including black, blue, and red. These color ranges were carefully selected to accurately isolate each color as follows:

- **black**: `RGB ∈ (0, 0, 0) ~ (180, 255, 50)`
- **blue**: `RGB ∈ (100, 100, 100) ~ (140, 255, 255)`
- **red**: `RGB ∈ (0, 70, 50) ~ (10, 255, 255) ∪ (170, 70, 50) ~ (180, 255, 255)`

The video frames are first converted from the RGB color space to the HSV color space using OpenCV. Within the HSV space, masks for each target color (black, blue, and red) are generated using predefined thresholds:

```
Mask_color = cv2.inRange(HSV, Lower_color, Upper_color)
```

Given the recognized masks, morphological operations, including opening and closing, are then applied to these masks to enhance the accuracy of the detected regions by removing noise. Following the morphological operations, regions in the masks with an area exceeding a threshold (e.g., 10,000 pixels) are considered outliers and filtered out. This is implemented as:

```
Filtered_Mask = Mask[cv2.contourArea(cnt) < 10000]
```
Finally, contours are extracted from the masks to precisely identify the positions of the pen caps in each frame. The centroids of these contours are calculated to determine the positions (x, y coordinates) of the pen caps. Figure 1 shows an example of pen cap color detection.

### 2D Kalman Filtering

Given the detected positions of pen caps with different colors, we leverage a 2D Kalman filter to track their positions over time. Specifically, we adapt the Kalman filter that we implemented in Assignment 2 for this task. The Kalman filter is initialized with parameters such as sampling time, control input, process noise covariance, and measurement noise covariance. The predict and update steps of the Kalman filter are then applied to estimate and refine the positions of the pen caps in each frame, based on the color-based detections. Figure 2 provides a visualization of the Kalman filter tracking results for pen caps with different colors in the video.


<br>
<br>





<p align="center">
  <img src="https://github.com/pengyumu/CapsTracking/assets/174324735/205a151b-5dfb-42ab-9101-e684bc64cb89" alt="Combined Image of Pen Cap Color Detection and Kalman Filter Tracking Results">
  <br>
</p>


<br>
<br>

https://github.com/pengyumu/CapsTracking/issues/1#issue-2385615384.mp4
