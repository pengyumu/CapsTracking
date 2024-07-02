# CapsTracking


Here, we introduce a method for tracking the positions of pen caps with different colors using linear filtering techniques. The method leverages color-based detection to distinguish between pen caps of varying colors and implements a 2D Kalman filter to track their trajectories over time. We introduce this method as follows:

### Color-based Detection

We employed predefined color ranges to distinguish between pen caps of different colors, including black, blue, and red. These color ranges were carefully selected to accurately isolate each color as follows:

- **black**: `RGB ∈ (0, 0, 0) ~ (180, 255, 50)`
- **blue**: `RGB ∈ (100, 100, 100) ~ (140, 255, 255)`
- **red**: `RGB ∈ (0, 70, 50) ~ (10, 255, 255) ∪ (170, 70, 50) ~ (180, 255, 255)`

The video frames are first converted from the RGB color space to the HSV color space using OpenCV. Within the HSV space, masks for each target color (black, blue, and red) are generated using predefined thresholds:

```python
Mask_color = cv2.inRange(HSV, Lower_color, Upper_color)```python

Given the recognized masks, morphological operations, including opening and closing, are then applied to these masks to enhance the accuracy of the detected regions by removing noise. Following the morphological operations, regions in the masks with an area exceeding a threshold (e.g., 10,000 pixels) are considered outliers and filtered out. This is implemented as:

Filtered_Mask = Mask[cv2.contourArea(cnt) < 10000]
