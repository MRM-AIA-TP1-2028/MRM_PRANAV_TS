 Blue Object and Tennis Ball Detection

A Python program using OpenCV that takes live camera input and detects blue objects and a tennis ball at the same time.

 Approach

The blue object and the tennis ball are different in colour and in shape, so I used that to keep them separate. The tennis ball is yellow-green and round. The blue object is blue and can be any shape, so I did not assume it is a circle.

I used HSV colour detection because it handles changes in light better than RGB. Blue objects can be different shades, so I added a click to select option. When you click on a blue object, the program takes the colour around that point and uses it as the blue range.

On the screen, the blue object shows its area, bounding box, a line following its center and its position compared to the tennis ball. The tennis ball shows its center, radius and area.

Method used

1. Read a frame from the camera, blur it a little to remove noise and convert it to HSV.
2. Make one mask for blue and one for the tennis ball colour using cv2.inRange.
3. Clean the masks with opening (removes small dots) and closing (fills small holes).
4. Find the contours in each mask.
5. Blue object: ignore contours smaller than 800 px². For the rest, the area comes from contourArea, the bounding box from boundingRect and the center from image moments. The center of the biggest blue object is saved for the last 64 frames and joined with a line to show how it moved.
6. Tennis ball: ignore contours smaller than 150 px². For the rest, check how round the shape is (circularity = 4 x pi x area / perimeter squared) and how much of its enclosing circle it fills. Only round shapes are accepted, so a yellow square is not taken as a ball. The center and radius come from minEnclosingCircle and the area from contourArea.
7. If the ball is in the frame, dx and dy are calculated between the blue center and the ball center. The position is shown as left/right and above/below with the distance in pixels, and a line is drawn between the two centers.
8. When there is a click, the median colour of a small patch around the click is used to set a new blue range.

 Other possible methods

- Hough circle transform to find the ball by its shape instead of its colour.
- Training a detector like YOLO on tennis ball images. It works better in bad lighting but needs a dataset and more computing power.
- Using LAB or YCrCb colour space instead of HSV.
- K-means clustering to find the colour of the object automatically.
- Kalman filter or a tracker like CSRT to make the tracking smoother when the object is hidden for a moment.
- Background subtraction to find moving objects first and then check their colour.
- A depth camera to get the real size and distance in cm instead of pixels.
