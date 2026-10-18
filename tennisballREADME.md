Blue Object Selection and Tennis Ball Detection using OpenCV

A Python program that reads a live camera stream, tracks blue objects of any shape, and detects a tennis ball at the same time. The blue objects and the tennis ball are treated as two separate things.

Features

Blue object

Area in px²
Bounding box
A line tracking the path of its centre
Its position with respect to the tennis ball (left/right, above/below, dx, dy and distance in px), with a line joining the two centres, when a ball is in view

Tennis ball

Centre (x, y)
Radius in px
Area in px²

The blue object can have any shape (rectangle, triangle, irregular). It is not assumed to be circular.

Requirements
Python 3
OpenCV and NumPy
bash
pip install opencv-python numpy
How to run
bash
python tennis_blue_tracker.py

The program opens camera 0. To use another camera, change the 0 in cv2.VideoCapture(0) inside main().

Controls
Left-click on a blue object to select it. The program learns that object's blue range from the clicked pixels.
Press r to reset to the default blue range.
Press q to quit.
My approach

The blue object and the tennis ball differ in colour and in shape, so I used those differences to keep them apart:

The tennis ball is yellow-green and round, so it has its own colour range and must pass a roundness test.
The blue object is blue and can be any shape, so it only has to be a large enough blue region. No circle is assumed.

I used the HSV colour space instead of RGB because hue stays fairly constant when the lighting changes, which makes colour thresholds more reliable.

Blue objects vary a lot (dark, light, glossy, matte), so the user can click on the object and the program learns its colour range instead of relying on one fixed range.

Method used

For every frame:

Blur the frame with a 7x7 Gaussian filter to reduce noise, then convert it from BGR to HSV.
Create a colour mask for each object with cv2.inRange.
Blue (default): H 100-130, S 120-255, V 50-255
Tennis ball: H 25-45, S 80-255, V 80-255
Clean each mask with a morphological opening (removes small specks) followed by a closing (fills small holes, such as the seam on the ball).
Find the outlines of the white regions with cv2.findContours.
Blue objects: keep every contour with an area of at least 800 px². For each one:
Area from cv2.contourArea
Bounding box from cv2.boundingRect
Centre from image moments (cv2.moments)
The centre of the largest blue object is added to a history of the last 64 positions, and the trail line is drawn through them.
Tennis ball: a contour is accepted only if all of these are true:
Area of at least 150 px²
Circularity (4 x pi x area / perimeter²) of at least 0.82
Fill ratio (contour area / area of its enclosing circle) of at least 0.78
The centre and radius come from cv2.minEnclosingCircle. If more than one contour passes, the one with the largest area x circularity is used. A square scores about 0.785 on circularity and 0.64 on fill ratio, so squares are rejected while slightly blurred balls are still accepted.
Position of the blue object with respect to the ball: dx = blue centre x - ball centre x and dy = blue centre y - ball centre y. Image y grows downward, so a positive dy means below the ball. Differences of 10 px or less are shown as "aligned".
Click to select: the median HSV value of an 11x11 patch around the click is taken, and the new blue range is that value with hue +/- 12 and saturation and value allowed to go 70 lower.

The text readouts are drawn in translucent panels on the side of the screen that covers the objects least.

Other possible methods
Hough circle transform (cv2.HoughCircles): finds circles from edges, independent of colour. It is sensitive to its parameters and can give false circles on a busy background, but it works well as a second check on the colour result.
Deep-learning detector (YOLO, SSD): train or fine-tune a model on tennis ball images. It is the most robust against lighting and clutter, but it needs data, training time and more computing power.
Other colour spaces (LAB, YCrCb): threshold on the chroma channels. These are sometimes more lighting-invariant than HSV.
K-means or Gaussian mixture colour clustering: learn the colour of the selected object automatically from a region. It adapts better than fixed ranges but is slower.
Kalman filter or CSRT/KCF tracker: predict the position between frames so tracking survives short occlusions and looks smoother, at the cost of extra complexity.
Background subtraction (MOG2): detect moving objects first and then filter by colour. It works for a fixed camera but not when the camera moves or the objects are still.
Feature matching (ORB, SIFT): match the features of a selected object. It is good for textured objects but poor for plain coloured ones.
Depth camera (RealSense, stereo): adds 3D position and real-world size, so measurements can be in cm instead of px, but it needs special hardware.
Limitations
Anything else that is yellow-green and round (a lime, a highlighter cap) can be detected as the tennis ball.
Blue objects that touch each other can merge into one region.
Very dim or strongly coloured lighting shifts the hues, so the HSV ranges at the top of the file may need adjusting. Clicking on the blue object usually fixes the blue range.
The trail follows only the largest blue object.
Sizes are in pixels, so they change with the distance from the camera.
