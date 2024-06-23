import cv2
import numpy as np
import matplotlib.pyplot as plt

# Basic image test
image = cv2.imread('finding-lanes/test_image.jpg') #Returns image as numpy array
cv2.imshow('original image', image) #Shows the image in a window called 'original image'
cv2.waitKey(0) #Image window is displayed until any key is pressed (indefinite)

# Gray-scale conversion for edge detection: gray-scale increase computational efficiency for edge detection
lane_image = np.copy(image) #Make copy to avoid affecting original image
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

# Add Gaussian Blur to reduce noise
blur_image = cv2.GaussianBlur(gray_image, (5,5), 0)

# Canny Edge Detection
#Uses derivative (by converting image to [x, y] coordinates) to find color gradient (contrast). If gradient is larger than upper threshold it is accepted as an edge pixel. If smaller, it is rejected.
canny_image = cv2.Canny(blur_image, 50, 150)
#cv2.imshow('result', canny_image)
#cv2.waitKey(0)

# Function Version
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

#Confirm canny function works
canny_im = canny(lane_image)
cv2.imshow("canny edge image", canny_im) 
cv2.waitKey(0)

# Identifying Lane Lines

#Create a function that encloses the area of the lane
def region_of_interest(image):
    height = image.shape[0] #Height of an image is essentially number of rows it has
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ]) #Create a polygon with 3 vertices to isolate a field of view of the lanes on the road on the test_image; Vertices are placed like [x,y] coordinate system
    mask = np.zeros_like(image) #Creates a mask with same dimensions as image. Mask image is black as all pixels are 0
    cv2.fillPoly(mask, polygons, 255) #Places the polygon vertices on the mask and fills the enclosed space with the color white (255)
    masked_image = cv2.bitwise_and(image, mask) #Bitwise_and will take the bitwise of each homologous pixel in both arrays, ultimately masking the canny image to only show region of interest traced by the polygonal contour of the mask
    return masked_image

cropped_image = region_of_interest(canny_image) #Create a cropped image enclosing the region of interest
cv2.imshow("cropped canny edge image", cropped_image)
cv2.waitKey(0)

#Finding Lane Lines using Hough Transform

def display_lines(image, lines):
    line_im = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            #Reshape the line array to get the coordinates of the endpoints
            x1, y1, x2, y2 = line.reshape(4)
            #The line is drawn from (x1, y1) to (x2, y2) with blue color (255, 0, 0) and thickness of 10 pixels
            cv2.line(line_im, (x1, y1), ( x2, y2), (255, 0, 0), 10)
    return line_im

def make_coordinates(image, line_parameters):
    #Unpack the slope and intercept from the line parameters
    slope, intercept = line_parameters
    #Set y1 to the height of the image (bottom of the image)
    y1 = image.shape[0]
    #Set y2 to 60% of the height of the image
    y2 = int(y1*0.6)
    #Calculate x1 and x2 using the line equation (y = mx + b), rearranged to x = (y - b)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    #Return the coordinates of the line as a NumPy array
    return np.array([x1, y1, x2, y2])
    
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        #Fit a linear polynomial (line; hence the parameter '1') to the points, return the slope and intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters[0], parameters[1]
        #Determine if the line is on the left or right based on the slope
        if slope < 0:
            #Negative slope indicates a left lane line
            left_fit.append((slope, intercept))
        else:
            #Positive slope indicates a right lane line
            right_fit.append((slope, intercept))
    #Calculate the average slope and intercept for the left and right lane lines; axis=0 means columns are averaged
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    #Generate the coordinates for the left and right lane lines
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    #Return the coordinates of the left and right lane lines as a NumPy array
    return np.array([left_line, right_line])

#Apply the Hough Line Transform algorithm to detect lines in the cropped image.
# -The first argument is the input image
# -The second argument is the distance resolution of the accumulator in pixels; rho
# -The third argument is the angle resolution of the accumulator in radians; theta
# -The fourth argument is the threshold parameter: only lines with enough votes are returned.
# -The fifth argument is a placeholder for output, which is ignored here.
# -The sixth argument is the minimum length of a line to be accepted.
# -The seventh argument is the maximum allowed gap between segments to be considered a single line.
#The function returns an array of detected lines, where each line is represented by its endpoints (x1, y1, x2, y2).
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

line_image = display_lines(lane_image, lines)

#Using average line values
averaged_lines = average_slope_intercept(lane_image, lines)
line_image_average = display_lines(lane_image, averaged_lines)

#Combine the original image (0.8 weight) and the line image (1 weight) to highlight lanes; last parameter is just a scalar value of 1
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
combo_image_average = cv2.addWeighted(lane_image, 0.8, line_image_average, 1, 1)

cv2.imshow("line image", line_image)
cv2.waitKey(0)
cv2.imshow("line transposed image", combo_image)
cv2.waitKey(0)
cv2.imshow("average line transposed image", combo_image_average)
cv2.waitKey(0)

# Lane Detection with Video Input

video = cv2.VideoCapture('finding-lanes/test2.mp4')
while(video.isOpened()):
    #Read each video frame
    _, frame = video.read()
    canny_image = canny(frame)
    #Copy code logic used for still image
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    #Wait for 1 millisecond and check if the 'a' key is pressed, if 'a' key is pressed, break out of the loop
    cv2.imshow("video", combo_image)
    if cv2.waitKey(1) == ord('a'):
        break
video.release() #Release the video capture object
cv2.destroyAllWindows() #Close all OpenCV windows