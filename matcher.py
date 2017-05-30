import numpy as np
import cv2

 
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def fill(imageA):
    th, im_th = cv2.threshold(imageA,220,255,cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h,w = im_th.shape[:2]
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(im_floodfill,mask,(0,0),255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out
 
image = cv2.imread('template3.jpg')
template = cv2.imread('template3.jpg')
 
# resize images
image = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
template = cv2.resize(template, (0,0), fx=0.5, fy=0.5) 

# Convert to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
# Find template
result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = min_loc
h,w = templateGray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cropped = image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
cropgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

#thresholding


#edge detection
edge1 = cv2.Canny(templateGray,100,200)
edge2 = cv2.Canny(cropgray,100,200)
im_out = fill(edge2)
mse_v = mse(edge1,im_out) #mse(templateGray,cropgray)

cv2.rectangle(image,top_left, bottom_right,(255,0,0),4)

# Show result
cv2.imshow("Template", edge1)
cv2.imshow("Result", image)
cv2.imshow("cropped",edge2)
cv2.imshow("x",im_out)

cv2.moveWindow("Template", 10, 50);
cv2.moveWindow("Result", 150, 50);
cv2.moveWindow("cropped", 10, 200);

cv2.waitKey(0)

