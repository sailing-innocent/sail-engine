import cv2

# read the image
img = cv2.imread('D:/data/images/mountains.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply canny edge detection
edges = cv2.Canny(gray, 100, 200)

# display the result
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
