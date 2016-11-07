import numpy as np
import cv2

# load the desired image
image = cv2.imread("Lenna.png")

# covert it to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# load the desired haar cascade
face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")

# use the MultiScale to perform matching the xml haar with image in multi level
faces = face_cascade.detectMultiScale(gray,minNeighbors=5, scaleFactor=1.3, minSize=(30, 30))

# create a rectangle on each face
for (x, y, w, h) in faces :
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# represent the image
cv2.imshow("image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

# Done :)