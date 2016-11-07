import numpy as np
import cv2
# opening camera
camera = cv2.VideoCapture(0)

# open the image that we want to replace
replace_image = cv2.imread("phoenix.png")
rows, cols, _ = replace_image.shape

# setting 4 points for out image (for 4 point transformation of OpenCV)
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

while camera.isOpened():

    # reading a single frame from camera
    _ , frame = camera.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # finding a (n*n) sub-node of a chessboard
    ret, corners = cv2.findChessboardCorners(gray, (5,5), None)

    # if a chessboard detected
    if ret :
        # for c in corners:
        #     cv2.circle(frame, (c[0,0], c[0, 1]), 3, (0, 255, 0))

        # creating 4 point from detected ROI for perspective transformation
        pts2 = np.float32([corners[0,0], corners[4, 0], corners[len(corners) - 1, 0], corners[len(corners) - 5, 0]])

        # calculating the transformation Matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        rows, cols , ch = frame.shape

        # warp the image into the desired ROI
        dst_image = cv2.warpPerspective(replace_image, M, (cols, rows))

        # threshold the dst image to inverse threshold
        _, mask = cv2.threshold(cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY), 10, 1, cv2.THRESH_BINARY_INV)

        # masking the images on every channel
        for c in range(0, 3):
            frame[: , : , c] = dst_image[: ,: ,c]*( 1 - mask ) + frame[: ,: , c]*mask[: ,: ]

    # show the image in a window called frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
camera.release()
cv2.destroyAllWindows() 