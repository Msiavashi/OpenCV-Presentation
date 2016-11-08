import numpy as np
import cv2

cam = cv2.VideoCapture(0)
replace_image = cv2.imread("1-beach.jpg")
rows, cols, _ = replace_image.shape
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])

while cam.isOpened():

    ret,frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_val, corners = cv2.findChessboardCorners(gray, (5, 5), None)

    if ret_val:
        pts2 = np.float32([corners[0, 0], corners[4, 0], corners[len(corners) - 1, 0], corners[len(corners) - 5, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        rows, cols, ch = frame.shape
        dst = cv2.warpPerspective(replace_image, M, (cols, rows))

        _, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), 10, 1, cv2.THRESH_BINARY_INV)

        for c in range(0, 3):
            frame[:, :, c] = dst[:, :, c]*(1 - mask) + frame[: ,: ,c]*mask[:, :]
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.imshow("image", frame)
    
cam.release()