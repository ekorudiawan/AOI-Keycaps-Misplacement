import numpy as np
import cv2 as cv
import glob

# Define chess board size
chessboard = (7, 7)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
object_point = np.zeros((chessboard[0]*chessboard[1], 3), np.float32)
object_point[:, :2] = np.mgrid[0:chessboard[0],
                               0:chessboard[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the list_images.
object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.

list_images = glob.glob('./calibration-images/*.png')
print("List Images : ")
print(list_images)

for img_filename in list_images:
    bgr_img = cv.imread(img_filename)
    gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray_img, chessboard, None)
    # ret, corners = cv.findChessboardCorners(gray_img, (8,8), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        object_points.append(object_point)

        corners = cv.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)

        # Draw and display the corners
        bgr_img = cv.drawChessboardCorners(bgr_img, chessboard, corners, ret)
        cv.imshow('Image', bgr_img)
        cv.waitKey(500)

if len(list_images) == 0:
    print("Calibration fail, unable to open calibration images !")
else:
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        object_points, image_points, gray_img.shape[::-1], None, None)

    # Save Camera Matrix
    np.save("mtx", mtx)
    # Save Distortion Coeficient
    np.save("dist", dist)

    cv.destroyAllWindows()
    print("Calibration successful")
