import numpy as np
import cv2 as cv
from pypylon import pylon
import imutils
import os

# mode emulation
emulation_mode = True

# Load camera pand distortion matrix
with open('mtx.npy', 'rb') as f:
    mtx = np.load(f)
with open('dist.npy', 'rb') as f:
    dist = np.load(f)

# If using emulation camera
if emulation_mode:
    os.environ["PYLON_CAMEMU"] = "1"

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grab_result)
        bgr_img = image.GetArray()
        h, w = bgr_img.shape[:2]
        new_mtx, roi = cv.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        undistort_bgr_img = cv.undistort(bgr_img, mtx, dist, None, new_mtx)
        # undistort image
        undistort_bgr_img = imutils.rotate(undistort_bgr_img, angle=180)

        cv.namedWindow('Camera Viewer', cv.WINDOW_NORMAL)
        cv.imshow('Camera Viewer', undistort_bgr_img)
        k = cv.waitKey(1)
        if k == 27:
            break
    grab_result.Release()

camera.StopGrabbing()
cv.destroyAllWindows()
