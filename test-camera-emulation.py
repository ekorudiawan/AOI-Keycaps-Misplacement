import numpy as np
import cv2 as cv
import glob
from pypylon import pylon
import imutils
import os
import tempfile

os.environ["PYLON_CAMEMU"] = "1"

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.ImageFilename = "./calibration-images"
camera.ImageFileMode = "On"
camera.TestImageSelector = "Off"
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print(camera.GetDeviceInfo())
while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grab_result)
        bgr_img = image.GetArray()
        h, w = bgr_img.shape[:2]

        cv.namedWindow('Camera Viewer', cv.WINDOW_NORMAL)
        cv.imshow('Camera Viewer', bgr_img)
        k = cv.waitKey(1)
        if k == 27:
            break
    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv.destroyAllWindows()
