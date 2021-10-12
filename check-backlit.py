import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    # image_path
    image = cv.imread("./dummy-images/13.png")
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    with open('ROI.npy', 'rb') as f:
        roi = np.load(f)
    with open('backlit.npy', 'rb') as f:
        backlit_temp = np.load(f)

    for i in range(0, roi.shape[0]):
        x = roi[i, 0]
        y = roi[i, 1]
        w = roi[i, 2]
        h = roi[i, 3]
        start_point = (x, y)
        end_point = (x+w, y+h)
        result_img = cv.rectangle(image, start_point, end_point,
                                  (0, 255, 255), thickness=2)
        check_roi = gray_image[y:y+h, x:x+w].copy()
        check_hist = cv.calcHist([check_roi], [0], None, [256], [0, 256])
        norm_check_hist = check_hist/np.sum(check_hist)
        dist = np.linalg.norm(backlit_temp[i] - norm_check_hist)
        print("Distance ", dist)

    cv.imshow("Hasil", result_img)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
