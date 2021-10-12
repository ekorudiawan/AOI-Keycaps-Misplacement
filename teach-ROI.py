import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# image_path
image = cv.imread("./dummy-images/golden.png")
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# image = imutils.rotate(test_img, angle=180)
# select ROIs function
ROIs = cv.selectROIs("Select ROI's", image)
# print rectangle points of selected roi
print(ROIs)

i = 0
total_backlit = 21
total_ocr = 43
backlit_template = np.zeros((total_backlit, 256), dtype=np.float32)
for rect in ROIs:
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    start_point = (x, y)
    end_point = (x+w, y+h)

    if i < total_ocr:
        img = cv.rectangle(image, start_point, end_point,
                           (0, 0, 255), thickness=2)
    else:
        img = cv.rectangle(image, start_point, end_point,
                           (0, 255, 255), thickness=2)
        sub_img = gray_image[y:y+h, x:x+w].copy()
        sub_img_fix = cv.resize(sub_img, (100, 100))
        hist = cv.calcHist([sub_img_fix], [0], None, [256], [0, 256])
        norm_hist = hist/np.sum(hist)
        backlit_template[i-total_ocr,:] = norm_hist.reshape(1, 256).copy()

    i += 1

print("Backlit ", backlit_template)

with open(r'Backlit.npy', 'wb') as file:
    np.save(file, backlit_template)
with open(r'OCR.npy', 'wb') as file:
    np.save(file, ROIs)

cv.imshow("Selected ROI's", img)
cv.waitKey(0)
