import numpy as np
import cv2 as cv
import os

def main():
    cwd = os.getcwd()
    filename = os.path.join(cwd, "./dummy-images/3.png") 
    img = cv.imread(filename)
    cv.imshow("Image", img)
    r = cv.selectROI(img)
    print(r)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()