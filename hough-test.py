from skimage import io, color, feature, transform, draw, util
from matplotlib import pyplot as plt
import numpy as np
import cv2

def test_skimage():
    img = io.imread("fence.png")
    img2 = color.rgb2gray(img)
    img2 = feature.canny(img2, 3)

    io.imsave("skimage-canny.png", util.img_as_uint(img2))

    lines = transform.probabilistic_hough_line(img2, line_length=3, line_gap=2)

    plt.figure(figsize=(10, 10))
    i = plt.imshow(img)
    i.set_cmap('hot')
    plt.axis('off')

    for p0, p1 in lines:
        plt.plot([p0[0], p1[0]], [p0[1], p1[1]], "r-")

    plt.savefig("skimage.png", bbox_inches='tight')

def test_opencv():
    img = cv2.imread("fence.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sigma = 0.3
    v = np.median(img2)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    img2 = cv2.GaussianBlur(img2, (7,7), 3)
    img2 = cv2.Canny(img2, lower, upper)

    cv2.imwrite('opencv-canny.png', img2)

    lines = cv2.HoughLinesP(img2, 1, np.pi / 180, 10, 3, 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imwrite('opencv.png',img)

test_skimage()
test_opencv()