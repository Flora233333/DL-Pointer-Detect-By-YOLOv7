import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import imutils


def test():
    img = cv2.imread('detect_obj_img/number/1.jpg', 0)

    # plt.hist(img.ravel(), 256, [0, 256]) # 分析灰度值，确定分割点
    # plt.show()

    ret, img_line = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    masked = cv2.bitwise_and(img, img, mask=img_line)

    masked[masked < 1] = np.mean(masked)

    _, img_th = cv2.threshold(masked, 105, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # img_blur = cv2.GaussianBlur(img_th, (3, 3), 1)

    img_b = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, (5, 5), iterations=5)

    dilation = cv2.dilate(img_b, (5, 5), iterations=20)

    nums = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    nums = imutils.grab_contours(nums)
    # print(len(nums[0]))
    draw_con_temple = cv2.drawContours(img, nums[0], -1, (0, 255, 255), 1)
    cv2.imshow('draw_con_temple', draw_con_temple)

    for i, c in enumerate(nums):
        x, y, w, h = cv2.boundingRect(c)
        if h > 15:
            cv2.rectangle(img, (x - 3, y - 5), (x + w + 3, y + h - 10), (0, 0, 255), 1)

    cv2.imshow('dilation', dilation)
    cv2.imshow('img_b', img_b)
    cv2.imshow('img', img)
    cv2.waitKey(0)


def read_number(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_line = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_blur = cv2.GaussianBlur(img_line, (3, 3), 0)

    # dilation = cv2.dilate(img_blur, (3, 3), iterations=2)

    text = pytesseract.image_to_string(img_blur, lang='eng',
                                       config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')

    print(text)

    cv2.imshow('number', img_blur)

    return img

#
# img_blur = cv2.GaussianBlur(img_line, (7, 7), 1)
#
# img_canny = cv2.Canny(img_blur, 128, 86)
#


# # erosion = cv2.erode(img_canny, (1, 1), iterations=1)
# erosion = cv2.dilate(img_canny, (3, 3), iterations=2)
# erosion = cv2.erode(erosion, (3, 3), iterations=2)
#
# # opening = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, (100, 100))
# # try:
# #     contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # except ValueError:
# #     image, contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
# # contours = [cnt for cnt in contours]
#
# print('len(contours)', len(contours))
#


# threshCnts = cv2.findContours(img_blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
# cv2.drawContours(img, threshCnts, -1, (0, 0, 255), 2)
#
# cv2.imshow('1', img_canny)
# cv2.imshow('2', img)
#
# cv2.waitKey(0)
