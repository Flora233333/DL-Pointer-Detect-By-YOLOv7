import cv2
import math


def draw_line(line, src_img):
    if line is not None:
        print(f'line_num={len(line)}')
        for i in range(0, len(line)):
            rho = line[i][0][0]
            theta = line[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(src_img, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)

    return src_img


img = cv2.imread("./pointer/15.jpg", 0)

# ret, img_ = cv2.threshold(img, 170, 205, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
img_canny = cv2.Canny(img, 160, 205)

img_blur = cv2.GaussianBlur(img_canny, (3, 3), 0)

img_erode = cv2.erode(img_blur, (5, 5), iterations=1)

lines = cv2.HoughLines(img_erode, 1, math.pi / 180, 180)

img_erode = draw_line(lines, img_erode)
cv2.imshow("erosion", img_erode)
# cv2.imshow("img_blur", img_blur)
# cv2.imshow("erosion", erosion)
cv2.waitKey(0)

# def nothing(x):
#     pass
#
#
# # 创建窗口
# cv2.namedWindow('Canny')
#
# # 创建滑动条，分别对应Canny的两个阈值
# cv2.createTrackbar('threshold1', 'Canny', 0, 255, nothing)
# cv2.createTrackbar('threshold2', 'Canny', 0, 255, nothing)
#
# while (1):
#
#     # 返回当前阈值
#     threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
#     threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
#
#     img_output = cv2.Canny(img, threshold1, threshold2)
#
#     # 显示图片
#     cv2.imshow('original', img)
#     cv2.imshow('Canny', img_output)
#
#     # 空格跳出
#     if cv2.waitKey(1) == ord(' '):
#         break
#
#     # 摧毁所有窗口
# cv2.destroyAllWindows()
