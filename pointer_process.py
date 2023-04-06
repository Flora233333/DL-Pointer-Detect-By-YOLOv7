import cv2
import math


def draw_line(line, src_img):
    sum_x1 = 0
    sum_y1 = 0
    sum_x2 = 0
    sum_y2 = 0
    if line is not None:
        print(f'line_num={len(line)}')
        for i in range(0, len(line)):
            rho = line[i][0][0]
            theta = line[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            sum_x1 += x1
            sum_y1 += y1
            sum_x2 += x2
            sum_y2 += y2
            # cv2.line(src_img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)
        sum_x1 = sum_x1 // len(line)
        sum_y1 = sum_y1 // len(line)
        sum_x2 = sum_x2 // len(line)
        sum_y2 = sum_y2 // len(line)
        cv2.line(src_img, (sum_x1, sum_y1), (sum_x2, sum_y2), (255, 0, 0), 3, cv2.LINE_AA)
    return src_img, [[sum_x1, sum_y1], [sum_x2, sum_y2]]


def find_lines(img):
    # ret, img_ = cv2.threshold(img, 170, 205, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    img_canny = cv2.Canny(img, 160, 205)

    img_blur = cv2.GaussianBlur(img_canny, (3, 3), 0)

    img_erode = cv2.erode(img_blur, (5, 5), iterations=1)

    lines = cv2.HoughLines(img_erode, 1, math.pi / 180, 180)

    img_erode, xy = draw_line(lines, img_erode)
    # cv2.imshow("erosion", img_erode)
    # cv2.waitKey(0)
    print(xy)
    return xy


# img = cv2.imread("./pointer/19.jpg", 0)
# img_erode1 = find_lines(img)
# cv2.imshow("erosion", img_erode1)
# cv2.waitKey(0)

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
