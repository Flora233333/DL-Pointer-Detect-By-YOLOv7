import cv2
import math
from pointer_process import find_lines


def find_center(xy1, xy2, x_hat):
    k = (xy1[1] - xy2[1]) / (xy1[0] - xy2[0])
    b = xy1[1] - k * xy1[0]
    center = (x_hat, int(k * x_hat + b))
    return center


def draw_line(src_img, center_nut, point):
    bias = 125  # 需修正的参数

    
    xy = find_lines(src_img[point[2] + 5:point[3] - 5, point[4] + 5:point[5] - 5])

    xy[0][0] += point[4]  # 检测指针直线修正
    xy[0][1] += point[2]
    xy[1][0] += point[4]
    xy[1][1] += point[2]

    center = find_center(xy[0], xy[1], center_nut[0])
    print(center)

    cv2.line(src_img, (xy[0][0], xy[0][1]), (center[0], center[1]), (0, 255, 0), 5)  # 画指针线

    fix_center_nut_y = center_nut[1] - bias  # 修正后的中心y坐标

    # cv2.line(src_img, (point[0], point[1]), (center_nut[0], fix_center_nut_y), (0, 255, 0), 5)  # 画指针线

    cv2.line(src_img, (center[0], 0), (center[0], src_img.shape[0]), (255, 255, 0), 5)  # 中心y轴
    cv2.line(src_img, (0, center[1]), (src_img.shape[1], center[1]), (255, 255, 0), 5)  # 中心x轴

    a = math.radians(90)  # 旋转到左边-0.1 (-45度)
    r_x = (center[0] - center[0]) * math.cos(a) - (center[1] - src_img.shape[0]) * math.sin(-a) + \
          center[0]
    r_y = (center[0] - center[0]) * math.sin(-a) + (center[1] - src_img.shape[0]) * math.cos(a) + \
          src_img.shape[0]

    cv2.line(src_img, (center[0], center[1]), (int(r_x), int(r_y)), (255, 255, 0), 5)

    b = math.radians(90)  # 旋转到右边0.9 (-45度)
    r_x = (center[0] - center[0]) * math.cos(b) - (center[1] - src_img.shape[0]) * math.sin(b) + \
          center[0]
    r_y = (center[0] - center[0]) * math.sin(b) + (center[1] - src_img.shape[0]) * math.cos(b) + \
          src_img.shape[0]

    # print(r_x, r_y)

    cv2.line(src_img, (center[0], center[1]), (int(r_x), int(r_y)), (255, 255, 0), 5)

    a = [r_x - center_nut[0], r_y - fix_center_nut_y]  # 重新标定以表盘中心为原点的坐标

    b = [point[0] - center_nut[0], point[1] - fix_center_nut_y]  # 重新标定以表盘中心为原点的坐标

    beta = math.acos(  # ab=|a||b|cos(a)
        (a[0] * b[0] + a[1] * b[1]) / (math.sqrt(a[0] ** 2 + a[1] ** 2) * math.sqrt(b[0] ** 2 + b[1] ** 2)))

    # print(a, b)
    print(f'beta={beta * 180 / math.pi}')

    print(f'center_nut = {center_nut[0], fix_center_nut_y}')
    print(f'pointer = {point[0], point[1]}')

    k = -(fix_center_nut_y - point[1]) / (center_nut[0] - point[0])
    print('k = %f' % k)

    eps = math.radians(45) - math.atan(k)  # (180 - 98) / 2

    ra = 1 / math.radians(270)
    num = ra * eps - 0.1
    print(f'num = {num}')

    cv2.putText(src_img, f'num={num:.5f}', (point[0], point[1] + 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                thickness=3, lineType=cv2.LINE_AA)

    return src_img
