import cv2
import numpy as np

# 读取图像
# image = cv2.imread('test_pic/leaf_test/0005_02.jpg')
# image = cv2.imread('test_pic/leaf_test/0000_00.jpg')
image = cv2.imread('test_pic/leaf_test/10.jpg')

# 转换到HSV空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取V通道
h, s, v = cv2.split(hsv_image)
# 对V通道进行均衡化
v_eq = cv2.equalizeHist(v)

# 合并通道
hsv_eq = cv2.merge((h, s, v_eq))

# 转换回BGR空间
result_image = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# 显示结果
cv2.namedWindow('Original Image', 0)
cv2.namedWindow('Equalized Image', 0)
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
