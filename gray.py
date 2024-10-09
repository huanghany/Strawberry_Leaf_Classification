import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread('test_pic/leaf_test/0005_02.jpg')
# image = cv2.imread('test_pic/leaf_test/10.jpg')

# 将图像从 BGR 转换为 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 提取 V 通道（亮度）
v_channel = hsv_image[:, :, 2]

# 设置一个阈值，忽略黑色部分（亮度值接近 0 的部分）
threshold = 10  # 可以根据需要调整
mask = v_channel > threshold

# 对 V 通道中亮度大于阈值的区域进行均衡化
# 提取满足条件的像素进行均衡化
v_channel_equalized = v_channel.copy()
# 提取满足条件的像素进行均衡化
v_channel_to_equalize = v_channel[mask]

# 进行直方图均衡化，返回均衡化后的值
if v_channel_to_equalize.size > 0:  # 确保有像素被均衡化
    equalized_values = cv2.equalizeHist(v_channel_to_equalize.astype(np.uint8))

    # 将均衡化后的值放回原数组
    v_channel_equalized[mask] = equalized_values  # 这里确保 equalized_values 与 mask 匹配


# 将均衡化后的 V 通道重新赋值回 HSV 图像
hsv_image[:, :, 2] = v_channel_equalized

# 将图像从 HSV 转换回 BGR
equalized_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# 显示原始图像和均衡化后的图像
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 将 BGR 转换为 RGB 以便显示

plt.subplot(1, 2, 2)
plt.title('Equalized Image (Ignore Black)')
plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))  # 将 BGR 转换为 RGB 以便显示

plt.show()
