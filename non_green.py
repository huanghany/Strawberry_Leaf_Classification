import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 输入文件夹路径
# input_folder = 'test_pic/test_3/'
input_folder = 'save/test_4/'
kernel = np.ones((5, 5), np.uint8)  # 定义一个 5x5 的卷积核，用于腐蚀
# kernel = np.ones((10, 10), np.uint8)  # 定义一个 5x5 的卷积核，用于腐蚀

# 定义绿色的HSV范围
lower_green = np.array([35, 40, 40])  # 绿色下限
upper_green = np.array([85, 255, 255])  # 绿色上限

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 处理 jpg 和 png 图片
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 创建非黑色区域的掩膜
        mask_non_black = (hsv_image[:, :, 0] > 0) | (hsv_image[:, :, 1] > 0) | (hsv_image[:, :, 2] > 0)
        non_black_pixel_count = np.sum(mask_non_black)
        mask_non_black = mask_non_black.astype(np.uint8) * 255  # 将布尔掩膜转换为二值掩膜

        # 增加腐蚀操作的迭代次数，使掩膜进一步向内收缩
        eroded_mask = cv2.erode(mask_non_black, kernel, iterations=5)  # 增加到5次迭代
        # 使用缩小后的掩膜，提取缩小后的非黑色部分
        eroded_image = cv2.bitwise_and(image, image, mask=eroded_mask)

        # 创建掩码，保留绿色部分
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        mask_green = (hsv_image[:, :, 0] >= 35) & (hsv_image[:, :, 0] <= 85) & (hsv_image[:, :, 1] > 0) & (
                hsv_image[:, :, 2] > 0)
        green_pixel_count = np.sum(mask_green)
        green_percentage = (green_pixel_count / non_black_pixel_count) * 100
        print("绿色像素数：", green_pixel_count)
        print("绿色像素占叶子总占比：", green_percentage, '%')
        # 反转掩码，保留非绿色部分
        non_green_mask = cv2.bitwise_not(green_mask)
        # 应用掩码到原图像，提取非绿色部分
        non_green_image = cv2.bitwise_and(eroded_image, eroded_image, mask=non_green_mask)
        # 创建子图并显示
        fig, axs = plt.subplots(2, 2, figsize=(8, 5))

        # 显示原始图像
        axs[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV读取的图像为BGR格式，需转换为RGB
        axs[0][0].set_title('Original Image')
        axs[0][0].axis('off')  # 隐藏坐标轴
        # 显示腐蚀后的掩膜
        axs[0][1].imshow(eroded_mask, cmap='gray')  # 显示腐蚀后的掩膜
        # axs[0][1].imshow(non_green_image, cmap='gray')  # 显示腐蚀后的掩膜
        axs[0][1].set_title('Further Eroded Mask')
        axs[0][1].axis('off')  # 隐藏坐标轴
        # 显示缩小后的图像（腐蚀效果）
        axs[1][0].imshow(cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式
        axs[1][0].set_title('Image with Further Eroded Mask')
        axs[1][0].axis('off')  # 隐藏坐标轴

        # axs[0][1].imshow(eroded_mask, cmap='gray')  # 显示腐蚀后的掩膜
        axs[1][1].imshow(non_green_image, cmap='gray')  # 显示腐蚀后的掩膜
        axs[1][1].set_title('NO green Mask')
        axs[1][1].axis('off')  # 隐藏坐标轴
        # 调整布局并显示图片
        plt.tight_layout()
        plt.show()
        plt.close()

print("处理完成，所有对比图已显示。")
