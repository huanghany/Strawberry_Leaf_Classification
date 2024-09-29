import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取文件夹中的所有图片
input_folder = 'test_pic/test_5/2/'
# input_folder = 'test_pic/leaf_9_24/'
output_folder = 'save/test_4/processed_2/'
os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 处理 jpg 和 png 图片
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 分离H, S, V通道
        h, s, v = cv2.split(hsv_image)
        mask = (hsv_image[:, :, 0] > 0) | (hsv_image[:, :, 1] > 0) | (hsv_image[:, :, 2] > 0)

        # 创建图像和直方图的网格布局（2行2列）
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))  # 改为四个子图
        # 统计非黑色像素数量
        non_black_pixel_count = np.sum(mask)
        # 创建掩膜：红色像素
        mask_red = (hsv_image[:, :, 0] < 20) & (hsv_image[:, :, 1] > 0) & (hsv_image[:, :, 2] > 0)

        # 创建掩膜：黄色像素
        mask_yellow = (hsv_image[:, :, 0] >= 20) & (hsv_image[:, :, 0] <= 33) & (hsv_image[:, :, 1] > 0) & (
                    hsv_image[:, :, 2] > 0)

        # 统计红色和黄色像素数量
        red_pixel_count = np.sum(mask_red)
        yellow_pixel_count = np.sum(mask_yellow)

        # 计算总像素数量
        total_pixel_count = image.shape[0] * image.shape[1]

        # 计算红色和黄色占比
        red_percentage = (red_pixel_count / non_black_pixel_count) * 100
        yellow_percentage = (yellow_pixel_count / non_black_pixel_count) * 100

        # 输出结果
        print(f'非黑色像素点数量: {non_black_pixel_count}')
        print(f'总像素点数量: {total_pixel_count}')
        print(f'红色像素点数量: {red_pixel_count}')
        print(f'红色像素占比: {red_percentage:.2f}%')
        print(f'黄色像素点数量: {yellow_pixel_count}')
        print(f'黄色像素占比: {yellow_percentage:.2f}%')
        if red_percentage+yellow_percentage > 11:
            print("有明显红斑")
        elif yellow_percentage > 5:
            print("有明显黄斑")
        else:
            print("健康")
        # 显示原图
        axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV读取的图像为BGR格式，需转换为RGB
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')  # 隐藏坐标轴

        # 绘制H和S通道的直方图
        axs[0, 1].set_title('Channel histogram (H_S)')
        axs[0, 1].set_xlim([0, 256])
        axs[0, 1].set_xticks(np.arange(0, 256, 15))  # 每隔15单位增加一个X轴刻度
        hist_h = cv2.calcHist([h], [0], mask.astype(np.uint8), [181], [0, 181])
        axs[0, 1].plot(hist_h, color='b', label='H Channel')
        hist_s = cv2.calcHist([s], [0], mask.astype(np.uint8), [256], [0, 256])
        axs[0, 1].plot(hist_s, color='r', label='S Channel')
        axs[0, 1].legend()

        # 显示红色掩码
        red_mask_colored = np.zeros_like(image)  # 创建一个与原图大小相同的空图像
        red_mask_colored[mask_red] = image[mask_red]  # 仅在红色区域显示原图像素
        axs[1, 0].imshow(cv2.cvtColor(red_mask_colored, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Red Mask')
        axs[1, 0].axis('off')  # 隐藏坐标轴

        # 显示黄色掩码
        yellow_mask_colored = np.zeros_like(image)  # 创建一个与原图大小相同的空图像
        yellow_mask_colored[mask_yellow] = image[mask_yellow]  # 仅在黄色区域显示原图像素
        axs[1, 1].imshow(cv2.cvtColor(yellow_mask_colored, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title('Yellow Mask')
        axs[1, 1].axis('off')  # 隐藏坐标轴
        axs[0, 1].grid(True)
        # 调整布局
        plt.tight_layout()

        # 保存结果图像
        output_image_path = os.path.join(input_folder, f'hs_{filename}.png')
        # plt.savefig(output_image_path)

        plt.show()

print("处理完成，所有图片已保存。")
