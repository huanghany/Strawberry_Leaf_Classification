import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取文件夹中的所有图片
# input_folder = 'test_pic/test_5/2/'
# input_folder = 'test_pic/leaf_9_24/'
input_folder = 'save/leaf_9_24/'
output_folder = 'save/leaf_9_24/hs_mean'
os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # 处理 jpg 和 png 图片
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 分离H, S, V通道
        h, s, v = cv2.split(hsv_image)
        mask = (hsv_image[:, :, 0] > 0) | (hsv_image[:, :, 1] > 0) | (hsv_image[:, :, 2] > 0)
        # 创建图像和直方图的网格布局
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # 统计非黑色像素数量
        non_black_pixel_count = np.sum(mask)
        # 计算总像素数量
        total_pixel_count = image.shape[0] * image.shape[1]
        # 输出结果
        # print(f'非黑色像素点数量: {non_black_pixel_count}')
        # print(f'总像素点数量: {total_pixel_count}')
        # 显示原图
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV读取的图像为BGR格式，需转换为RGB
        axs[0].set_title('Original Image')
        axs[0].axis('off')  # 隐藏坐标轴
        h_pixels_in_range_150_180_num = np.sum((h[mask] >= 150) & (h[mask] <= 180))
        h_pixels_in_range_0_20_num = np.sum((h[mask] >= 5) & (h[mask] <= 20))
        h_pixels_in_range_150_180 = (h_pixels_in_range_150_180_num+h_pixels_in_range_0_20_num)>200
        print("H150-180数量:", h_pixels_in_range_150_180_num)
        print("H0-20数量:", h_pixels_in_range_0_20_num)
        print(h_pixels_in_range_150_180)

        axs[1].set_title('Channel histogram (H_S)')  # 绘制 H 和 S 通道的直方图
        hist_h = cv2.calcHist([h], [0], mask.astype(np.uint8), [181], [0, 181])
        axs[1].plot(hist_h, color='b', label='H Channel')
        axs[1].set_xlim([0, 256])
        axs[1].set_xticks(np.arange(0, 256, 15))
        hist_s = cv2.calcHist([s], [0], mask.astype(np.uint8), [256], [0, 256])
        axs[1].plot(hist_s, color='r', label='S Channel')
        mean_h = np.mean(h[mask])  # 只计算掩码区域的均值
        mean_s = np.mean(s[mask])
        print(mean_h)
        print(mean_s)
        if h_pixels_in_range_150_180:
            print("have red")
        if 38 <= mean_h <= 59 and not h_pixels_in_range_150_180:
            axs[1].text(50, max(max(hist_h), max(hist_s)) * 0.95, 'Healthy Leaf', color='green', fontsize=12,
                           fontweight='bold')
        else:
            axs[1].text(50, max(max(hist_h), max(hist_s)) * 0.95, 'Sick Leaf', color='r', fontsize=12,
                        fontweight='bold')

        axs[1].axvline(mean_h, color='black', linestyle='--', label=f'H Mean: {mean_h:.2f}')
        axs[1].text(mean_h + 5, max(hist_h) * 0.9, f'{mean_h:.2f}', color='black')  # 标注均值
        # 在 S 通道上标出均值并添加垂直线
        axs[1].axvline(mean_s, color='g', linestyle='--', label=f'S Mean: {mean_s:.2f}')
        axs[1].text(mean_s + 5, max(hist_s) * 0.9, f'{mean_s:.2f}', color='g')  # 标注均
        axs[1].legend()
        plt.tight_layout()
        axs[1].grid(True)

        output_image_path = os.path.join(output_folder, f'hs_{filename}.png')
        # plt.savefig(output_image_path)
        plt.show()
        plt.close()

print("处理完成，所有图片已保存。")
