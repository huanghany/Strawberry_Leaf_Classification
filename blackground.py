import os
import cv2
import numpy as np
from ultralytics import YOLO

# model = YOLO(r'D:\华毅\叶片数据集制作\yolov8\best.pt', task='segment')
model = YOLO('weights/sickleaf_and_healthyleaf.pt', task='segment')
# 输入文件夹路径
input_folder = r'test_pic/test_3'
# 输出文件夹路径
output_folder = r'save/test_4'

# 创建输出文件夹，如果不存在的话
os.makedirs(output_folder, exist_ok=True)

# 处理文件夹中的每一张图片
for filename in os.listdir(input_folder):
    # 只处理图片文件
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 构造图片的完整路径
        img_path = os.path.join(input_folder, filename)
        # 加载图片并进行推理
        results = model(img_path, task='segment')

        # 读取原始图片
        img = cv2.imread(img_path)
        height, width, _ = img.shape  # 获取图片的尺寸

        # 获取分割掩膜 (Tensor)、边界框和类别标签
        masks = results[0].masks.data  # 获取掩膜
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 获取边界框 (x1, y1, x2, y2)
        class_ids = results[0].boxes.cls.cpu().numpy()  # 获取每个实例的类别标签

        # 将掩膜从 Tensor 转为 NumPy 数组
        masks = masks.cpu().numpy()

        # 创建一个黑色背景
        black_background = np.zeros_like(img)

        # 遍历每个实例分割的掩膜
        for i in range(masks.shape[0]):
            # 获取当前实例的掩膜并调整大小为与原始图片一致
            mask = masks[i]
            mask_resized = cv2.resize(mask, (width, height))  # 调整掩膜大小为与图像相同
            mask_resized = mask_resized.astype(bool)  # 将掩膜转换为布尔类型

            # 将掩膜对应的实例部分保留，背景部分设为黑色
            black_background[mask_resized] = img[mask_resized]

            # 找到掩膜的边界轮廓
            contours, _ = cv2.findContours(mask_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 保存处理后的图片
        output_img_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_path, black_background)

        print(f'保存处理后的图片到: {output_img_path}')
