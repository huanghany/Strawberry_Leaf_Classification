import os
import cv2
import random
import numpy as np
from ultralytics import YOLO

model = YOLO(r'D:\ultralytics\sickleaf_and_healthyleaf_v03\weights\best.pt', task='segment')
# 输入文件夹路径
# input_folder = r'test_pic/test_2'
# input_folder = r'D:\ultralytics\save\CVPPA'
input_folder = r'D:\ultralytics\20241009'
# 输出文件夹路径
output_folder = r'save/20241009'
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
        height, weights, _ = img.shape
        # 获取分割掩膜
        masks = results[0].masks.data  # 掩膜数据
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classid = results[0].boxes.cls.cpu().numpy()
        masks = masks.cpu().numpy()

        # 遍历每个实例分割的掩膜
        for i in range(masks.shape[0]):
            # 获取当前实例的掩膜
            mask = masks[i]
            mask_resize = cv2.resize(mask, (weights, height))
            mask_resize = mask_resize.astype(bool)  # 将掩膜转换为布尔值

            # 生成随机颜色
            color = [random.randint(0, 255) for _ in range(3)]
            x1, y1, x2, y2 = boxes[i].astype(int)
            if classid[i] == 1:
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            # 将掩膜区域设置为颜色
            img[mask_resize] = img[mask_resize] * 0.5 + np.array(color) * 0.5  # 将掩膜区域与颜色融合
            # cv2.namedWindow('img', 0)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
        # 保存处理后的图片
        output_img_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_img_path, img)

        print(f'保存处理后的图片到: {output_img_path}')
