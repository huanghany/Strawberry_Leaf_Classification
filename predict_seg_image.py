import numpy as np

from ultralytics import YOLO
import glob

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO(r'D:\ultralytics\sickleaf_and_healthyleaf_v03\weights\best.pt')

    # 设置图像路径
    # path = r'D:\ultralytics\save\CVPPA'
    # path = r'D:\ultralytics\save\aiwei'
    # path = r'D:\ultralytics\save\aiwei_1'
    # path = r'D:\ultralytics\save\Sick'
    # path = r'D:\ultralytics\save\healthy'
    # path = r'D:\ultralytics\aiwei_phone'
    path = r'D:\ultralytics\20241009'

    # path = r'D:\华毅\叶片数据集制作\camera\aiwei_9_14\realsense_image_20240914_152528.png'
    files = glob.iglob(path)
    sorted_files=sorted(files)
    count = 0
    for file in sorted_files:
        # 开始检测并保存结果
        # results = model.predict(source=file, save=True, show=False, conf=0.1)
        results = model.predict(source=file, save=True, show=False)
        # res = results[0]
        # cls = res.boxes.cls.cpu().numpy().astype(np.int8).tolist()
        # if 6 in cls:
        #     file_name = file.split("/")[-1]
        #     print(file_name)
        #     f.write(file_name)
        #     f.write("\n")


