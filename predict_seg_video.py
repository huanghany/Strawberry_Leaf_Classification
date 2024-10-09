from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的模型
    model = YOLO(r'D:\ultralytics\sickleaf_and_healthyleaf_v03\weights\best.pt')

    # 设置视频路径
    video_path = r'test2.mp4'

    # 开始检测并保存结果
    results = model.predict(source=video_path, task='segment', save=True, show=False)
