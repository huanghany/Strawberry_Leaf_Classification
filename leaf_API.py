import copy
import os
import cv2
from utils import random_color
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
import argparse


@dataclass
class Opt:
    """用于存储所有可配置参数的选项类

    Attributes:
        model_path (str): YOLO 模型的路径
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
        iou (float): IoU 阈值
        conf (float): 置信度阈值
        H_min (int): H 通道的最小值，用于分类健康叶片
        H_max (int): H 通道的最大值，用于分类健康叶片
        S_min (int): S 通道的最小值，用于分类健康叶片
        S_max (int): S 通道的最大值，用于分类健康叶片
        lower_green (np.ndarray): 绿色的HSV范围下界
        upper_green (np.ndarray): 绿色的HSV范围上界
    """
    model_path: str = 'weights/sickleaf_and_healthyleaf.pt'
    input_folder: str = 'D:/华毅/叶片数据集制作/Dataset_1/img_5'
    output_folder: str = 'save/test_healthy/'
    iou: float = 0.85
    conf: float = 0.4
    H_min: int = 38
    H_max: int = 59
    S_min: int = 105
    S_max: int = 165
    save: bool = False
    show: bool = False
    lower_green: np.ndarray = field(default_factory=lambda: np.array([35, 40, 40]))
    upper_green: np.ndarray = field(default_factory=lambda: np.array([85, 255, 255]))


@dataclass
class YOLODetector:
    """YOLO 模型检测器类，用于加载和运行YOLO模型。

    Attributes:
        opt (Opt): 配置参数实例。
    """

    def __init__(self, opt: Opt):
        """初始化YOLO模型。

        Args:
            opt (Opt): 配置参数实例。
        """
        self.opt = opt
        self.model = YOLO(self.opt.model_path, task='segment')

    def predict(self, img_path):
        """对图片进行YOLO预测

        Args:
            img_path (str): 要进行检测的图片路径

        Returns:
            results: YOLO模型的预测结果
            img: 原始图片
            img_1: 原始图片的深拷贝
        """
        results = self.model(img_path, task='segment', iou=self.opt.iou, conf=self.opt.conf)
        img = cv2.imread(img_path)  # 读取原始图片
        img_1 = copy.deepcopy(img)  # 深拷贝图片
        return results, img, img_1


@dataclass
class LeafClassifier:
    """叶片分类器类，处理叶片的健康和病害分类
    """

    def __init__(self, opt: Opt, detector: YOLODetector):
        """初始化叶片分类器，创建输出文件夹。

        Args:
            opt (Opt): 配置参数实例。
            detector (YOLODetector): 用于执行YOLO预测的YOLODetector实例。
        """
        self.opt = opt
        self.detector = detector
        os.makedirs(self.opt.output_folder, exist_ok=True)

    def process_images(self):
        """处理输入文件夹中的所有图片

        从 input_folder 中读取图片，对每一张图片调用 classify_leaf 进行叶片分类处理
        """
        for filename in os.listdir(self.opt.input_folder):
            if filename.endswith(('.jpg', '.png', '.JPG')):
                img_path = os.path.join(self.opt.input_folder, filename)
                try:
                    self.classify_leaf(img_path, filename)  # 分类叶片
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    def process_single_image(self, img_path, output_filename):
        """处理单张图片

        Args:
            img_path (str): 单张图片的路径
            output_filename (str): 输出文件名
        """
        try:
            self.classify_leaf(img_path, output_filename)  # 分类叶片
        except Exception as e:
            print(f"处理单张图片时出错: {e}")

    def classify_leaf(self, img_path, filename):
        """对每张图片进行叶片分类

        调用YOLO模型预测，分析图片的HSV值并保存分类结果

        Args:
            img_path (str): 图片路径
            filename (str): 输出文件名
        """
        results, img, img_1 = self.detector.predict(img_path)  # 使用YOLO检测器进行预测
        masks, boxes = results[0].masks.data.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()

        for i in range(masks.shape[0]):
            mask_resize = self.analyze_hsv(img, masks[i], boxes[i])  # 分析HSV
            if mask_resize is not None:
                if self.opt.save:
                    self.save_results(img, img_1, mask_resize, filename)  # 保存结果
                if self.opt.show:
                    self.show_results(img, img_1, mask_resize)

    def analyze_hsv(self, img, mask, box):
        """分析图片的HSV值并标记叶片类型

        Args:
            img: 原始图片
            mask: 当前实例的掩膜
            box: 当前实例的边界框

        Returns:
            mask_resize: 调整大小的掩膜或None
        """
        height, weights, _ = img.shape
        mask_resize = cv2.resize(mask, (weights, height)).astype(bool)
        instance = cv2.bitwise_and(img, img, mask=mask_resize.astype(np.uint8))

        hsv_image = cv2.cvtColor(instance, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv_image)
        mask = (hsv_image[:, :, 0] > 0) | (hsv_image[:, :, 1] > 0) | (hsv_image[:, :, 2] > 0)
        non_black_pixel_count = np.sum(mask)
        total_pixel_count = img.shape[0] * img.shape[1]

        if non_black_pixel_count / total_pixel_count < 0.01:
            return None  # 若非黑色像素占比小于1%，则返回None

        # 计算H通道在150到180范围内的像素数量
        h_pixels_in_range_150_180 = np.sum((h[mask] >= 150) & (h[mask] <= 180))

        green_mask = cv2.inRange(hsv_image, self.opt.lower_green, self.opt.upper_green)  # 绿色掩码
        mask_green = (hsv_image[:, :, 0] >= 35) & (hsv_image[:, :, 0] <= 85) & \
                     (hsv_image[:, :, 1] > 40) & (hsv_image[:, :, 2] > 40)  # 绿色掩码 用于计算绿色占比 目前没采用

        mean_h = np.mean(h[mask])
        mean_s = np.mean(s[mask])
        self.mark_leaf_type(img, box, mean_h, mean_s, h_pixels_in_range_150_180)  # 标记叶片类型

        return mask_resize

    def mark_leaf_type(self, img, box, mean_h, mean_s, h_pixels_in_range_150_180):
        """根据H和S通道的均值标记叶片类型

        Args:
            img: 原始图片
            box: 当前实例的边界框
            mean_h: H通道的均值
            mean_s: S通道的均值
            h_pixels_in_range_150_180 (int): H通道在150到180范围内的像素数量
        """
        x1, y1, x2, y2 = box.astype(int)
        color = np.array(random_color())

        if (self.opt.H_min <= mean_h <= self.opt.H_max and self.opt.S_min <= mean_s <= self.opt.S_max and
                h_pixels_in_range_150_180 <= 200):
            cv2.putText(img, 'Healthyleaf', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=color.tolist(), thickness=8)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 6)
            cv2.putText(img, 'Sickleaf', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=color.tolist(), thickness=8)

    def save_results(self, img, img_1, mask_resize, filename):
        """保存处理后的图片

        Args:
            img: 原始图片
            img_1: 处理后的图片
            mask_resize: 当前掩膜
            filename (str): 输出文件名
        """
        color = np.array(random_color())
        img_1[mask_resize] = img[mask_resize] * 0.5 + color * 0.5
        output_img_path = os.path.join(self.opt.output_folder, filename)
        cv2.imwrite(output_img_path, img_1)
        print(f'保存处理后的图片到: {output_img_path}')

    def show_results(self, img, img_1, mask_resize):
        """显示处理后的图片

        Args:
            img: 原始图片
            img_1: 处理后的图片
            mask_resize: 当前掩膜
        """
        color = np.array(random_color())
        img_1[mask_resize] = img[mask_resize] * 0.5 + color * 0.5
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img_1)
        cv2.waitKey(0)


def parse_args():
    """解析命令行参数

    Returns:
        argparse.Namespace: 命令行参数
    """
    parser = argparse.ArgumentParser(description="Leaf classification using YOLO and HSV analysis.")
    parser.add_argument('--model_path', type=str, default='weights/sickleaf_and_healthyleaf.pt',
                        help='Path to the YOLO model.')
    parser.add_argument('--input_folder', type=str, default='D:/华毅/叶片数据集制作/Dataset_1/img_5',
                        help='Input folder path.')
    parser.add_argument('--output_folder', type=str, default='save/test_1/', help='Output folder path.')
    parser.add_argument('--iou', type=float, default=0.85, help='IoU threshold for YOLO.')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold for YOLO.')
    parser.add_argument('--H_min', type=int, default=38, help='Minimum H value for healthy leaf classification.')
    parser.add_argument('--H_max', type=int, default=59, help='Maximum H value for healthy leaf classification.')
    parser.add_argument('--S_min', type=int, default=105, help='Minimum S value for healthy leaf classification.')
    parser.add_argument('--S_max', type=int, default=165, help='Maximum S value for healthy leaf classification.')
    parser.add_argument('--save', type=bool, default=False, help='Whether to save the results.')
    parser.add_argument('--show', type=bool, default=False, help='Whether to show the results.')
    return parser.parse_args()


if __name__ == '__main__':
    # 从命令行解析参数
    args = parse_args()

    # 初始化配置参数
    opt = Opt(
        model_path=args.model_path,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        iou=args.iou,
        conf=args.conf,
        H_min=args.H_min,
        H_max=args.H_max,
        S_min=args.S_min,
        S_max=args.S_max,
        save=False,
        show=True
    )

    # 初始化YOLO检测器
    yolo_detector = YOLODetector(opt=opt)

    # 初始化叶片分类器
    classifier = LeafClassifier(opt=opt, detector=yolo_detector)

    # 处理所有图片
    # classifier.process_images()

    # 示例：处理单张图片
    single_image_path = r'D:\华毅\叶片数据集制作\Dataset_1\img_5\394.jpg'
    classifier.process_single_image(single_image_path, '394_processed.jpg')  # 处理单张图片
