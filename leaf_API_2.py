import copy
import os
from dataclasses import dataclass

import cv2
import random
import numpy as np
from ultralytics import YOLO


@dataclass(order=True)
class LeafClassifier:
    """
    初始化叶片分类器
    Args:
        model_path: (str)模型文件路径
        input_folder:  (str)输入图片文件夹路径
        output_folder: (str)处理后图片输出文件夹路径
        H_min (int): H通道最小值。
        H_max (int): H通道最大值。
        S_min (int): S通道最小值。
        S_max (int): S通道最大值。
        iou (float): 交并比阈值。
        conf (float): 置信度阈值。
    """
    model_path: str
    input_folder: str
    output_folder: str
    H_min: int = 38
    H_max: int = 59
    S_min: int = 105
    S_max: int = 165
    iou: float = 0.85
    conf: float = 0.4
    lower_green: np.ndarray = np.array([35, 40, 40])
    upper_green: np.ndarray = np.array([85, 255, 255])

    def __post_init__(self):
        """
        初始化模型和输出文件夹
        """
        self.model = YOLO(self.model_path, task='segment')
        os.makedirs(self.output_folder, exist_ok=True)  # 创建输出文件夹

    def process_images(self):
        """
        处理输入文件夹中的所有图片
        Returns:

        """
        for filename in os.listdir(self.input_folder):
            if filename.endswith(('.jpg', '.png', '.JPG')):
                img_path = os.path.join(self.input_folder, filename)
                try:
                    self.classify_leaf(img_path, filename)  # 分类叶片
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    def process_single_image(self, img_path, output_filename):
        """
        处理单张图片
        Args:
            img_path:(str)单张图片的路径
            output_filename:(str)输出文件名

        Returns:

        """
        try:
            self.classify_leaf(img_path, output_filename)  # 分类叶片
        except Exception as e:
            print(f"处理单张图片时出错: {e}")

    def classify_leaf(self, img_path, filename):
        """
        对每张图片进行叶片分类
        Args:
            img_path (str): 图片路径。
            filename (str): 输出文件名。

        Returns:

        """
        results, img, img_1 = self.predict(img_path)  # 进行预测
        masks, boxes = results[0].masks.data.cpu().numpy(), results[0].boxes.xyxy.cpu().numpy()

        for i in range(masks.shape[0]):
            mask_resize = self.analyze_hsv(img, masks[i], boxes[i])  # 分析HSV
            if mask_resize is not None:
                self.save_results(img, img_1, mask_resize, filename)  # 保存结果

    def predict(self, img_path):
        """
        使用模型进行图片预测
        Args:
            img_path (str): 图片路径

        Returns:
            results: 预测结果
            img: 原始图片
            img_1: 深拷贝的原始图片
        """
        results = self.model(img_path, task='segment', iou=self.iou, conf=self.conf)
        img = cv2.imread(img_path)
        img_1 = copy.deepcopy(img)
        return results, img, img_1

    def analyze_hsv(self, img, mask, box):
        """
        分析图片的HSV值并标记叶片类型
        Args:
            img: 原始图片
            mask: 当前实例的掩膜
            box: 当前实例的边界框

        Returns:
            mask_resize: 调整大小的掩膜或None。
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
        print(h_pixels_in_range_150_180)  # 测试用：输出150-180范围内像素数量
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)
        mask_green = (hsv_image[:, :, 0] >= 35) & (hsv_image[:, :, 0] <= 85) & \
                     (hsv_image[:, :, 1] > 40) & (hsv_image[:, :, 2] > 40)

        mean_h = np.mean(h[mask])
        mean_s = np.mean(s[mask])
        self.mark_leaf_type(img, mask_resize, box, mean_h, mean_s, h_pixels_in_range_150_180)  # 标记叶片类型

        return mask_resize

    def mark_leaf_type(self, img, mask_resize, box, mean_h, mean_s, h_pixels_in_range_150_180):
        """
        根据H和S通道的均值标记叶片类型
        Args:
            img: 原始图片
            mask_resize: 当前掩膜
            box: 当前实例的边界框
            mean_h: H通道的均值
            mean_s: S通道的均值
            h_pixels_in_range_150_180 (int): H通道在150到180范围内的像素数量

        Returns:

        """
        x1, y1, x2, y2 = box.astype(int)
        color = np.array(random_color())

        if self.H_min <= mean_h <= self.H_max and self.S_min <= mean_s <= self.S_max and h_pixels_in_range_150_180 <= 200:
            cv2.putText(img, 'Healthyleaf', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=color.tolist(), thickness=8)
        else:  # 为病害叶片
            cv2.rectangle(img, (x1, y1), (x2, y2), color.tolist(), 6)
            cv2.putText(img, 'Sickleaf', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=3, color=color.tolist(), thickness=8)

    def save_results(self, img, img_1, mask_resize, filename):
        """
        保存处理后的图片
        Args:
            img: 原始图片
            img_1: 处理后的图片
            mask_resize: 当前掩膜
            filename (str): 输出文件名

        Returns:

        """
        color = np.array(random_color())
        img_1[mask_resize] = img[mask_resize] * 0.5 + color * 0.5
        output_img_path = os.path.join(self.output_folder, filename)
        cv2.imwrite(output_img_path, img_1)  # 保存处理后的图片
        print(f'保存处理后的图片到: {output_img_path}')


def random_color():
    """
    生成随机颜色
    Returns:
        list: RGB随机颜色值
    """
    return [random.randint(0, 255) for _ in range(3)]


if __name__ == '__main__':
    model_path = r'weights/sickleaf_and_healthyleaf.pt'
    # input_folder = r'D:\华毅\叶片数据集制作\Dataset_1\img_5'
    input_folder = r'test_pic/test_aiwei/'
    output_folder = r'save/test_API/'

    classifier = LeafClassifier(model_path, input_folder, output_folder, iou=0.85, conf=0.4)
    # classifier.process_images()  # 处理所有图片

    # 示例：处理单张图片
    single_image_path = r'test_pic/test_aiwei/6.jpg'
    classifier.process_single_image(single_image_path, '6.jpg')  # 处理单张图片

    # 代修改：将yolo检测类分出
