# Strawberry_Leaf_Classification

## 介绍

这是一个草莓叶片分割项目，在利用YoloV8进行实例分割后对单个叶片实例进行HSV颜色分析，从而将图片中叶片分为健康叶片和不健康叶片。
（不健康叶片有矩形框）

## 仓库结构

- 主要实现代码：`leaf_API.py`
- 分析结果：`save`
- 测试图片：`test_pic`
- 权重文件：`weight`
- 四个版本的数据集训练结果和日志：`sickleaf_and_healthyleaf_v0*`
- 预测视频代码：`predict_seg_video.py`
- 预测图片代码(原)：`predict_seg_image.py`
- 预测图片代码(每个叶片不一样的颜色掩膜，病害叶片有矩形框)：`predict_leaf.py`
- 测试视频：`test.mp4`

