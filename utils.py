import random


def random_color():
    """生成随机颜色

    Returns:
        list: RGB随机颜色值
    """
    return [random.randint(0, 255) for _ in range(3)]