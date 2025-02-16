import numpy as np


def sample_on_line(p1, p2, n, method='grid'):
    """
    在两个点 (x1, y1) 和 (x2, y2) 的直线上随机采样 n_bcs 个点。
    
    参数:
        p1: 第一个坐标点
        p2: 第二个坐标点
        n: 需要采样的点数
        method: 采样方法，'random' 或 'grid'

    返回:
        sampled_points: 形状为 (n, 2) 的数组，每一行是采样点的 (x, y) 坐标
    """
    if method == 'grid':
        # 在 [0, 1] 范围内生成等间距的权重
        t = np.linspace(0, 1, n)
    elif method == 'random':
        # 在 [0, 1] 范围内生成随机权重
        t = np.random.rand(n)

    x1, y1 = p1
    x2, y2 = p2

    # 计算采样点的坐标
    sampled_x = (1 - t) * x1 + t * x2
    sampled_y = (1 - t) * y1 + t * y2
    
    # 合并坐标并返回
    sampled_points = np.stack([sampled_x, sampled_y], axis=1)
    return sampled_points

if __name__ == "__main__":
    # 测试示例
    p1 = (0, 0)
    p2 = (1, 1)
    n = 5
    sampled_points = sample_on_line(p1, p2, n, method='random')
    print(sampled_points)