import numpy as np


def sample_on_line(p1, p2, n, method='grid'):
    r"""
    Randomly sample n_bcs points on the straight line between two points (x1, y1) and (x2, y2).

    Args:
        p1: The first coordinate point.
        p2: The second coordinate point.
        n: The number of points to be sampled.
        method: The sampling method, either 'random' or 'grid'.

    Returns:
        sampled_points: An array of shape (n, 2), where each row represents the (x, y) coordinates of a sampled point.
    """
    if method == 'grid':
        t = np.linspace(0, 1, n)
    elif method == 'random':
        t = np.random.rand(n)

    x1, y1 = p1
    x2, y2 = p2

    sampled_x = (1 - t) * x1 + t * x2
    sampled_y = (1 - t) * y1 + t * y2
    
    sampled_points = np.stack([sampled_x, sampled_y], axis=1)
    return sampled_points

if __name__ == "__main__":
    p1 = (0, 0)
    p2 = (1, 1)
    n = 5
    sampled_points = sample_on_line(p1, p2, n, method='random')
    print(sampled_points)