import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt


def u(x, y, alpha=-10, x_0=0.5, y_0=0.5):
    return np.exp(alpha * ((x - x_0)**2 + (y - y_0)**2))

if __name__ == '__main__':
    x = np.linspace(-1, 1, 201)
    y = np.linspace(-1, 1, 201)
    xx, yy = np.meshgrid(x, y)
    u = u(xx, yy)

    plt.figure(figsize=(10, 6))
    plt.pcolor(xx, yy, u)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution u(x, y)')
    plt.show()

    if not os.path.exists('./data'):
        os.makedirs('./data')
    io.savemat('./data/Poisson_Sol.mat', {'u': u, 'x': x, 'y': y})
