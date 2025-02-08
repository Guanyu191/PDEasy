'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 15:38:22
LastEditTime: 2025-02-08 15:55:48
'''
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_loss_from_logger(logger, figure_dir, show=True):
    log = logger.log

    fig = plt.figure(figsize=(9, 7), dpi=64)
    ax = fig.subplots()

    loss_keys = [key for key in log if key.startswith("loss_")]

    for key in loss_keys:
        label = r'$\mathcal{L}_{' + key.replace("loss_", "") + '}$'
        ax.plot(log["iter"], log[key], label=label, linewidth=3)

    ax.set_yscale('log')
    ax.set_xticks(np.linspace(0, log["iter"][-1]+1, 5))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc='upper right')
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'loss.png'), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_loss_from_data():
    pass