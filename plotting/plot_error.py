import matplotlib.pyplot as plt
import numpy as np
import os


def plot_error_from_logger(logger, figure_dir, show=True):
    plt.rcParams.update({'font.size':18})
    log = logger.log

    fig = plt.figure(figsize=(9, 7), dpi=64)
    ax = fig.subplots()

    error_keys = [key for key in log if key.startswith("error_")]

    for key in error_keys:
        label = r'$\mathcal{E}_{' + key.replace("error_", "") + '}$'
        ax.plot(log["iter"], log[key], label=label, linewidth=3)

    ax.set_yscale('log')
    ax.set_xticks(np.linspace(0, log["iter"][-1]+1, 5))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative L2 Error")
    ax.legend(loc='upper right')
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'error.png'), bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_error_from_data():
    pass