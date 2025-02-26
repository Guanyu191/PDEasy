import matplotlib.pyplot as plt
import numpy as np
import os


def plot_solution_from_data(figure_dir, **kwargs):
    plt.rcParams.update({'font.size':18})

    required_keys = ['x_grid','sol','sol_pred']
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(f"Missing required parameter: {key}")
        
    shapes = [kwargs[key].shape for key in required_keys]
    first_shape = shapes[0]
    for i, shape in enumerate(shapes[1:], start=1):
        if shape != first_shape:
            raise ValueError(f"The shape of {required_keys[0]} and {required_keys[i]} are different. {required_keys[0]} shape: {first_shape}, {required_keys[i]} shape: {shape}")


    dimension = kwargs['sol'].ndim
    if dimension == 1:
        _plot_solution_from_data_1D(figure_dir, **kwargs)
    elif dimension == 2:
        _plot_solution_from_data_2D(figure_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


def _plot_solution_from_data_2D(figure_dir, **kwargs):

    required_params = ['x_grid', 'y_grid', 'sol', 'sol_pred']
    for param in required_params:
        if param not in kwargs:
            raise ValueError(f"Missing required parameter: {param}")
        
    xx = kwargs['x_grid']
    yy = kwargs['y_grid']
    sol = kwargs['sol']
    sol_pred = kwargs['sol_pred']

    cmap = kwargs.get('cmap', 'jet')
    sol_min = kwargs.get('sol_min', sol.min())
    sol_max = kwargs.get('sol_max', sol.max())

    x_min = kwargs.get('x_min', xx.min())
    x_max = kwargs.get('x_max', xx.max())
    y_min = kwargs.get('y_min', yy.min())
    y_max = kwargs.get('y_max', yy.max())

    x_label = kwargs.get('x_label', '$x$')
    y_label = kwargs.get('y_label', '$y$')

    title_left = kwargs.get('title_left', r'Reference $u(x,y)$')
    title_middle = kwargs.get('title_middle', r'Predicted $u(x,y)$')
    title_right = kwargs.get('title_right', r'Absolute error')

    x_ticks = kwargs.get('x_ticks', np.linspace(x_min, x_max, 5))
    y_ticks = kwargs.get('y_ticks', np.linspace(y_min, y_max, 5))

    figure_name = kwargs.get('figure_name', 'Sol_PINN.png')
    dpi = kwargs.get('dpi', 64)
    show = kwargs.get('show', True)


    fig = plt.figure(figsize=(20, 5))
    axes = fig.subplots(1, 3)

    ax = axes[0]
    cax = ax.pcolor(xx, yy, sol, cmap=cmap, vmin=sol_min, vmax=sol_max)
    cbar = fig.colorbar(cax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_left)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect(1./ax.get_data_ratio())

    ax = axes[1]
    cax = ax.pcolor(xx, yy, sol_pred, cmap=cmap, vmin=sol_min, vmax=sol_max)
    cbar = fig.colorbar(cax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_middle)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect(1./ax.get_data_ratio())

    ax = axes[2]
    cax = ax.pcolor(xx, yy, abs(sol - sol_pred), cmap=cmap)
    cbar = fig.colorbar(cax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_right)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect(1./ax.get_data_ratio())

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, figure_name), dpi=dpi, bbox_inches='tight')

    plt.show() if show else plt.close()


def _plot_solution_from_data_1D(figure_dir, **kwargs):

    required_params = ['x_grid', 'sol', 'sol_pred']
    for param in required_params:
        if param not in kwargs:
            raise ValueError(f"Missing required parameter: {param}")
        
    x = kwargs['x_grid']
    sol = kwargs['sol']
    sol_pred = kwargs['sol_pred']

    sol_min = kwargs.get('sol_min', sol.min())
    sol_max = kwargs.get('sol_max', sol.max())

    x_min = kwargs.get('x_min', x.min())
    x_max = kwargs.get('x_max', x.max())

    x_label = kwargs.get('x_label', '$x$')
    y_label = kwargs.get('y_label', '$u$')

    title_left = kwargs.get('title_left', r'Solution $u(x)$')
    title_right = kwargs.get('title_right', r'Absolute error')

    x_ticks = kwargs.get('x_ticks', np.linspace(x_min, x_max, 5))
    y_ticks = kwargs.get('y_ticks', np.linspace(sol_min, sol_max, 5))

    figure_name = kwargs.get('figure_name', 'Sol_PINN.png')
    dpi = kwargs.get('dpi', 64)
    show = kwargs.get('show', True)

    e = (sol_max - sol_min) * 0.05


    fig = plt.figure(figsize=(12, 5))
    axes = fig.subplots(1, 2)

    ax = axes[0]
    ax.plot(x, sol_pred, 'b-', linewidth=2, label='Predicted')
    ax.plot(x, sol, 'r--', linewidth=2, label='Reference')
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_left)
    ax.set_ylim(sol_min-e, sol_max+e)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.grid()
    ax.legend(loc='upper right', fontsize=14)

    ax = axes[1]
    ax.plot(x, abs(sol - sol_pred), 'orange', linewidth=2)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title_right)
    ax.set_xticks(x_ticks)
    ax.set_aspect(1./ax.get_data_ratio())
    ax.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, figure_name), dpi=dpi, bbox_inches='tight')
    plt.show() if show else plt.close()


# def _plot_solution_from_data_1D_2_row_1_col(figure_dir, **kwargs):

#     required_params = ['x_grid', 'sol', 'sol_pred']
#     for param in required_params:
#         if param not in kwargs:
#             raise ValueError(f"Missing required parameter: {param}")
        
#     x = kwargs['x_grid']
#     sol = kwargs['sol']
#     sol_pred = kwargs['sol_pred']

#     sol_min = kwargs.get('sol_min', sol.min())
#     sol_max = kwargs.get('sol_max', sol.max())

#     x_min = kwargs.get('x_min', x.min())
#     x_max = kwargs.get('x_max', x.max())

#     x_label = kwargs.get('x_label', '$x$')
#     y_label = kwargs.get('y_label', '$u$')

#     title_left = kwargs.get('title_left', r'Solution $u(x)$')
#     title_right = kwargs.get('title_right', r'Absolute error')

#     x_ticks = kwargs.get('x_ticks', np.linspace(x_min, x_max, 5))
#     y_ticks = kwargs.get('y_ticks', np.linspace(sol_min, sol_max, 5))

#     figure_name = kwargs.get('figure_name', 'Sol_PINN.png')
#     dpi = kwargs.get('dpi', 64)
#     show = kwargs.get('show', True)

#     e = (sol_max - sol_min) * 0.05


#     fig = plt.figure(figsize=(12, 8))
#     axes = fig.subplots(2, 1)

#     ax = axes[0]
#     ax.plot(x, sol_pred, 'b-', linewidth=2, label='Predicted')
#     ax.plot(x, sol, 'r--', linewidth=2, label='Reference')
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_title(title_left)
#     ax.set_ylim(sol_min-e, sol_max+e)
#     ax.set_xticks(x_ticks)
#     ax.set_yticks(y_ticks)
#     ax.grid()
#     ax.legend(loc='upper right', fontsize=14)

#     ax = axes[1]
#     ax.plot(x, abs(sol - sol_pred), 'orange', linewidth=2)
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(y_label)
#     ax.set_title(title_right)
#     ax.set_xticks(x_ticks)
#     ax.grid()

#     plt.tight_layout()
#     plt.savefig(os.path.join(figure_dir, figure_name), dpi=dpi, bbox_inches='tight')
#     plt.show() if show else plt.close()
