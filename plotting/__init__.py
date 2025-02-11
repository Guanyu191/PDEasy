'''
Descripttion: 
Author: Guanyu
Date: 2025-02-09 11:51:11
LastEditTime: 2025-02-09 11:56:08
'''
"""
__init__.py

This file is located within the plotting package. Its primary purpose is to 
centrally manage and export the core functions from each module in the package.
By importing these functions in this file, other modules can import multiple 
functions at once with a single line of code, enhancing code conciseness and maintainability.

Specific functions imported are as follows:
- plot_loss_from_logger: From the plot_loss.py module, 
    used to plot loss-related charts from the logger.
- plot_error_from_logger: From the plot_error.py module, 
    used to plot error-related charts from the logger.
- plot_solution_from_data: From the plot_solution.py module, 
    used to plot solution-related charts based on given data for 1D or 2Ds.

Usage:
In other modules, you can import these functions all at once in the following way:
from plotting import plot_loss_from_logger, plot_error_from_logger, plot_solution_from_data
"""

from .plot_loss import plot_loss_from_logger
from .plot_error import plot_error_from_logger
from .plot_solution import plot_solution_from_data