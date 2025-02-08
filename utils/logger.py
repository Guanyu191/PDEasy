'''
Descripttion: 
Author: Guanyu
Date: 2025-02-08 14:57:49
LastEditTime: 2025-02-08 19:53:58
'''
import os
import time
import torch
import numpy as np

from utils.elapsed_time import elapsed_time


class Logger:
    def __init__(self, log_dir, log_keys=['iter', 'loss'], num_iters=1000, print_interval=100):
        # Directory to store logs
        self.log_dir = log_dir

        self.log_keys = log_keys
        self.log = {key: [] for key in self.log_keys}
        self.num_iters = num_iters
        self.print_interval = print_interval
        self.current_iter = 0
        self.start_time = time.time()
        self.best_loss = np.inf

    def record(self, **kwargs):
        """Record the training information for each iteration."""
        # Iterate through the provided key-value pairs
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            # If the key is already in the log, append the value
            if key in self.log:
                self.log[key].append(value)
            else:
                # If the key is not in the log, initialize it
                self.log[key] = [value]

        # Check for empty lists in the logs
        self._validate_logs()

        # Print the logs if the iteration is a multiple of the print interval
        if self.current_iter % self.print_interval == 0:
            self._print_logs()
            self.save()

        # Increment the iteration counter
        self.current_iter += 1

    def _print_logs(self):
        """Print the latest logs."""
        log_str = f'Iter # {self.log['iter'][-1]:4d}/{self.num_iters} '
        log_str += f'Time {time.time() - self.start_time:.1f}s\t'
        for key in self.log_keys[1:-1]:
            log_str += f"{key}: {self.log[key][-1]:.3e}, "
        log_str += f"{self.log_keys[-1]}: {self.log[self.log_keys[-1]][-1]:.3e}"
        print(log_str)

    def save(self):
        """Save the logs to a file."""
        np.save(os.path.join(self.log_dir, 'log.npy'), self.log)

    def load(self):
        """Load the logs from a file."""
        self.log = np.load(os.path.join(self.log_dir, 'log.npy'), allow_pickle=True).item()

    def elapsed_time(self):
        """Calculate and print the elapsed time since the start."""
        hours, minutes, seconds = elapsed_time(self.start_time, time.time())
        print(f"Elapsed time: {hours:.0f}h {minutes:.0f}m {seconds:.0f}s")

    def _validate_logs(self):
        """Ensure there are no empty lists in the logs."""
        for key, value in self.log.items():
            if isinstance(value, list) and not value:
                raise ValueError(f"Empty list found for key: {key}")
