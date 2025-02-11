import os

def init_dir(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)
