from setuptools import setup, find_packages
import os

setup(
    name='pdeasy',
    version='0.1.3',
    packages=find_packages(),
    author='Guanyu Pan',
    author_email='guanyu191@163.com',
    description='Lightweight PINN & Operator PDE Solver for Research, Balancing Abstraction and Flexibility for Algorithm Innovation.',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/Guanyu191/PDEasy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'pdeasy-cli = pdeasy.cli:main',
        ],
    },
)