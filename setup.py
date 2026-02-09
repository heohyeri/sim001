from setuptools import setup, find_packages

setup(
    name='ir_sim',
    packages=find_packages(),
    version= '2.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pyyaml',
        'pynput',
        'imageio',
        'pathlib'
    ],
    description="A simple 2D simulator for the intelligent mobile robots",
    author="Han Ruihua",
)
