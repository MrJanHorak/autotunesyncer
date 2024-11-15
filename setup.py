from setuptools import setup, find_packages

setup(
    name="autotunesyncer",
    version="0.1",
    packages=find_packages(),  # Include all packages
    install_requires=[
        'moviepy',
        'numpy',
        'opencv-python',
        'ffmpeg-python',
    ],
    python_requires='>=3.8',
)
