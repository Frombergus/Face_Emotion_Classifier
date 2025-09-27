from setuptools import setup, find_packages

setup(
    name="emotion-detection-portfolio",
    version="1.0.0",  # Final v1.0 for portfolio release (Sep 27, 2025)
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.15.0",
        "opencv-python==4.8.0",
        "numpy==1.24.0",
        "matplotlib==3.7.0",
        "pandas==2.0.0",
        "seaborn==0.12.0",
        "keras-tuner==1.4.0",
        "pytest==7.4.0",
        "PyYAML==6.0",
        "jupyter==1.0.0"
    ],
    author="Christofer Fromberg",
    author_email="your.email@example.com",  # Update with yours
    description="Real-time facial emotion recognition using CNN (TensorFlow) & OpenCV – Portfolio project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emotion-detection-portfolio",  # Update post-upload
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "emotion-app=src.emotion_classifier:main",
        ],
    },
    python_requires='>=3.8',
)
# Usage: pip install -e . then emotion-app --video path.mp4