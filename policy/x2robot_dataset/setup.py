from setuptools import setup, find_packages

setup(
    name='x2robot_dataset',  
    version='0.1.0', 
    author='Hao Wang',  
    author_email='wanghao@x2robot.com',  
    description='x2dataset', 
    packages=find_packages(),  
    python_requires='>=3.6', 
    install_requires=[  
        'numpy',
        'imagecodecs-numcodecs',
    ],
)