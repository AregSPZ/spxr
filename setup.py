from setuptools import setup, find_packages

setup(
    name='spxr',
    version='0.1.9',
    author='Areg Sargsyan',
    author_email='quant0027@gmail.com',
    description='Custom functions for repetitive tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AregSP/spxr',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)