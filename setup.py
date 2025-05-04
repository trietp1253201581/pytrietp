from setuptools import setup, find_packages

setup(
    name='pytrietp',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'numpy',
        'tiktoken'
    ],
    author='Le Minh Triet',
    author_email='trietlm0306@gmail.com',
    description='A package for utility functions',
    url='https://github.com/trietp1253201581/pytrietp',
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
