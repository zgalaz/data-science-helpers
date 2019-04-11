from setuptools import setup

setup(
    name='Data Science Helpers',
    version=0.1,
    description='Useful data-science helpers (visualizations, utility functions, etc.',
    long_description=open('README.md').read(),
    author='Zoltán Galáž',
    author_email='xgalaz00@gmail.com',
    packages=['helpers'],
    license='MIT',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'scipy'
    ]
)
