

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='ivtools',
    version='0.1',
    author='Benjamin H. Ellis',
    description='Tools for processing and analyzing current-voltage (IV) data.',
    packages=['ivtools'],
    install_requires=['pvlib', 'cufflinks', 'pandas', 'numpy', 'scipy',
                      'seaborn', 'matplotlib', 'plotly', 'scikit_learn'],
)

