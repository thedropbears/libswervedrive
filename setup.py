from setuptools import setup, find_packages

setup(name='swervedrive',
      version='0.1',
      description='FRC Team 4774\'s Swerve Drive library',
      url='http://github.com/thedropbears/libswervedrive',
      author='FRC Team 4774',
      author_email='enquiries@thedropbears.org.au',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
      ])
