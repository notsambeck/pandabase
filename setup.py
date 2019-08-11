from setuptools import setup

setup(
    version='0.1',
    author='Sam Beck',
    author_email='notsambeck@gmail.com',
    name='pandabase',
    packages=['pandabase'],
    description="pandabase links pandas DataFrames to SQL databases. Supports read, append, and upsert.",
    url="https://github.com/notsambeck/pandabase",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
