from setuptools import setup

with open("README.md", "r") as readme:
    long_description = readme.read()

setup(
    version='0.4.5',
    author='Sam Beck',
    author_email='notsambeck@gmail.com',
    name='pandabase',
    packages=['pandabase'],
    description="pandabase links pandas DataFrames to SQL databases. Upsert, append, read, drop, describe...",
    long_description=long_description,
    long_description_content_type = 'text/markdown',
    url="https://github.com/notsambeck/pandabase",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    install_requires=[
        'pandas>=0.24.0',
        'sqlalchemy>=1.3.0',
        ],
    python_requires='>=3.6',
)
