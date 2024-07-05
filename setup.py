from setuptools import setup, find_packages

setup(
    name='diffshortcut',
    version='0.1',
    packages=find_packages(),
    description='A library to automatically remove noise from protected images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='',
    license='MIT',
    install_requires=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)
