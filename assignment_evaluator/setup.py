from setuptools import setup, find_packages

setup(
    name='assignment_evaluator',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='An NLP-based assignment evaluator that assesses semantic similarity between student answers and model answers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/assignment_evaluator',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'spacy>=3.0.0',
        'nltk>=3.5'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)