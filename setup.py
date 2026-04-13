from setuptools import setup, find_packages

setup(
    name="lotto_ai",
    version="3.0.0",
    description="Loto Serbia Smart Portfolio Manager - Coverage optimization with mathematical guarantees",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
        'sqlalchemy>=2.0.0',
        'streamlit>=1.28.0',
        'requests>=2.31.0',
        'beautifulsoup4>=4.12.0',
        'PyPDF2>=3.0.0',
        'python-dateutil>=2.8.2',
    ]
)