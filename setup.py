from setuptools import setup, find_packages
import os

setup(
    name="genhrl",
    version="0.1.0",
    description="GenHRL: Generative Hierarchical Reinforcement Learning Framework",
    long_description=open("README.md").read() if os.path.exists("README.md") else "GenHRL Framework",
    long_description_content_type="text/markdown",
    author="GenHRL Development Team",
    author_email="contact@genhrl.dev",
    url="https://github.com/your-org/genhrl",
    packages=find_packages(),
    
    # Core dependencies
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "anthropic>=0.3.0",
        "google-genai>=1.0.0",
        "openai>=1.14.0",
        "pydantic>=1.8.0",
        "psutil>=5.8.0",
        "pathlib",
        "json5",
        "gymnasium>=0.28.0",
        "scikit-learn>=1.0.0",
        "flask>=2.2.0",
        "flask-cors>=3.0.10",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "ui": [
            "flask>=2.0.0",
            "dash>=2.0.0",
            "plotly>=5.0.0",
        ],
        "isaaclab": [
            # IsaacLab should be installed separately
            # We'll check for it at runtime
        ],
    },
    
    # Entry points for both CLI and training
    entry_points={
        "console_scripts": [
            "genhrl=genhrl.cli:main",
        ],
    },
    
    # Package data
    package_data={
        "genhrl": [
            "prompts/*.py",
            "templates/*.py",
            "configs/*.yaml",
            "scripts/*.py",
            "scripts/*.sh",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
)