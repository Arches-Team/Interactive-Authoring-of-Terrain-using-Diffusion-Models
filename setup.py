from setuptools import find_packages, setup

setup(
    name="terrain_diffusion",
    version="1.0",
    install_requires=[],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        # Core
        "all": ["diffusers[torch]", "torch", "torchvision", "numpy", "Pillow", "tqdm", "scikit-learn"],
        
        # Terrain analysis
        "dev": ["pysheds", "pyproj", "affine", "opencv-python"],

        # Visualization
        "viz": ["PyOpenGL", "PyOpenGL_accelerate", "pyrr", "pygame", "matplotlib", "SciencePlots"],

        # Evaluation
        "eval": ["pytorch-fid", "lpips", "pytorch-msssim"],
    },
)