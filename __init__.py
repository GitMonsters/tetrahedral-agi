"""
64-Point Tetrahedron AI Framework
A revolutionary geometric deep learning platform for 3D spatial intelligence

This package implements the complete 64-point tetrahedron AI architecture as described
in the Product Requirements Document, providing breakthrough capabilities in:
- 3D data processing and spatial reasoning
- Molecular modeling and scientific computing
- Manufacturing quality control and optimization
- Autonomous systems and robotics control

Core Components:
- TetrahedralGrid: 64-point geometric structure
- TetrahedralAGINetwork: Main neural network architecture
- Specialized application modules
- Training and optimization framework
- API gateway for deployment

Example Usage:
    from tetrahedral_agi import TetrahedralAGINetwork, ScientificComputingModule
    
    # Create model
    model = TetrahedralAGINetwork(device='cuda')
    
    # Use scientific computing module
    sci_module = ScientificComputingModule(device='cuda')
    protein_structure = sci_module.predict_protein_structure("ACDEFGHIKLMNPQRSTVWY")
"""

__version__ = "1.0.0"
__author__ = "Tetrahedral AI Team"
__email__ = "contact@tetrahedral-ai.com"

# Core imports
from .geometry.tetrahedral_grid import TetrahedralGrid, TetrahedronGeometry, GeometricOperations
from .neural_network.tetrahedral_network import (
    TetrahedralAGINetwork,
    TetrahedralConvolution,
    OctahedralCavityProcessor,
    TetrahedralMessagePassing,
    SpatialAttentionMechanism
)
from .training.trainer import (
    TetrahedralTrainer,
    TrainingConfig,
    GeometricDataset,
    GeometricLoss,
    HyperparameterOptimizer
)

# Application modules
from .applications.scientific_computing import ScientificComputingModule
from .applications.manufacturing import ManufacturingModule
from .applications.autonomous_systems import AutonomousSystemsModule

# API components
from .api.api_gateway import app, model_manager, training_manager

# Utility functions
def create_model(device: str = 'cuda', **kwargs) -> TetrahedralAGINetwork:
    """Create a tetrahedral AI model with default configuration"""
    return TetrahedralAGINetwork(device=device, **kwargs)

def get_system_info() -> dict:
    """Get system information and capabilities"""
    import torch
    return {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
        'tetrahedral_agi_version': __version__
    }

# Export main classes
__all__ = [
    # Core classes
    'TetrahedralGrid',
    'TetrahedralAGINetwork',
    'TetrahedralTrainer',
    'TrainingConfig',
    
    # Application modules
    'ScientificComputingModule',
    'ManufacturingModule', 
    'AutonomousSystemsModule',
    
    # API
    'app',
    'model_manager',
    'training_manager',
    
    # Utilities
    'create_model',
    'get_system_info'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'tetrahedral-agi',
    'version': __version__,
    'description': '64-Point Tetrahedron AI Framework for Geometric Deep Learning',
    'author': __author__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'install_requires': [
        'torch>=1.12.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'pydantic>=1.8.0',
        'wandb>=0.12.0',
        'optuna>=3.0.0',
        'opencv-python>=4.5.0',
        'open3d>=0.13.0',
        'biopython>=1.79',
        'pymatgen>=2022.0.0',
        'ase>=3.22.0'
    ]
}

def _check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import scipy
    except ImportError:
        missing_deps.append('scipy')
    
    if missing_deps:
        print(f"Warning: Missing dependencies: {', '.join(missing_deps)}")
        print("Please install them using: pip install tetrahedral-agi[all]")
    
    return len(missing_deps) == 0

# Run dependency check on import
_dependencies_ok = _check_dependencies()

if not _dependencies_ok:
    print("Some dependencies are missing. The framework may not work correctly.")
    print("Run: pip install -e .[all] to install all dependencies.")