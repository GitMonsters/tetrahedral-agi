# Installation and Setup Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)

## Quick Installation

1. **Clone the repository:**
```bash
git clone https://github.com/tetrahedral-ai/tetrahedral-agi.git
cd tetrahedral-agi
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
```

## Basic Usage

### 1. Core Framework Test
```python
import sys
sys.path.append('.')
from geometry.tetrahedral_grid import TetrahedralGrid
from neural_network.tetrahedral_network import TetrahedralAGINetwork

# Create tetrahedral grid
grid = TetrahedralGrid(device='cpu')
print(f"Grid created with {len(grid.points)} points")

# Create model
model = TetrahedralAGINetwork(device='cpu')
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### 2. Run Demo
```bash
python3 examples/demo.py
```

### 3. Start API Server
```bash
python3 -m api.api_gateway
```

## Dependencies

Required packages are listed in `requirements.txt`. For full functionality, install additional packages:

```bash
# For scientific computing
pip install biopython pymatgen ase

# For manufacturing applications
pip install opencv-python scikit-learn open3d

# For training and optimization
pip install wandb optuna
```

## Docker Installation

```bash
# Build Docker image
docker build -t tetrahedral-agi .

# Run container
docker run --gpus all -p 8000:8000 tetrahedral-agi
```

## Troubleshooting

### Common Issues

1. **CUDA errors:**
   - Ensure NVIDIA drivers are up to date
   - Check PyTorch CUDA compatibility
   - Use CPU mode if GPU unavailable

2. **Memory issues:**
   - Reduce batch size in training
   - Use mixed precision training
   - Ensure sufficient RAM/VRAM

3. **Import errors:**
   - Verify all dependencies installed
   - Check Python version compatibility
   - Ensure proper PYTHONPATH

### Getting Help

- Check the documentation: `docs/`
- Review examples: `examples/`
- Open an issue on GitHub
- Join our Discord community