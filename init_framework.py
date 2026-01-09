"""
Initialize tetrahedral AGI framework
"""

import os
import sys
import logging
from pathlib import Path

# Add the framework root to Python path
framework_root = Path(__file__).parent
sys.path.insert(0, str(framework_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = ['torch', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.warning(f"Missing dependencies: {', '.join(missing_modules)}")
        logger.info("Install with: pip install torch numpy")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'checkpoints',
        'models',
        'data',
        'logs',
        'outputs'
    ]
    
    for directory in directories:
        dir_path = framework_root / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def verify_framework_structure():
    """Verify framework structure"""
    required_components = [
        'geometry/tetrahedral_grid.py',
        'neural_network/tetrahedral_network.py',
        'training/trainer.py',
        'api/api_gateway.py',
        'applications/scientific_computing.py',
        'applications/manufacturing.py',
        'applications/autonomous_systems.py'
    ]
    
    missing_components = []
    for component in required_components:
        component_path = framework_root / component
        if not component_path.exists():
            missing_components.append(component)
    
    if missing_components:
        logger.error(f"Missing framework components: {missing_components}")
        return False
    
    logger.info("All framework components verified")
    return True

def test_core_imports():
    """Test core module imports"""
    try:
        from geometry.tetrahedral_grid import TetrahedralGrid
        from neural_network.tetrahedral_network import TetrahedralAGINetwork
        from applications.scientific_computing import ScientificComputingModule
        logger.info("✓ Core imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Core import failed: {e}")
        return False

def main():
    """Main initialization function"""
    logger.info("Initializing 64-Point Tetrahedron AI Framework...")
    logger.info(f"Framework path: {framework_root}")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed. Please install required packages.")
        return False
    
    # Create directories
    create_directories()
    
    # Verify structure
    if not verify_framework_structure():
        logger.error("Framework structure verification failed.")
        return False
    
    # Test imports
    if not test_core_imports():
        logger.error("Core import test failed.")
        return False
    
    # Test basic functionality
    try:
        from geometry.tetrahedral_grid import TetrahedralGrid
        grid = TetrahedralGrid(device='cpu')
        logger.info(f"✓ Tetrahedral grid created with {len(grid.points)} points")
        
        from neural_network.tetrahedral_network import TetrahedralAGINetwork
        model = TetrahedralAGINetwork(device='cpu')
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Neural network created with {param_count:,} parameters")
        
        logger.info("🎉 Framework initialization successful!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Framework functionality test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*80)
        print("64-POINT TETRAHEDRON AI FRAMEWORK INITIALIZED SUCCESSFULLY")
        print("="*80)
        print("Ready to use:")
        print("• Scientific Computing: Protein prediction, Material properties, Physics simulation")
        print("• Manufacturing: Defect detection, Process optimization, Predictive maintenance")
        print("• Autonomous Systems: 3D navigation, Object recognition, Robotics control")
        print("• API Gateway: RESTful API with WebSocket support")
        print("• Training Framework: Custom training and hyperparameter optimization")
        print("\nNext steps:")
        print("1. Run demo: python3 examples/demo.py")
        print("2. Start API: python3 -m api.api_gateway")
        print("3. View docs: Open README.md")
        print("="*80)
    else:
        print("\n❌ Framework initialization failed. Check logs for details.")
        sys.exit(1)