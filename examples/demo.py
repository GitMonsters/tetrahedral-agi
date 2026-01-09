"""
Example usage and demonstration of the 64-Point Tetrahedron AI Framework
"""

import torch
import numpy as np
import time
from pathlib import Path

# Import the framework
from tetrahedral_agi import (
    TetrahedralAGINetwork,
    ScientificComputingModule,
    ManufacturingModule,
    AutonomousSystemsModule,
    get_system_info
)


def main():
    """Main demonstration function"""
    print("=" * 80)
    print("64-Point Tetrahedron AI Framework - Demonstration")
    print("=" * 80)
    
    # Check system information
    system_info = get_system_info()
    print(f"\nSystem Information:")
    print(f"  PyTorch Version: {system_info['pytorch_version']}")
    print(f"  CUDA Available: {system_info['cuda_available']}")
    print(f"  Device: {system_info['device_name']}")
    print(f"  Framework Version: {system_info['tetrahedral_agi_version']}")
    
    device = 'cuda' if system_info['cuda_available'] else 'cpu'
    print(f"  Using device: {device}")
    
    # Demonstrate core model
    print("\n" + "=" * 60)
    print("Core Model Demonstration")
    print("=" * 60)
    
    model = TetrahedralAGINetwork(device=device)
    model.to(device)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 64, device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        output = model(dummy_input)
    inference_time = time.time() - start_time
    
    print(f"Output shape: {output.shape}")
    print(f"Inference time: {inference_time*1000:.2f} ms")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get grid information
    grid_info = model.get_grid_info()
    print(f"\nTetrahedral Grid Information:")
    print(f"  Number of points: {grid_info['num_points']}")
    print(f"  Number of tetrahedra: {grid_info['num_tetrahedra']}")
    print(f"  Number of cavities: {grid_info['num_cavities']}")
    print(f"  Grid volume: {grid_info['volume']:.4f}")
    print(f"  Mean radius: {grid_info['mean_radius']:.4f}")
    
    # Demonstrate Scientific Computing Module
    print("\n" + "=" * 60)
    print("Scientific Computing Module")
    print("=" * 60)
    
    sci_module = ScientificComputingModule(device=device)
    
    # Protein structure prediction
    print("\n1. Protein Structure Prediction:")
    sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"
    protein_results = sci_module.predict_protein_structure(sequence)
    
    print(f"   Sequence length: {len(sequence)}")
    print(f"   Predicted points: {len(protein_results['coordinates'])}")
    print(f"   Mean confidence: {protein_results['mean_confidence']:.3f}")
    print(f"   Processing time: {time.time():.2f}s")
    
    # Material property prediction
    print("\n2. Material Property Prediction:")
    crystal_data = {
        'lattice_vectors': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        'atomic_positions': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        'atom_types': ['Si', 'Si'],
        'space_group': 227
    }
    
    material_results = sci_module.predict_material_properties(crystal_data)
    print("   Predicted properties:")
    for prop, value in material_results.items():
        print(f"     {prop}: {value:.3f}")
    
    # Physics simulation
    print("\n3. Physics Simulation:")
    physics_state = {
        'positions': [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        'velocities': [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    }
    
    physics_results = sci_module.simulate_physics(physics_state, num_steps=5)
    print(f"   Simulation steps: {len(physics_results)}")
    print(f"   Final energy: {physics_results[-1]['energy']:.3f}")
    print(f"   Final positions: {len(physics_results[-1]['positions'])} particles")
    
    # Demonstrate Manufacturing Module
    print("\n" + "=" * 60)
    print("Manufacturing Module")
    print("=" * 60)
    
    mfg_module = ManufacturingModule(device=device)
    
    # Defect detection
    print("\n1. Defect Detection:")
    point_cloud_data = {
        'points': np.random.randn(100, 3).tolist(),
        'normals': np.random.randn(100, 3).tolist(),
        'colors': np.random.rand(100, 3).tolist()
    }
    
    defects = mfg_module.detect_defects(point_cloud_data)
    print(f"   Point cloud size: {len(point_cloud_data['points'])} points")
    print(f"   Defects detected: {len(defects)}")
    for i, defect in enumerate(defects[:3]):
        print(f"     {i+1}. {defect.defect_type} (confidence: {defect.confidence:.3f})")
    
    # Process optimization
    print("\n2. Process Optimization:")
    current_params = {
        'temperature': 200.0,
        'pressure': 50.0,
        'speed': 100.0,
        'material_flow': 25.0,
        'vibration': 0.1
    }
    
    quality_metrics = {
        'dimensional_accuracy': 0.95,
        'surface_finish': 0.88,
        'material_density': 0.92,
        'tensile_strength': 0.90,
        'hardness': 0.87,
        'fatigue_resistance': 0.85,
        'corrosion_resistance': 0.89,
        'thermal_stability': 0.91,
        'electrical_conductivity': 0.86,
        'optical_clarity': 0.84
    }
    
    optimization_results = mfg_module.optimize_process(current_params, quality_metrics)
    print(f"   Expected improvement: {optimization_results['expected_improvement']:.3f}")
    print(f"   New temperature: {optimization_results['optimized_parameters']['temperature']:.1f}°C")
    print(f"   New pressure: {optimization_results['optimized_parameters']['pressure']:.1f} bar")
    
    # Predictive maintenance
    print("\n3. Predictive Maintenance:")
    sensor_data = np.random.randn(50, 20).tolist()
    maintenance_results = mfg_module.predict_maintenance(sensor_data)
    
    print(f"   Failure probability: {maintenance_results['failure_probability']:.3f}")
    print(f"   Remaining useful life: {maintenance_results['remaining_useful_life_hours']:.1f} hours")
    print(f"   Recommended action: {maintenance_results['recommended_action']}")
    print(f"   Action confidence: {maintenance_results['action_confidence']:.3f}")
    
    # Demonstrate Autonomous Systems Module
    print("\n" + "=" * 60)
    print("Autonomous Systems Module")
    print("=" * 60)
    
    auto_module = AutonomousSystemsModule(device=device)
    
    # Spatial navigation
    print("\n1. Spatial Navigation:")
    current_state = {
        'position': [0.0, 0.0, 0.0],
        'orientation': [1.0, 0.0, 0.0, 0.0],
        'velocity': [0.1, 0.0, 0.0],
        'angular_velocity': [0.0, 0.0, 0.1],
        'goal_position': [5.0, 5.0, 2.0]
    }
    
    obstacles = [[2.0, 1.0, 0.0], [3.0, 3.0, 1.0], [1.0, 4.0, 0.5]]
    navigation_results = auto_module.navigate_to_goal(current_state, obstacles)
    
    print(f"   Navigation direction: [{navigation_results['direction'][0]:.3f}, {navigation_results['direction'][1]:.3f}, {navigation_results['direction'][2]:.3f}]")
    print(f"   Velocity magnitude: {navigation_results['velocity_magnitude']:.3f}")
    print(f"   Collision probability: {navigation_results['collision_probability']:.3f}")
    print(f"   Is safe: {navigation_results['is_safe']}")
    
    # Object recognition
    print("\n2. Object Recognition:")
    point_cloud = np.random.randn(100, 3).tolist()
    perception_results = auto_module.recognize_objects(point_cloud)
    
    print(f"   Objects detected: {len(perception_results.objects)}")
    print(f"   Scene complexity: {perception_results.scene_understanding['scene_complexity']:.3f}")
    print(f"   Dominant object: {perception_results.scene_understanding['dominant_object']}")
    
    # Robotics control
    print("\n3. Robotics Control:")
    current_joints = {
        'positions': [0.0, 0.5, -0.3, 0.2, 0.0, 0.1, -0.1],
        'velocities': [0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0]
    }
    
    target_pose = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    sensor_data = np.random.randn(20).tolist()
    
    control_results = auto_module.control_robot(current_joints, target_pose, sensor_data)
    print(f"   Stability score: {control_results['stability_score']:.3f}")
    print(f"   Is stable: {control_results['is_stable']}")
    print(f"   Joint commands generated: {len(control_results['joint_position_commands'])}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Memory usage
    if device == 'cuda':
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"  GPU memory allocated: {memory_allocated:.1f} MB")
        print(f"  GPU memory reserved: {memory_reserved:.1f} MB")
    
    print(f"\nModule Capabilities:")
    print(f"  ✓ Scientific Computing: Protein prediction, Material properties, Physics simulation")
    print(f"  ✓ Manufacturing: Defect detection, Process optimization, Predictive maintenance")
    print(f"  ✓ Autonomous Systems: 3D navigation, Object recognition, Robotics control")
    print(f"  ✓ API Gateway: RESTful API, WebSocket support, Model management")
    print(f"  ✓ Training Framework: Custom training, Hyperparameter optimization")
    
    print("\n" + "=" * 80)
    print("Demonstration completed successfully!")
    print("The 64-Point Tetrahedron AI Framework is ready for use.")
    print("=" * 80)


if __name__ == "__main__":
    main()