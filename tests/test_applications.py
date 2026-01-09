"""
Test application modules
"""

import pytest
import torch
import numpy as np
from applications.scientific_computing import ScientificComputingModule
from applications.manufacturing import ManufacturingModule
from applications.autonomous_systems import AutonomousSystemsModule


class TestScientificComputingModule:
    """Test cases for ScientificComputingModule"""
    
    def test_module_creation(self):
        """Test module initialization"""
        module = ScientificComputingModule(device='cpu')
        assert module.device == 'cpu'
        assert module.grid is not None
        assert module.protein_predictor is not None
        assert module.material_predictor is not None
        assert module.physics_simulator is not None
    
    def test_protein_prediction(self):
        """Test protein structure prediction"""
        module = ScientificComputingModule(device='cpu')
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        
        result = module.predict_protein_structure(sequence)
        
        assert 'coordinates' in result
        assert 'confidence_scores' in result
        assert 'mean_confidence' in result
        assert len(result['coordinates']) == 64
        assert 0 <= result['mean_confidence'] <= 1
    
    def test_material_prediction(self):
        """Test material property prediction"""
        module = ScientificComputingModule(device='cpu')
        crystal_data = {
            'lattice_vectors': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            'atomic_positions': [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            'atom_types': ['Si', 'Si'],
            'space_group': 227
        }
        
        result = module.predict_material_properties(crystal_data)
        
        assert 'band_gap' in result
        assert 'elastic_modulus' in result
        assert 'thermal_conductivity' in result
        assert all(isinstance(v, (int, float)) for v in result.values())


class TestManufacturingModule:
    """Test cases for ManufacturingModule"""
    
    def test_module_creation(self):
        """Test module initialization"""
        module = ManufacturingModule(device='cpu')
        assert module.device == 'cpu'
        assert module.quality_detector is not None
        assert module.process_optimizer is not None
        assert module.maintenance_predictor is not None
    
    def test_defect_detection(self):
        """Test defect detection"""
        module = ManufacturingModule(device='cpu')
        point_cloud_data = {
            'points': np.random.randn(50, 3).tolist(),
            'normals': np.random.randn(50, 3).tolist(),
            'colors': np.random.rand(50, 3).tolist()
        }
        
        defects = module.detect_defects(point_cloud_data)
        
        assert isinstance(defects, list)
        assert all(hasattr(d, 'defect_type') for d in defects)
        assert all(hasattr(d, 'confidence') for d in defects)
        assert all(hasattr(d, 'severity') for d in defects)
    
    def test_process_optimization(self):
        """Test process optimization"""
        module = ManufacturingModule(device='cpu')
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
        
        result = module.optimize_process(current_params, quality_metrics)
        
        assert 'optimized_parameters' in result
        assert 'expected_improvement' in result
        assert 'adjustments' in result
        assert isinstance(result['expected_improvement'], (int, float))


class TestAutonomousSystemsModule:
    """Test cases for AutonomousSystemsModule"""
    
    def test_module_creation(self):
        """Test module initialization"""
        module = AutonomousSystemsModule(device='cpu')
        assert module.device == 'cpu'
        assert module.navigation_network is not None
        assert module.object_recognition is not None
        assert module.robotics_controller is not None
    
    def test_spatial_navigation(self):
        """Test spatial navigation"""
        module = AutonomousSystemsModule(device='cpu')
        current_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'velocity': [0.1, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.1],
            'goal_position': [5.0, 5.0, 2.0]
        }
        obstacles = [[2.0, 1.0, 0.0], [3.0, 3.0, 1.0]]
        
        result = module.navigate_to_goal(current_state, obstacles)
        
        assert 'direction' in result
        assert 'velocity_vector' in result
        assert 'velocity_magnitude' in result
        assert 'collision_probability' in result
        assert 'is_safe' in result
        assert isinstance(result['is_safe'], bool)
    
    def test_object_recognition(self):
        """Test object recognition"""
        module = AutonomousSystemsModule(device='cpu')
        point_cloud = np.random.randn(50, 3).tolist()
        
        result = module.recognize_objects(point_cloud)
        
        assert hasattr(result, 'objects')
        assert hasattr(result, 'scene_understanding')
        assert hasattr(result, 'confidence_scores')
        assert 'num_objects' in result.scene_understanding
        assert 'scene_complexity' in result.scene_understanding


if __name__ == "__main__":
    pytest.main([__file__])