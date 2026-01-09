"""
Manufacturing Module for 64-Point Tetrahedron AI
Specialized applications for quality control, process optimization, and predictive maintenance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import cv2
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import open3d as o3d

from ..neural_network.tetrahedral_network import TetrahedralAGINetwork
from ..geometry.tetrahedral_grid import TetrahedralGrid


@dataclass
class ManufacturingDefect:
    """Represents a manufacturing defect"""
    defect_type: str
    location: Tuple[float, float, float]
    severity: float
    confidence: float
    dimensions: Dict[str, float]


@dataclass
class ProcessParameters:
    """Manufacturing process parameters"""
    temperature: float
    pressure: float
    speed: float
    material_flow: float
    vibration: float
    timestamp: float


class QualityControlDetector(nn.Module):
    """Detects defects in manufactured parts using tetrahedral AI"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256):
        super(QualityControlDetector, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        
        # 3D point cloud encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Surface normal encoder
        self.normal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Color/texture encoder
        self.texture_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # RGB
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Tetrahedral defect detection network
        self.defect_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 3,  # point + normal + texture
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Defect classification heads
        self.defect_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, len(['scratch', 'dent', 'crack', 'hole', 'deformation']))
        )
        
        # Severity predictor
        self.severity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, point_cloud: torch.Tensor, 
                normals: Optional[torch.Tensor] = None,
                colors: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Detect defects in 3D point cloud
        Args:
            point_cloud: 3D points [batch_size, num_points, 3]
            normals: Surface normals [batch_size, num_points, 3]
            colors: RGB colors [batch_size, num_points, 3]
        Returns:
            Dictionary with defect predictions
        """
        batch_size, num_points, _ = point_cloud.shape
        
        # Encode point cloud
        point_features = self.point_encoder(point_cloud)  # [batch_size, num_points, hidden_dim]
        
        # Encode normals (compute if not provided)
        if normals is None:
            normals = self._compute_normals(point_cloud)
        normal_features = self.normal_encoder(normals)  # [batch_size, num_points, hidden_dim]
        
        # Encode colors (use default if not provided)
        if colors is None:
            colors = torch.ones_like(point_cloud)
        texture_features = self.texture_encoder(colors)  # [batch_size, num_points, hidden_dim]
        
        # Combine features
        combined_features = torch.cat([point_features, normal_features, texture_features], dim=-1)
        
        # Map to tetrahedral grid
        grid_features = self._map_point_cloud_to_grid(combined_features, point_cloud)
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [batch_size, hidden_dim*3, 64]
        
        # Process through tetrahedral network
        processed_features = self.defect_network(grid_features)  # [batch_size, hidden_dim, 64]
        
        # Classify defects
        defect_features = processed_features.transpose(1, 2)  # [batch_size, 64, hidden_dim]
        defect_logits = self.defect_classifier(defect_features)  # [batch_size, 64, num_defect_types]
        defect_probs = torch.softmax(defect_logits, dim=-1)
        
        # Predict severity
        severity = self.severity_predictor(defect_features)  # [batch_size, 64, 1]
        
        return {
            'defect_probabilities': defect_probs,
            'severity': severity.squeeze(-1),
            'processed_features': processed_features,
            'grid_features': grid_features
        }
    
    def _compute_normals(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Compute surface normals from point cloud"""
        batch_size, num_points, _ = point_cloud.shape
        normals = torch.zeros_like(point_cloud)
        
        for b in range(batch_size):
            for i in range(num_points):
                # Find nearest neighbors
                distances = torch.norm(point_cloud[b] - point_cloud[b, i].unsqueeze(0), dim=1)
                nearest_indices = torch.topk(distances, k=min(6, num_points), largest=False).indices
                
                if len(nearest_indices) >= 3:
                    # Compute normal using PCA
                    neighbors = point_cloud[b, nearest_indices]
                    centered = neighbors - torch.mean(neighbors, dim=0)
                    cov_matrix = torch.matmul(centered.T, centered)
                    eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
                    
                    # Normal is eigenvector with smallest eigenvalue
                    normal = eigenvectors[:, torch.argmin(eigenvalues.real)].real
                    normals[b, i] = normal / torch.norm(normal)
        
        return normals
    
    def _map_point_cloud_to_grid(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Map point cloud features to tetrahedral grid"""
        batch_size, num_points, feature_dim = features.shape
        
        # Initialize grid features
        grid_features = torch.zeros(batch_size, 64, feature_dim, device=features.device)
        
        for b in range(batch_size):
            for i in range(min(num_points, 64)):
                # Find nearest grid point
                grid_idx = self._find_nearest_grid_point(points[b, i])
                grid_features[b, grid_idx] = features[b, i]
        
        return grid_features
    
    def _find_nearest_grid_point(self, point: torch.Tensor) -> int:
        """Find nearest grid point to given 3D point"""
        distances = torch.norm(self.grid.points - point.unsqueeze(0), dim=1)
        return torch.argmin(distances).item()


class ProcessOptimizer(nn.Module):
    """Optimizes manufacturing processes using tetrahedral AI"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256):
        super(ProcessOptimizer, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        
        # Process parameter encoder
        self.parameter_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # temp, pressure, speed, flow, vibration
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Quality metrics encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(10, hidden_dim // 2),  # Various quality metrics
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Tetrahedral optimization network
        self.optimization_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Parameter adjustment predictor
        self.adjustment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # 5 process parameters
            nn.Tanh()
        )
        
        # Quality improvement predictor
        self.improvement_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_parameters: torch.Tensor, 
                quality_metrics: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Optimize manufacturing process parameters
        Args:
            current_parameters: Current process parameters [batch_size, 5]
            quality_metrics: Current quality metrics [batch_size, 10]
        Returns:
            Dictionary with optimization suggestions
        """
        batch_size = current_parameters.shape[0]
        
        # Encode inputs
        param_features = self.parameter_encoder(current_parameters)  # [batch_size, hidden_dim]
        quality_features = self.quality_encoder(quality_metrics)     # [batch_size, hidden_dim]
        
        # Combine features
        combined_features = torch.cat([param_features, quality_features], dim=-1)
        
        # Expand to grid dimensions
        grid_features = combined_features.unsqueeze(1).expand(-1, 64, -1)  # [batch_size, 64, hidden_dim*2]
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [batch_size, hidden_dim*2, 64]
        
        # Process through tetrahedral network
        processed_features = self.optimization_network(grid_features)  # [batch_size, hidden_dim, 64]
        
        # Aggregate features for prediction
        aggregated_features = torch.mean(processed_features, dim=-1)  # [batch_size, hidden_dim]
        
        # Predict parameter adjustments
        adjustments = self.adjustment_predictor(aggregated_features)  # [batch_size, 5]
        
        # Predict quality improvement
        improvement = self.improvement_predictor(aggregated_features)  # [batch_size, 1]
        
        return {
            'parameter_adjustments': adjustments,
            'quality_improvement': improvement.squeeze(-1),
            'processed_features': processed_features
        }


class PredictiveMaintenanceModel(nn.Module):
    """Predicts equipment maintenance needs using tetrahedral AI"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256):
        super(PredictiveMaintenanceModel, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        
        # Sensor data encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim),  # 20 sensor channels
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Tetrahedral maintenance network
        self.maintenance_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Failure probability predictor
        self.failure_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Remaining useful life predictor
        self.rul_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Maintenance recommendation
        self.maintenance_recommender = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 maintenance actions
            nn.Softmax(dim=-1)
        )
    
    def forward(self, sensor_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict maintenance needs from sensor data
        Args:
            sensor_data: Time series sensor data [batch_size, sequence_length, 20]
        Returns:
            Dictionary with maintenance predictions
        """
        batch_size, seq_len, num_sensors = sensor_data.shape
        
        # Encode sensor data
        encoded_sensors = self.sensor_encoder(sensor_data)  # [batch_size, seq_len, hidden_dim]
        
        # Temporal encoding
        temporal_features, _ = self.temporal_encoder(encoded_sensors)  # [batch_size, seq_len, hidden_dim*2]
        
        # Use last time step for prediction
        last_features = temporal_features[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # Map to tetrahedral grid
        grid_features = last_features.unsqueeze(1).expand(-1, 64, -1)  # [batch_size, 64, hidden_dim*2]
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [batch_size, hidden_dim*2, 64]
        
        # Process through tetrahedral network
        processed_features = self.maintenance_network(grid_features)  # [batch_size, hidden_dim, 64]
        
        # Aggregate features for prediction
        aggregated_features = torch.mean(processed_features, dim=-1)  # [batch_size, hidden_dim]
        
        # Predict failure probability
        failure_prob = self.failure_predictor(aggregated_features)  # [batch_size, 1]
        
        # Predict remaining useful life
        rul = self.rul_predictor(aggregated_features)  # [batch_size, 1]
        
        # Recommend maintenance action
        maintenance_action = self.maintenance_recommender(aggregated_features)  # [batch_size, 3]
        
        return {
            'failure_probability': failure_prob.squeeze(-1),
            'remaining_useful_life': rul.squeeze(-1),
            'maintenance_recommendation': maintenance_action,
            'processed_features': processed_features
        }


class ManufacturingModule:
    """Main module for manufacturing applications"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.grid = TetrahedralGrid(device)
        
        # Initialize specialized models
        self.quality_detector = QualityControlDetector(self.grid).to(device)
        self.process_optimizer = ProcessOptimizer(self.grid).to(device)
        self.maintenance_predictor = PredictiveMaintenanceModel(self.grid).to(device)
        
        # Defect type mapping
        self.defect_types = ['scratch', 'dent', 'crack', 'hole', 'deformation']
        self.maintenance_actions = ['continue_operation', 'schedule_maintenance', 'immediate_shutdown']
    
    def detect_defects(self, point_cloud_data: Dict[str, Any]) -> List[ManufacturingDefect]:
        """Detect defects in 3D point cloud data"""
        # Convert to tensors
        points = torch.tensor(point_cloud_data['points'], device=self.device, dtype=torch.float32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension
        
        normals = None
        if 'normals' in point_cloud_data:
            normals = torch.tensor(point_cloud_data['normals'], device=self.device, dtype=torch.float32)
            if normals.dim() == 2:
                normals = normals.unsqueeze(0)
        
        colors = None
        if 'colors' in point_cloud_data:
            colors = torch.tensor(point_cloud_data['colors'], device=self.device, dtype=torch.float32)
            if colors.dim() == 2:
                colors = colors.unsqueeze(0)
        
        # Detect defects
        with torch.no_grad():
            self.quality_detector.eval()
            results = self.quality_detector(points, normals, colors)
        
        # Process results
        defects = []
        defect_probs = results['defect_probabilities'].cpu().numpy()[0]  # [64, 5]
        severity = results['severity'].cpu().numpy()[0]           # [64]
        
        for grid_idx in range(64):
            # Find most likely defect type
            max_prob = np.max(defect_probs[grid_idx])
            if max_prob > 0.5:  # Confidence threshold
                defect_type_idx = np.argmax(defect_probs[grid_idx])
                defect_type = self.defect_types[defect_type_idx]
                
                # Get grid point location
                grid_point = self.grid.points[grid_idx].cpu().numpy()
                
                defect = ManufacturingDefect(
                    defect_type=defect_type,
                    location=tuple(grid_point),
                    severity=float(severity[grid_idx]),
                    confidence=float(max_prob),
                    dimensions={'estimated_size': 0.1}  # Placeholder
                )
                defects.append(defect)
        
        return defects
    
    def optimize_process(self, current_parameters: Dict[str, float], 
                        quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Optimize manufacturing process parameters"""
        # Convert to tensors
        param_tensor = torch.tensor([
            current_parameters['temperature'],
            current_parameters['pressure'],
            current_parameters['speed'],
            current_parameters['material_flow'],
            current_parameters['vibration']
        ], device=self.device, dtype=torch.float32).unsqueeze(0)
        
        quality_tensor = torch.tensor(list(quality_metrics.values()), 
                                     device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Optimize
        with torch.no_grad():
            self.process_optimizer.eval()
            results = self.process_optimizer(param_tensor, quality_tensor)
        
        # Process results
        adjustments = results['parameter_adjustments'].cpu().numpy()[0]
        improvement = results['quality_improvement'].cpu().numpy()[0]
        
        # Calculate new parameters
        param_names = ['temperature', 'pressure', 'speed', 'material_flow', 'vibration']
        new_parameters = {}
        
        for i, name in enumerate(param_names):
            current_value = current_parameters[name]
            adjustment = adjustments[i] * 0.1  # Scale down adjustments
            new_parameters[name] = current_value + adjustment
        
        return {
            'optimized_parameters': new_parameters,
            'expected_improvement': float(improvement),
            'adjustments': {name: float(adj) for name, adj in zip(param_names, adjustments)}
        }
    
    def predict_maintenance(self, sensor_data: List[List[float]]) -> Dict[str, Any]:
        """Predict maintenance needs from sensor data"""
        # Convert to tensor
        sensor_tensor = torch.tensor(sensor_data, device=self.device, dtype=torch.float32)
        if sensor_tensor.dim() == 2:
            sensor_tensor = sensor_tensor.unsqueeze(0)  # Add batch dimension
        
        # Predict maintenance
        with torch.no_grad():
            self.maintenance_predictor.eval()
            results = self.maintenance_predictor(sensor_tensor)
        
        # Process results
        failure_prob = float(results['failure_probability'].cpu().numpy()[0])
        rul = float(results['remaining_useful_life'].cpu().numpy()[0])
        maintenance_action = results['maintenance_recommendation'].cpu().numpy()[0]
        
        # Get recommended action
        action_idx = np.argmax(maintenance_action)
        recommended_action = self.maintenance_actions[action_idx]
        action_confidence = float(maintenance_action[action_idx])
        
        return {
            'failure_probability': failure_prob,
            'remaining_useful_life_hours': max(0, rul),  # Ensure non-negative
            'recommended_action': recommended_action,
            'action_confidence': action_confidence,
            'action_probabilities': {
                action: float(prob) 
                for action, prob in zip(self.maintenance_actions, maintenance_action)
            }
        }
    
    def analyze_quality_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality trends from historical data"""
        if not historical_data:
            return {'error': 'No historical data provided'}
        
        # Extract quality metrics
        quality_scores = [data.get('quality_score', 0) for data in historical_data]
        defect_rates = [data.get('defect_rate', 0) for data in historical_data]
        
        # Compute statistics
        mean_quality = np.mean(quality_scores)
        std_quality = np.std(quality_scores)
        mean_defect_rate = np.mean(defect_rates)
        
        # Detect trends
        if len(quality_scores) >= 10:
            recent_quality = np.mean(quality_scores[-5:])
            older_quality = np.mean(quality_scores[-10:-5])
            quality_trend = recent_quality - older_quality
        else:
            quality_trend = 0
        
        return {
            'mean_quality_score': float(mean_quality),
            'quality_std': float(std_quality),
            'mean_defect_rate': float(mean_defect_rate),
            'quality_trend': float(quality_trend),
            'trend_direction': 'improving' if quality_trend > 0 else 'declining' if quality_trend < 0 else 'stable',
            'data_points': len(historical_data)
        }


if __name__ == "__main__":
    # Test the manufacturing module
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = ManufacturingModule(device)
    
    # Test defect detection
    point_cloud_data = {
        'points': np.random.randn(100, 3).tolist(),
        'normals': np.random.randn(100, 3).tolist(),
        'colors': np.random.rand(100, 3).tolist()
    }
    
    defects = module.detect_defects(point_cloud_data)
    print(f"Defect detection found {len(defects)} defects")
    for defect in defects[:3]:  # Show first 3
        print(f"  {defect.defect_type} at {defect.location} (confidence: {defect.confidence:.3f})")
    
    # Test process optimization
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
    
    optimization_results = module.optimize_process(current_params, quality_metrics)
    print(f"\nProcess optimization results:")
    print(f"  Expected improvement: {optimization_results['expected_improvement']:.3f}")
    print(f"  New temperature: {optimization_results['optimized_parameters']['temperature']:.1f}")
    
    # Test predictive maintenance
    sensor_data = np.random.randn(50, 20).tolist()  # 50 time steps, 20 sensors
    maintenance_results = module.predict_maintenance(sensor_data)
    print(f"\nPredictive maintenance results:")
    print(f"  Failure probability: {maintenance_results['failure_probability']:.3f}")
    print(f"  Remaining useful life: {maintenance_results['remaining_useful_life_hours']:.1f} hours")
    print(f"  Recommended action: {maintenance_results['recommended_action']}")
    
    print("\nManufacturing module test completed successfully!")