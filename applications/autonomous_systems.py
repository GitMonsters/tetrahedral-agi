"""
Autonomous Systems Module for 64-Point Tetrahedron AI
Specialized applications for spatial navigation, object recognition, and robotics control
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import math
from scipy.spatial.transform import Rotation
from collections import deque

from ..neural_network.tetrahedral_network import TetrahedralAGINetwork
from ..geometry.tetrahedral_grid import TetrahedralGrid


@dataclass
class NavigationState:
    """Represents the navigation state of an autonomous system"""
    position: torch.Tensor  # [3] current position
    orientation: torch.Tensor  # [4] quaternion orientation
    velocity: torch.Tensor   # [3] current velocity
    angular_velocity: torch.Tensor  # [3] angular velocity
    goal_position: torch.Tensor  # [3] target position
    obstacles: torch.Tensor  # [N, 3] obstacle positions


@dataclass
class PerceptionResult:
    """Result of 3D perception and object recognition"""
    objects: List[Dict[str, Any]]
    scene_understanding: Dict[str, Any]
    confidence_scores: torch.Tensor
    spatial_features: torch.Tensor


class SpatialNavigationNetwork(nn.Module):
    """Handles 3D spatial navigation and path planning"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256):
        super(SpatialNavigationNetwork, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(16, hidden_dim // 2),  # position(3) + orientation(4) + velocity(3) + angular_vel(3) + goal(3)
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Obstacle encoder
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # 3D obstacle position
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Tetrahedral navigation network
        self.navigation_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Path planning heads
        self.path_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3D direction vector
            nn.Tanh()
        )
        
        # Velocity controller
        self.velocity_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Collision predictor
        self.collision_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, nav_state: NavigationState) -> Dict[str, torch.Tensor]:
        """
        Plan navigation path and control movements
        Args:
            nav_state: Current navigation state
        Returns:
            Dictionary with navigation commands and predictions
        """
        # Encode current state
        state_vector = torch.cat([
            nav_state.position,
            nav_state.orientation,
            nav_state.velocity,
            nav_state.angular_velocity,
            nav_state.goal_position
        ])
        state_features = self.state_encoder(state_vector.unsqueeze(0))  # [1, hidden_dim]
        
        # Encode obstacles
        if len(nav_state.obstacles) > 0:
            obstacle_features = self.obstacle_encoder(nav_state.obstacles)  # [N, hidden_dim]
            aggregated_obstacles = torch.mean(obstacle_features, dim=0, keepdim=True)  # [1, hidden_dim]
        else:
            aggregated_obstacles = torch.zeros(1, self.hidden_dim, device=state_vector.device)
        
        # Combine features
        combined_features = torch.cat([state_features, aggregated_obstacles], dim=-1)
        
        # Map to tetrahedral grid
        grid_features = combined_features.unsqueeze(1).expand(-1, 64, -1)  # [1, 64, hidden_dim*2]
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [1, hidden_dim*2, 64]
        
        # Process through tetrahedral network
        processed_features = self.navigation_network(grid_features)  # [1, hidden_dim, 64]
        
        # Aggregate features for prediction
        aggregated_features = torch.mean(processed_features, dim=-1)  # [1, hidden_dim]
        
        # Predict navigation direction
        direction = self.path_predictor(aggregated_features)  # [1, 3]
        
        # Predict velocity magnitude
        velocity_mag = self.velocity_controller(aggregated_features)  # [1, 1]
        
        # Predict collision probability
        collision_prob = self.collision_predictor(aggregated_features)  # [1, 1]
        
        return {
            'navigation_direction': direction.squeeze(0),
            'velocity_magnitude': velocity_mag.squeeze(0),
            'collision_probability': collision_prob.squeeze(0),
            'processed_features': processed_features
        }


class ObjectRecognitionNetwork(nn.Module):
    """3D object recognition and scene understanding"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256, num_classes: int = 100):
        super(ObjectRecognitionNetwork, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Point cloud encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Feature descriptor encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),  # Geometric features
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Tetrahedral recognition network
        self.recognition_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 2,
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Object classification heads
        self.object_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Pose estimator
        self.pose_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 7)  # 7D pose (position + quaternion)
        )
        
        # Size estimator
        self.size_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3D dimensions
            nn.Sigmoid()
        )
    
    def forward(self, point_cloud: torch.Tensor, 
                features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Recognize objects in 3D point cloud
        Args:
            point_cloud: 3D points [batch_size, num_points, 3]
            features: Optional geometric features [batch_size, num_points, 64]
        Returns:
            Dictionary with recognition results
        """
        batch_size, num_points, _ = point_cloud.shape
        
        # Encode point cloud
        point_features = self.point_encoder(point_cloud)  # [batch_size, num_points, hidden_dim]
        
        # Generate or encode features
        if features is None:
            features = self._compute_geometric_features(point_cloud)
        feature_encodings = self.feature_encoder(features)  # [batch_size, num_points, hidden_dim]
        
        # Combine features
        combined_features = torch.cat([point_features, feature_encodings], dim=-1)
        
        # Map to tetrahedral grid
        grid_features = self._map_point_cloud_to_grid(combined_features, point_cloud)
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [batch_size, hidden_dim*2, 64]
        
        # Process through tetrahedral network
        processed_features = self.recognition_network(grid_features)  # [batch_size, hidden_dim, 64]
        
        # Aggregate features for classification
        aggregated_features = torch.mean(processed_features, dim=-1)  # [batch_size, hidden_dim]
        
        # Classify objects
        class_logits = self.object_classifier(aggregated_features)  # [batch_size, num_classes]
        class_probs = torch.softmax(class_logits, dim=-1)
        
        # Estimate pose
        pose = self.pose_estimator(aggregated_features)  # [batch_size, 7]
        
        # Estimate size
        size = self.size_estimator(aggregated_features)  # [batch_size, 3]
        
        return {
            'class_probabilities': class_probs,
            'pose_estimate': pose,
            'size_estimate': size,
            'processed_features': processed_features,
            'grid_features': grid_features
        }
    
    def _compute_geometric_features(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Compute geometric features for point cloud"""
        batch_size, num_points, _ = point_cloud.shape
        features = torch.zeros(batch_size, num_points, 64, device=point_cloud.device)
        
        for b in range(batch_size):
            for i in range(num_points):
                point = point_cloud[b, i]
                
                # Compute local geometric features
                # Find nearest neighbors
                distances = torch.norm(point_cloud[b] - point.unsqueeze(0), dim=1)
                nearest_indices = torch.topk(distances, k=min(10, num_points), largest=False).indices
                
                if len(nearest_indices) >= 3:
                    neighbors = point_cloud[b, nearest_indices]
                    
                    # Compute local covariance
                    centered = neighbors - torch.mean(neighbors, dim=0)
                    cov_matrix = torch.matmul(centered.T, centered) / len(neighbors)
                    
                    # Extract eigenvalues
                    eigenvalues = torch.linalg.eigvals(cov_matrix).real
                    eigenvalues = torch.sort(eigenvalues, descending=True).values
                    
                    # Create feature vector
                    feat = torch.cat([
                        eigenvalues[:3],  # 3 eigenvalues
                        torch.norm(eigenvalues[:3]),  # 1 norm
                        (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else torch.tensor(0.0),  # 1 linearity
                        (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else torch.tensor(0.0),  # 1 planarity
                        eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else torch.tensor(0.0),  # 1 sphericity
                        torch.zeros(56)  # Pad to 64 dimensions
                    ])
                    
                    features[b, i] = feat
        
        return features
    
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


class RoboticsController(nn.Module):
    """Robotics control system using tetrahedral AI"""
    
    def __init__(self, grid: TetrahedralGrid, hidden_dim: int = 256, num_joints: int = 7):
        super(RoboticsController, self).__init__()
        self.grid = grid
        self.hidden_dim = hidden_dim
        self.num_joints = num_joints
        
        # Current state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, hidden_dim // 2),  # joint positions + velocities
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim // 2),  # target pose (position + quaternion)
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Sensor encoder
        self.sensor_encoder = nn.Sequential(
            nn.Linear(20, hidden_dim // 2),  # various sensor inputs
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Tetrahedral control network
        self.control_network = TetrahedralAGINetwork(
            input_channels=hidden_dim * 3,
            hidden_channels=hidden_dim,
            output_channels=hidden_dim,
            device='cuda'
        )
        
        # Joint control heads
        self.position_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_joints),
            nn.Tanh()
        )
        
        self.velocity_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_joints),
            nn.Tanh()
        )
        
        # Force predictor
        self.force_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_joints)
        )
        
        # Stability predictor
        self.stability_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, joint_positions: torch.Tensor, joint_velocities: torch.Tensor,
                target_pose: torch.Tensor, sensor_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate robot control commands
        Args:
            joint_positions: Current joint positions [batch_size, num_joints]
            joint_velocities: Current joint velocities [batch_size, num_joints]
            target_pose: Target end-effector pose [batch_size, 7]
            sensor_data: Sensor readings [batch_size, 20]
        Returns:
            Dictionary with control commands
        """
        batch_size = joint_positions.shape[0]
        
        # Encode inputs
        state_vector = torch.cat([joint_positions, joint_velocities], dim=-1)
        state_features = self.state_encoder(state_vector)  # [batch_size, hidden_dim]
        
        target_features = self.target_encoder(target_pose)  # [batch_size, hidden_dim]
        sensor_features = self.sensor_encoder(sensor_data)  # [batch_size, hidden_dim]
        
        # Combine features
        combined_features = torch.cat([state_features, target_features, sensor_features], dim=-1)
        
        # Map to tetrahedral grid
        grid_features = combined_features.unsqueeze(1).expand(-1, 64, -1)  # [batch_size, 64, hidden_dim*3]
        
        # Transpose for network input
        grid_features = grid_features.transpose(1, 2)  # [batch_size, hidden_dim*3, 64]
        
        # Process through tetrahedral network
        processed_features = self.control_network(grid_features)  # [batch_size, hidden_dim, 64]
        
        # Aggregate features for control
        aggregated_features = torch.mean(processed_features, dim=-1)  # [batch_size, hidden_dim]
        
        # Generate control commands
        position_commands = self.position_controller(aggregated_features)  # [batch_size, num_joints]
        velocity_commands = self.velocity_controller(aggregated_features)  # [batch_size, num_joints]
        force_predictions = self.force_predictor(aggregated_features)     # [batch_size, num_joints]
        stability_score = self.stability_predictor(aggregated_features)   # [batch_size, 1]
        
        return {
            'position_commands': position_commands,
            'velocity_commands': velocity_commands,
            'force_predictions': force_predictions,
            'stability_score': stability_score.squeeze(-1),
            'processed_features': processed_features
        }


class AutonomousSystemsModule:
    """Main module for autonomous systems applications"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.grid = TetrahedralGrid(device)
        
        # Initialize specialized models
        self.navigation_network = SpatialNavigationNetwork(self.grid).to(device)
        self.object_recognition = ObjectRecognitionNetwork(self.grid).to(device)
        self.robotics_controller = RoboticsController(self.grid).to(device)
        
        # Object class names (placeholder)
        self.object_classes = [f'object_{i}' for i in range(100)]
        
        # Navigation history
        self.navigation_history = deque(maxlen=1000)
    
    def navigate_to_goal(self, current_state: Dict[str, Any], 
                         obstacles: List[List[float]]) -> Dict[str, Any]:
        """Plan navigation to goal"""
        # Convert to tensors
        position = torch.tensor(current_state['position'], device=self.device, dtype=torch.float32)
        orientation = torch.tensor(current_state['orientation'], device=self.device, dtype=torch.float32)
        velocity = torch.tensor(current_state['velocity'], device=self.device, dtype=torch.float32)
        angular_velocity = torch.tensor(current_state['angular_velocity'], device=self.device, dtype=torch.float32)
        goal_position = torch.tensor(current_state['goal_position'], device=self.device, dtype=torch.float32)
        
        obstacle_tensor = torch.tensor(obstacles, device=self.device, dtype=torch.float32) if obstacles else torch.empty(0, 3, device=self.device)
        
        # Create navigation state
        nav_state = NavigationState(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            goal_position=goal_position,
            obstacles=obstacle_tensor
        )
        
        # Plan navigation
        with torch.no_grad():
            self.navigation_network.eval()
            results = self.navigation_network(nav_state)
        
        # Process results
        direction = results['navigation_direction'].cpu().numpy()
        velocity_mag = results['velocity_magnitude'].cpu().numpy()
        collision_prob = results['collision_probability'].cpu().numpy()
        
        # Calculate velocity vector
        velocity_vector = direction * velocity_mag
        
        # Store in history
        self.navigation_history.append({
            'timestamp': time.time(),
            'position': position.cpu().numpy().tolist(),
            'goal': goal_position.cpu().numpy().tolist(),
            'collision_probability': float(collision_prob)
        })
        
        return {
            'direction': direction.tolist(),
            'velocity_vector': velocity_vector.tolist(),
            'velocity_magnitude': float(velocity_mag),
            'collision_probability': float(collision_prob),
            'is_safe': float(collision_prob) < 0.3
        }
    
    def recognize_objects(self, point_cloud: List[List[float]]) -> PerceptionResult:
        """Recognize objects in 3D point cloud"""
        # Convert to tensor
        points = torch.tensor(point_cloud, device=self.device, dtype=torch.float32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension
        
        # Recognize objects
        with torch.no_grad():
            self.object_recognition.eval()
            results = self.object_recognition(points)
        
        # Process results
        class_probs = results['class_probabilities'].cpu().numpy()[0]
        pose = results['pose_estimate'].cpu().numpy()[0]
        size = results['size_estimate'].cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(class_probs)[-5:][::-1]
        top_objects = []
        
        for idx in top_indices:
            if class_probs[idx] > 0.1:  # Confidence threshold
                top_objects.append({
                    'class': self.object_classes[idx],
                    'confidence': float(class_probs[idx]),
                    'position': pose[:3].tolist(),
                    'orientation': pose[3:7].tolist(),
                    'size': size.tolist()
                })
        
        # Scene understanding
        scene_understanding = {
            'num_objects': len(top_objects),
            'dominant_object': top_objects[0]['class'] if top_objects else 'unknown',
            'average_confidence': float(np.mean([obj['confidence'] for obj in top_objects])) if top_objects else 0.0,
            'scene_complexity': min(1.0, len(top_objects) / 10.0)
        }
        
        return PerceptionResult(
            objects=top_objects,
            scene_understanding=scene_understanding,
            confidence_scores=torch.tensor(class_probs, device=self.device),
            spatial_features=results['processed_features']
        )
    
    def control_robot(self, current_joints: Dict[str, Any], 
                      target_pose: List[float],
                      sensor_data: List[float]) -> Dict[str, Any]:
        """Generate robot control commands"""
        # Convert to tensors
        joint_positions = torch.tensor(current_joints['positions'], device=self.device, dtype=torch.float32).unsqueeze(0)
        joint_velocities = torch.tensor(current_joints['velocities'], device=self.device, dtype=torch.float32).unsqueeze(0)
        target_pose_tensor = torch.tensor(target_pose, device=self.device, dtype=torch.float32).unsqueeze(0)
        sensor_tensor = torch.tensor(sensor_data, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        # Generate control commands
        with torch.no_grad():
            self.robotics_controller.eval()
            results = self.robotics_controller(joint_positions, joint_velocities, target_pose_tensor, sensor_tensor)
        
        # Process results
        position_commands = results['position_commands'].cpu().numpy()[0]
        velocity_commands = results['velocity_commands'].cpu().numpy()[0]
        force_predictions = results['force_predictions'].cpu().numpy()[0]
        stability_score = results['stability_score'].cpu().numpy()[0]
        
        return {
            'joint_position_commands': position_commands.tolist(),
            'joint_velocity_commands': velocity_commands.tolist(),
            'predicted_forces': force_predictions.tolist(),
            'stability_score': float(stability_score),
            'is_stable': float(stability_score) > 0.7
        }
    
    def analyze_navigation_performance(self) -> Dict[str, Any]:
        """Analyze navigation performance from history"""
        if len(self.navigation_history) < 10:
            return {'error': 'Insufficient navigation history'}
        
        # Extract metrics
        collision_probs = [entry['collision_probability'] for entry in self.navigation_history]
        
        # Compute statistics
        mean_collision_prob = np.mean(collision_probs)
        max_collision_prob = np.max(collision_probs)
        
        # Compute path efficiency (simplified)
        if len(self.navigation_history) >= 2:
            start_pos = np.array(self.navigation_history[0]['position'])
            current_pos = np.array(self.navigation_history[-1]['position'])
            goal_pos = np.array(self.navigation_history[0]['goal'])
            
            actual_distance = np.linalg.norm(current_pos - start_pos)
            direct_distance = np.linalg.norm(goal_pos - start_pos)
            
            if direct_distance > 0:
                path_efficiency = direct_distance / actual_distance
            else:
                path_efficiency = 1.0
        else:
            path_efficiency = 1.0
        
        return {
            'mean_collision_probability': float(mean_collision_prob),
            'max_collision_probability': float(max_collision_prob),
            'path_efficiency': float(path_efficiency),
            'navigation_samples': len(self.navigation_history),
            'safety_rating': 'excellent' if mean_collision_prob < 0.1 else 'good' if mean_collision_prob < 0.3 else 'poor'
        }


if __name__ == "__main__":
    import time
    
    # Test the autonomous systems module
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = AutonomousSystemsModule(device)
    
    # Test spatial navigation
    current_state = {
        'position': [0.0, 0.0, 0.0],
        'orientation': [1.0, 0.0, 0.0, 0.0],
        'velocity': [0.1, 0.0, 0.0],
        'angular_velocity': [0.0, 0.0, 0.1],
        'goal_position': [5.0, 5.0, 2.0]
    }
    
    obstacles = [[2.0, 1.0, 0.0], [3.0, 3.0, 1.0], [1.0, 4.0, 0.5]]
    
    navigation_results = module.navigate_to_goal(current_state, obstacles)
    print(f"Navigation results:")
    print(f"  Direction: {navigation_results['direction']}")
    print(f"  Velocity magnitude: {navigation_results['velocity_magnitude']:.3f}")
    print(f"  Collision probability: {navigation_results['collision_probability']:.3f}")
    print(f"  Is safe: {navigation_results['is_safe']}")
    
    # Test object recognition
    point_cloud = np.random.randn(100, 3).tolist()
    perception_results = module.recognize_objects(point_cloud)
    
    print(f"\nObject recognition results:")
    print(f"  Objects detected: {len(perception_results.objects)}")
    print(f"  Scene complexity: {perception_results.scene_understanding['scene_complexity']:.3f}")
    for obj in perception_results.objects[:3]:
        print(f"    {obj['class']} (confidence: {obj['confidence']:.3f})")
    
    # Test robotics control
    current_joints = {
        'positions': [0.0, 0.5, -0.3, 0.2, 0.0, 0.1, -0.1],
        'velocities': [0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0]
    }
    
    target_pose = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # position + quaternion
    sensor_data = np.random.randn(20).tolist()
    
    control_results = module.control_robot(current_joints, target_pose, sensor_data)
    print(f"\nRobotics control results:")
    print(f"  Stability score: {control_results['stability_score']:.3f}")
    print(f"  Is stable: {control_results['is_stable']}")
    print(f"  Joint commands generated: {len(control_results['joint_position_commands'])}")
    
    # Test navigation performance analysis
    time.sleep(0.1)  # Ensure some history
    performance = module.analyze_navigation_performance()
    print(f"\nNavigation performance:")
    print(f"  Mean collision probability: {performance['mean_collision_probability']:.3f}")
    print(f"  Path efficiency: {performance['path_efficiency']:.3f}")
    print(f"  Safety rating: {performance['safety_rating']}")
    
    print("\nAutonomous systems module test completed successfully!")