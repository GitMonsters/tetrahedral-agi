"""
Enhanced Spatial Attention Mechanism for improved pattern matching
Addresses SLE benchmark weakness in complex pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class EnhancedSpatialAttention(nn.Module):
    """
    Enhanced spatial attention with multi-scale pattern recognition
    Improves complex pattern matching capabilities
    """
    
    def __init__(self, channels: int, grid, num_heads: int = 8, num_scales: int = 3):
        super(EnhancedSpatialAttention, self).__init__()
        self.channels = channels
        self.grid = grid
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = channels // num_heads
        
        # Multi-scale feature extractors
        self.scale_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels // num_scales, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels // num_scales, channels // num_scales, kernel_size=3, padding=1),
                nn.ReLU()
            ) for _ in range(num_scales)
        ])
        
        # Enhanced attention projections
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)
        
        # Pattern memory networks
        self.pattern_memory = nn.Parameter(torch.randn(64, channels // 4))
        self.pattern_matching = nn.Sequential(
            nn.Linear(channels + channels // 4, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        
        # Multi-head attention with pattern integration
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(channels, channels)
        self.layer_norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with multi-scale pattern recognition
        Args:
            x: Input tensor [batch_size, channels, num_points]
        Returns:
            Enhanced output with better pattern recognition
        """
        batch_size, channels, num_points = x.shape
        
        # Transpose for processing
        features = x.transpose(1, 2)  # [batch_size, num_points, channels]
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for i, scale_net in enumerate(self.scale_networks):
            # Apply convolution on spatial dimension
            scale_features = scale_net(x)  # [batch_size, channels//num_scales, num_points]
            scale_features = scale_features.transpose(1, 2)  # [batch_size, num_points, channels//num_scales]
            multi_scale_features.append(scale_features)
        
        # Combine multi-scale features
        combined_scales = torch.cat(multi_scale_features, dim=-1)  # [batch_size, num_points, channels]
        
        # Integrate pattern memory
        pattern_expanded = self.pattern_memory.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_points, -1
        )  # [batch_size, num_points, channels//4]
        
        # Pattern matching with memory
        pattern_input = torch.cat([combined_scales, pattern_expanded], dim=-1)
        pattern_features = self.pattern_matching(pattern_input)
        
        # Enhanced multi-head attention
        queries = self.query_proj(pattern_features)
        keys = self.key_proj(pattern_features)
        values = self.value_proj(pattern_features)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.pattern_attention(
            queries, keys, values
        )
        
        # Apply geometric bias
        geo_bias = self._compute_geometric_bias()
        attended_features = attended_features + geo_bias.unsqueeze(0)
        
        # Residual connection and layer norm
        output = self.layer_norm(attended_features + pattern_features)
        
        # Final projection
        output = self.output_proj(output)
        
        return output.transpose(1, 2)  # [batch_size, channels, num_points]
    
    def _compute_geometric_bias(self) -> torch.Tensor:
        """Compute geometric bias based on tetrahedral structure"""
        # Distance-based bias for closer points
        geo_bias = torch.zeros(64, 64, device=self.pattern_memory.device)
        
        for i in range(64):
            for j in range(64):
                if self.grid.adjacency_matrix[i, j] > 0:
                    # Connected points get positive bias
                    geo_bias[i, j] = 0.5
                else:
                    # Non-connected points get negative bias
                    distance = torch.norm(self.grid.points[i] - self.grid.points[j])
                    geo_bias[i, j] = -0.1 / (1 + distance)
        
        return geo_bias


class WorkingMemoryModule(nn.Module):
    """
    Working memory module for assembly planning
    Addresses SLE benchmark weakness in complex assembly tasks
    """
    
    def __init__(self, hidden_dim: int, memory_slots: int = 8):
        super(WorkingMemoryModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        
        # Memory storage
        self.memory = nn.Parameter(torch.randn(memory_slots, hidden_dim))
        self.memory_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Assembly planning networks
        self.state_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3D action space
        )
        
        # Constraint checking
        self.constraint_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Search strategy
        self.search_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [exploitation, exploration]
            nn.Softmax(dim=-1)
        )
    
    def forward(self, current_state: torch.Tensor, 
                assembly_goal: torch.Tensor,
                step: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Plan assembly sequence using working memory and constraint checking
        Args:
            current_state: Current assembly state [batch_size, hidden_dim]
            assembly_goal: Target assembly state [batch_size, hidden_dim]
            step: Current planning step
        Returns:
            Tuple of (action, constraint_violation, search_strategy)
        """
        # Encode current state
        state_features = self.state_encoder(current_state)
        
        # Read from working memory
        memory_content = self.memory.unsqueeze(0).expand(current_state.shape[0], -1, -1)
        
        # Memory attention
        attention_weights = torch.matmul(state_features.unsqueeze(1), memory_content.transpose(-2, -1))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        attended_memory = torch.matmul(attention_weights, memory_content).squeeze(1)
        
        # Combine state with memory
        state_with_memory = torch.cat([state_features, attended_memory], dim=-1)
        
        # Predict next action
        action = self.action_predictor(state_with_memory)
        
        # Check constraint violations
        constraint_input = torch.cat([current_state, action], dim=-1)
        constraint_violation = self.constraint_network(constraint_input)
        
        # Determine search strategy
        search_strategy = self.search_policy(state_features)
        
        return action, constraint_violation, search_strategy
    
    def write_to_memory(self, experience: torch.Tensor):
        """Write experience to working memory"""
        with torch.no_grad():
            # Update memory with new experience
            gate = torch.sigmoid(self.memory_gate(
                torch.cat([experience[:self.hidden_dim], experience[self.hidden_dim:]], dim=-1)
            ))
            self.memory.data = gate * experience[:self.memory_slots] + (1 - gate) * self.memory.data


class CubeFoldingSimulator(nn.Module):
    """
    Cube folding simulator for mental 3D transformation
    Addresses SLE benchmark weakness in cube folding tasks
    """
    
    def __init__(self, hidden_dim: int = 128):
        super(CubeFoldingSimulator, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Net encoding networks
        self.net_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6 squares in cube net
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3D folding simulation
        self.folding_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 12),  # 8 vertices + 4 face normals
            nn.Tanh()
        )
        
        # Fold validation
        self.validator = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Mental rotation capability
        self.rotation_predictor = nn.Sequential(
            nn.Linear(12, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3D rotation angles
            nn.Tanh()
        )
    
    def forward(self, cube_net: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate cube folding from 2D net
        Args:
            cube_net: 2D cube net representation [batch_size, 6, 2] (x, y coordinates)
        Returns:
            Tuple of (folded_3d, validity_score, rotation_angles)
        """
        batch_size = cube_net.shape[0]
        
        # Flatten net for encoding
        net_flat = cube_net.view(batch_size, -1)  # [batch_size, 12]
        
        # Encode net structure
        net_features = self.net_encoder(net_flat)
        
        # Simulate 3D folding
        folding_result = self.folding_network(net_features)
        
        # Validate folded cube
        validity_score = self.validator(folding_result)
        
        # Predict optimal viewing rotation
        rotation_angles = self.rotation_predictor(folding_result)
        
        return folding_result, validity_score, rotation_angles
    
    def mental_rotation(self, cube_3d: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Apply mental rotation to 3D cube"""
        batch_size = cube_3d.shape[0]
        
        # Create rotation matrices
        rotation_matrices = self._create_3d_rotation_matrices(angles)
        
        # Apply rotation
        vertices = cube_3d[:, :8]  # First 8 elements are vertices
        vertices_reshaped = vertices.view(batch_size, 8, 3)
        
        rotated_vertices = torch.matmul(rotation_matrices, vertices_reshaped.transpose(-2, -1))
        rotated_vertices = rotated_vertices.transpose(-2, -1).view(batch_size, 8, 3)
        
        return torch.cat([rotated_vertices.view(batch_size, 24), cube_3d[:, 24:]], dim=-1)
    
    def _create_3d_rotation_matrices(self, angles: torch.Tensor) -> torch.Tensor:
        """Create 3D rotation matrices from angles"""
        batch_size = angles.shape[0]
        rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
        
        # Rotation matrices
        Rx = torch.stack([
            torch.ones_like(rx), torch.zeros_like(rx), torch.zeros_like(rx),
            torch.zeros_like(rx), torch.cos(rx), -torch.sin(rx),
            torch.zeros_like(rx), torch.sin(rx), torch.cos(rx)
        ], dim=-1).view(batch_size, 3, 3)
        
        Ry = torch.stack([
            torch.cos(ry), torch.zeros_like(ry), torch.sin(ry),
            torch.zeros_like(ry), torch.ones_like(ry), torch.zeros_like(ry),
            -torch.sin(ry), torch.zeros_like(ry), torch.cos(ry)
        ], dim=-1).view(batch_size, 3, 3)
        
        Rz = torch.stack([
            torch.cos(rz), -torch.sin(rz), torch.zeros_like(rz),
            torch.sin(rz), torch.cos(rz), torch.zeros_like(rz),
            torch.zeros_like(rz), torch.zeros_like(rz), torch.ones_like(rz)
        ], dim=-1).view(batch_size, 3, 3)
        
        # Combined rotation
        return torch.matmul(torch.matmul(Rz, Ry), Rx)


if __name__ == "__main__":
    print("Enhanced modules loaded successfully!")
    print("• EnhancedSpatialAttention: Multi-scale pattern recognition")
    print("• WorkingMemoryModule: Assembly planning with constraints")
    print("• CubeFoldingSimulator: Mental 3D transformation")
    print("\nReady for integration with tetrahedral AI framework!")