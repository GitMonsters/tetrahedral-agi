"""
Neural Network Architecture for 64-Point Tetrahedron AI
Implements the core neural network components based on tetrahedral geometry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from .tetrahedral_grid import TetrahedralGrid, TetrahedronGeometry


class TetrahedralConvolution(nn.Module):
    """
    Tetrahedral Convolution Layer
    Performs convolution operations on the tetrahedral grid structure
    """
    
    def __init__(self, in_channels: int, out_channels: int, grid: TetrahedralGrid):
        super(TetrahedralConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid = grid
        self.num_points = len(grid.points)
        
        # Learnable weights for tetrahedral convolution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 4))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Geometric attention weights
        self.geo_attention = nn.Parameter(torch.ones(self.num_points))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of tetrahedral convolution
        Args:
            x: Input tensor of shape [batch_size, in_channels, num_points]
        Returns:
            Output tensor of shape [batch_size, out_channels, num_points]
        """
        batch_size, in_channels, num_points = x.shape
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, num_points, device=x.device)
        
        # Perform tetrahedral convolution
        for point_idx in range(num_points):
            # Get neighboring points
            neighbors = self.grid.get_neighbors(point_idx)
            
            if len(neighbors) > 0:
                # Gather neighbor features
                neighbor_features = x[:, :, neighbors]  # [batch_size, in_channels, num_neighbors]
                
                # Apply geometric attention
                attention_weights = F.softmax(self.geo_attention[neighbors], dim=0)
                neighbor_features = neighbor_features * attention_weights.unsqueeze(0).unsqueeze(0)
                
                # Aggregate neighbor features
                aggregated = torch.sum(neighbor_features, dim=2)  # [batch_size, in_channels]
                
                # Apply linear transformation
                output[:, :, point_idx] = F.linear(aggregated, self.weight[:, :, point_idx], self.bias)
            else:
                # No neighbors, use identity transformation
                output[:, :, point_idx] = x[:, :, point_idx]
        
        return output


class OctahedralCavityProcessor(nn.Module):
    """
    Octahedral Cavity Processor
    Processes information within the 14 octahedral cavities
    """
    
    def __init__(self, channels: int, grid: TetrahedralGrid):
        super(OctahedralCavityProcessor, self).__init__()
        self.channels = channels
        self.grid = grid
        self.num_cavities = len(grid.octahedral_cavities)
        
        # Cavity processing networks
        self.cavity_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.ReLU(),
                nn.Linear(channels * 2, channels),
                nn.Tanh()
            ) for _ in range(self.num_cavities)
        ])
        
        # Cross-cavity attention
        self.cross_attention = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through octahedral cavities
        Args:
            x: Input tensor of shape [batch_size, channels, num_points]
        Returns:
            Processed tensor of same shape
        """
        batch_size, channels, num_points = x.shape
        
        # Extract cavity features
        cavity_features = []
        for cavity_idx in range(self.num_cavities):
            cavity_center = self.grid.get_octahedral_cavity(cavity_idx)
            
            # Find points near this cavity
            distances = torch.norm(self.grid.points - cavity_center.unsqueeze(0), dim=1)
            nearby_points = torch.where(distances < 0.5)[0]
            
            if len(nearby_points) > 0:
                # Aggregate features from nearby points
                cavity_feat = torch.mean(x[:, :, nearby_points], dim=2)
            else:
                # No nearby points, use zeros
                cavity_feat = torch.zeros(batch_size, channels, device=x.device)
            
            cavity_features.append(cavity_feat)
        
        # Stack cavity features
        cavity_features = torch.stack(cavity_features, dim=1)  # [batch_size, num_cavities, channels]
        
        # Process through cavity networks
        processed_cavities = []
        for i, network in enumerate(self.cavity_networks):
            processed = network(cavity_features[:, i, :])
            processed_cavities.append(processed)
        
        processed_cavities = torch.stack(processed_cavities, dim=1)
        
        # Apply cross-cavity attention
        attended_cavities, _ = self.cross_attention(processed_cavities, processed_cavities, processed_cavities)
        
        # Distribute cavity features back to points
        output = x.clone()
        for point_idx in range(num_points):
            point = self.grid.points[point_idx]
            
            # Find nearest cavity
            cavity_distances = torch.norm(self.grid.octahedral_cavities - point.unsqueeze(0), dim=1)
            nearest_cavity = torch.argmin(cavity_distances)
            
            # Add cavity contribution
            output[:, :, point_idx] += attended_cavities[:, nearest_cavity, :]
        
        return output


class TetrahedralMessagePassing(nn.Module):
    """
    Tetrahedral Message Passing Neural Network
    Implements geometric message passing on the tetrahedral grid
    """
    
    def __init__(self, channels: int, grid: TetrahedralGrid, num_layers: int = 3):
        super(TetrahedralMessagePassing, self).__init__()
        self.channels = channels
        self.grid = grid
        self.num_layers = num_layers
        
        # Message passing networks
        self.message_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * 2, channels * 2),
                nn.ReLU(),
                nn.Linear(channels * 2, channels)
            ) for _ in range(num_layers)
        ])
        
        # Update networks
        self.update_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels * 2, channels * 2),
                nn.ReLU(),
                nn.Linear(channels * 2, channels)
            ) for _ in range(num_layers)
        ])
        
        # Edge features
        self.edge_network = nn.Sequential(
            nn.Linear(3, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels // 2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform message passing on the tetrahedral grid
        Args:
            x: Input features of shape [batch_size, channels, num_points]
        Returns:
            Updated features of same shape
        """
        batch_size, channels, num_points = x.shape
        
        # Transpose for easier indexing
        features = x.transpose(1, 2)  # [batch_size, num_points, channels]
        
        for layer_idx in range(self.num_layers):
            new_features = features.clone()
            
            for point_idx in range(num_points):
                # Get neighbors
                neighbors = self.grid.get_neighbors(point_idx)
                
                if len(neighbors) > 0:
                    messages = []
                    
                    for neighbor_idx in neighbors:
                        # Compute edge features
                        edge_vector = self.grid.points[neighbor_idx] - self.grid.points[point_idx]
                        edge_features = self.edge_network(edge_vector)
                        
                        # Create message
                        sender_features = features[:, neighbor_idx, :]
                        receiver_features = features[:, point_idx, :]
                        
                        message_input = torch.cat([sender_features, receiver_features], dim=-1)
                        message_input = torch.cat([message_input, edge_features], dim=-1)
                        
                        message = self.message_networks[layer_idx](message_input)
                        messages.append(message)
                    
                    # Aggregate messages
                    aggregated_message = torch.mean(torch.stack(messages, dim=1), dim=1)
                    
                    # Update features
                    update_input = torch.cat([features[:, point_idx, :], aggregated_message], dim=-1)
                    new_features[:, point_idx, :] = self.update_networks[layer_idx](update_input)
            
            features = new_features
        
        return features.transpose(1, 2)  # [batch_size, channels, num_points]


class SpatialAttentionMechanism(nn.Module):
    """
    Spatial Attention Mechanism for Tetrahedral Grid
    Implements attention that respects the geometric structure
    """
    
    def __init__(self, channels: int, grid: TetrahedralGrid, num_heads: int = 8):
        super(SpatialAttentionMechanism, self).__init__()
        self.channels = channels
        self.grid = grid
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)
        
        # Geometric bias
        self.geo_bias = nn.Parameter(torch.ones(len(grid.points), len(grid.points)))
        
        # Output projection
        self.output_proj = nn.Linear(channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input features
        Args:
            x: Input tensor of shape [batch_size, channels, num_points]
        Returns:
            Attended tensor of same shape
        """
        batch_size, channels, num_points = x.shape
        
        # Transpose for attention computation
        features = x.transpose(1, 2)  # [batch_size, num_points, channels]
        
        # Compute queries, keys, values
        queries = self.query_proj(features)  # [batch_size, num_points, channels]
        keys = self.key_proj(features)      # [batch_size, num_points, channels]
        values = self.value_proj(features)  # [batch_size, num_points, channels]
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, num_points, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add geometric bias
        geo_bias_expanded = self.geo_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, num_points, num_points]
        attention_scores = attention_scores + geo_bias_expanded
        
        # Apply adjacency mask (only attend to neighbors)
        adjacency_mask = self.grid.adjacency_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, num_points, num_points]
        attention_scores = attention_scores.masked_fill(adjacency_mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, values)  # [batch_size, num_heads, num_points, head_dim]
        
        # Reshape and project output
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_points, channels)
        output = self.output_proj(attended)
        
        return output.transpose(1, 2)  # [batch_size, channels, num_points]


class TetrahedralAGINetwork(nn.Module):
    """
    Main 64-Point Tetrahedron AI Network
    Combines all components into a unified architecture
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 hidden_channels: int = 256,
                 output_channels: int = 128,
                 num_conv_layers: int = 4,
                 num_message_passing_layers: int = 3,
                 device: str = 'cuda'):
        super(TetrahedralAGINetwork, self).__init__()
        
        # Initialize tetrahedral grid
        self.grid = TetrahedralGrid(device)
        self.num_points = len(self.grid.points)
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, hidden_channels)
        
        # Tetrahedral convolution layers
        self.conv_layers = nn.ModuleList([
            TetrahedralConvolution(hidden_channels, hidden_channels, self.grid)
            for _ in range(num_conv_layers)
        ])
        
        # Octahedral cavity processor
        self.cavity_processor = OctahedralCavityProcessor(hidden_channels, self.grid)
        
        # Message passing layers
        self.message_passing = TetrahedralMessagePassing(hidden_channels, self.grid, num_message_passing_layers)
        
        # Spatial attention
        self.spatial_attention = SpatialAttentionMechanism(hidden_channels, self.grid)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_channels, output_channels)
        
        # Normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_conv_layers + 2)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete tetrahedral AI network
        Args:
            x: Input tensor of shape [batch_size, input_channels, num_points]
        Returns:
            Output tensor of shape [batch_size, output_channels, num_points]
        """
        batch_size = x.shape[0]
        
        # Ensure input has correct shape
        if x.shape[2] != self.num_points:
            # Interpolate or pad to match grid size
            x = self._adapt_input_to_grid(x)
        
        # Input projection
        x = x.transpose(1, 2)  # [batch_size, num_points, input_channels]
        x = self.input_projection(x)  # [batch_size, num_points, hidden_channels]
        x = x.transpose(1, 2)  # [batch_size, hidden_channels, num_points]
        
        # Apply convolution layers
        for i, conv_layer in enumerate(self.conv_layers):
            residual = x
            x = conv_layer(x)
            x = x.transpose(1, 2)  # [batch_size, num_points, hidden_channels]
            x = self.layer_norms[i](x)
            x = x.transpose(1, 2)  # [batch_size, hidden_channels, num_points]
            x = F.relu(x)
            x = x + residual  # Residual connection
        
        # Apply cavity processing
        x = self.cavity_processor(x)
        
        # Apply message passing
        x = self.message_passing(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Final layer norm
        x = x.transpose(1, 2)
        x = self.layer_norms[-1](x)
        x = x.transpose(1, 2)
        
        # Output projection
        x = x.transpose(1, 2)  # [batch_size, num_points, hidden_channels]
        x = self.output_projection(x)  # [batch_size, num_points, output_channels]
        x = x.transpose(1, 2)  # [batch_size, output_channels, num_points]
        
        return x
    
    def _adapt_input_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt input tensor to match the tetrahedral grid size"""
        batch_size, input_channels, input_points = x.shape
        
        if input_points < self.num_points:
            # Pad with zeros
            padding = torch.zeros(batch_size, input_channels, self.num_points - input_points, device=x.device)
            x = torch.cat([x, padding], dim=2)
        elif input_points > self.num_points:
            # Truncate or interpolate
            x = x[:, :, :self.num_points]
        
        return x
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get the current attention weights for visualization"""
        return self.spatial_attention.geo_bias.detach()
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get information about the tetrahedral grid"""
        return {
            'num_points': self.num_points,
            'num_tetrahedra': len(self.grid.simplices),
            'num_cavities': len(self.grid.octahedral_cavities),
            'volume': self.grid.volume,
            'mean_radius': self.grid.mean_radius.item(),
            'std_radius': self.grid.std_radius.item()
        }


if __name__ == "__main__":
    # Test the network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create network
    net = TetrahedralAGINetwork(device=device)
    net.to(device)
    
    # Create dummy input
    batch_size = 4
    input_channels = 3
    dummy_input = torch.randn(batch_size, input_channels, 64, device=device)
    
    # Forward pass
    output = net(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Grid info: {net.get_grid_info()}")