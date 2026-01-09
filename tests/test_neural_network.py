"""
Test tetrahedral neural network functionality
"""

import pytest
import torch
import numpy as np
from neural_network.tetrahedral_network import (
    TetrahedralAGINetwork,
    TetrahedralConvolution,
    OctahedralCavityProcessor,
    TetrahedralMessagePassing,
    SpatialAttentionMechanism
)
from geometry.tetrahedral_grid import TetrahedralGrid


class TestTetrahedralAGINetwork:
    """Test cases for TetrahedralAGINetwork"""
    
    def test_model_creation(self):
        """Test model initialization"""
        model = TetrahedralAGINetwork(device='cpu')
        
        assert model.num_points == 64
        assert model.hidden_channels == 256
        assert sum(p.numel() for p in model.parameters()) > 0
    
    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        model = TetrahedralAGINetwork(device='cpu')
        
        batch_size = 2
        input_channels = 3
        dummy_input = torch.randn(batch_size, input_channels, 64)
        
        output = model(dummy_input)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 128  # output_channels
        assert output.shape[2] == 64
    
    def test_grid_info(self):
        """Test grid information extraction"""
        model = TetrahedralAGINetwork(device='cpu')
        grid_info = model.get_grid_info()
        
        assert 'num_points' in grid_info
        assert 'num_tetrahedra' in grid_info
        assert 'num_cavities' in grid_info
        assert 'volume' in grid_info
        assert grid_info['num_points'] == 64
        assert grid_info['num_cavities'] == 14


class TestTetrahedralConvolution:
    """Test cases for TetrahedralConvolution layer"""
    
    def test_convolution_creation(self):
        """Test convolution layer initialization"""
        grid = TetrahedralGrid(device='cpu')
        conv = TetrahedralConvolution(3, 64, grid)
        
        assert conv.in_channels == 3
        assert conv.out_channels == 64
        assert conv.weight.shape == (64, 3, 4)
        assert conv.bias.shape == (64,)
    
    def test_convolution_forward(self):
        """Test convolution forward pass"""
        grid = TetrahedralGrid(device='cpu')
        conv = TetrahedralConvolution(3, 64, grid)
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 64)
        
        output = conv(dummy_input)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 64
        assert output.shape[2] == 64


class TestOctahedralCavityProcessor:
    """Test cases for OctahedralCavityProcessor"""
    
    def test_processor_creation(self):
        """Test processor initialization"""
        grid = TetrahedralGrid(device='cpu')
        processor = OctahedralCavityProcessor(128, grid)
        
        assert processor.channels == 128
        assert processor.num_cavities == 14
        assert len(processor.cavity_networks) == 14
    
    def test_processor_forward(self):
        """Test processor forward pass"""
        grid = TetrahedralGrid(device='cpu')
        processor = OctahedralCavityProcessor(128, grid)
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 128, 64)
        
        output = processor(dummy_input)
        
        assert output.shape == dummy_input.shape


class TestTetrahedralMessagePassing:
    """Test cases for TetrahedralMessagePassing"""
    
    def test_message_passing_creation(self):
        """Test message passing initialization"""
        grid = TetrahedralGrid(device='cpu')
        mp = TetrahedralMessagePassing(64, grid, num_layers=2)
        
        assert mp.channels == 64
        assert mp.num_layers == 2
        assert len(mp.message_networks) == 2
        assert len(mp.update_networks) == 2
    
    def test_message_passing_forward(self):
        """Test message passing forward pass"""
        grid = TetrahedralGrid(device='cpu')
        mp = TetrahedralMessagePassing(64, grid, num_layers=2)
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 64, 64)
        
        output = mp(dummy_input)
        
        assert output.shape == dummy_input.shape


class TestSpatialAttentionMechanism:
    """Test cases for SpatialAttentionMechanism"""
    
    def test_attention_creation(self):
        """Test attention mechanism initialization"""
        grid = TetrahedralGrid(device='cpu')
        attention = SpatialAttentionMechanism(128, grid, num_heads=4)
        
        assert attention.channels == 128
        assert attention.num_heads == 4
        assert attention.head_dim == 32
    
    def test_attention_forward(self):
        """Test attention forward pass"""
        grid = TetrahedralGrid(device='cpu')
        attention = SpatialAttentionMechanism(128, grid, num_heads=4)
        
        batch_size = 2
        dummy_input = torch.randn(batch_size, 128, 64)
        
        output = attention(dummy_input)
        
        assert output.shape == dummy_input.shape


if __name__ == "__main__":
    pytest.main([__file__])