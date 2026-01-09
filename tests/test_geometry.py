"""
Test the tetrahedral grid geometry functionality
"""

import pytest
import torch
import numpy as np
from geometry.tetrahedral_grid import TetrahedralGrid, GeometricOperations


class TestTetrahedralGrid:
    """Test cases for TetrahedralGrid class"""
    
    def test_grid_creation(self):
        """Test basic grid creation"""
        grid = TetrahedralGrid(device='cpu')
        
        assert grid.num_points == 64
        assert len(grid.points) == 64
        assert len(grid.octahedral_cavities) == 14
        assert grid.adjacency_matrix.shape == (64, 64)
    
    def test_tetrahedron_geometry(self):
        """Test tetrahedron geometry computation"""
        grid = TetrahedralGrid(device='cpu')
        
        if len(grid.simplices) > 0:
            tetrahedron = grid.get_tetrahedron_geometry(0)
            
            assert tetrahedron.vertices.shape == (4, 3)
            assert tetrahedron.center.shape == (3,)
            assert tetrahedron.volume > 0
            assert tetrahedron.face_normals.shape == (4, 3)
            assert tetrahedron.circumradius > 0
            assert tetrahedron.inradius > 0
            assert tetrahedron.circumradius > tetrahedron.inradius
    
    def test_neighbors(self):
        """Test neighbor computation"""
        grid = TetrahedralGrid(device='cpu')
        
        for i in range(min(10, len(grid.points))):
            neighbors = grid.get_neighbors(i)
            assert isinstance(neighbors, list)
            assert all(isinstance(n, int) for n in neighbors)
    
    def test_octahedral_cavities(self):
        """Test octahedral cavity access"""
        grid = TetrahedralGrid(device='cpu')
        
        for i in range(len(grid.octahedral_cavities)):
            cavity = grid.get_octahedral_cavity(i)
            assert cavity.shape == (3,)
            assert torch.isfinite(cavity).all()


class TestGeometricOperations:
    """Test cases for GeometricOperations class"""
    
    def test_barycentric_coordinates(self):
        """Test barycentric coordinate computation"""
        grid = TetrahedralGrid(device='cpu')
        
        if len(grid.simplices) > 0:
            tetrahedron = grid.get_tetrahedron_geometry(0)
            center = tetrahedron.center
            
            bary_coords = GeometricOperations.barycentric_coordinates(center, tetrahedron)
            
            assert bary_coords.shape == (4,)
            assert torch.allclose(torch.sum(bary_coords), torch.tensor(1.0), atol=1e-6)
            assert torch.all(bary_coords >= 0)
            assert torch.all(bary_coords <= 1)
    
    def test_spherical_harmonics(self):
        """Test spherical harmonics computation"""
        point = torch.tensor([1.0, 0.0, 0.0])
        
        harmonics = GeometricOperations.spherical_harmonics(point, degree=2)
        
        assert harmonics.shape[0] == (2 + 1) ** 2
        assert torch.isfinite(harmonics).all()


if __name__ == "__main__":
    pytest.main([__file__])