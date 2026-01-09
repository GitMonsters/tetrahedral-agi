#!/usr/bin/env python3
"""Quick training test - 3 epochs to verify everything works"""

import sys
import torch
from pathlib import Path

# Modify training config before importing
import gaia_training
gaia_training.TrainingConfig.num_epochs = 3  # Override to 3 epochs
gaia_training.TrainingConfig.batch_size = 16  # Larger batch for speed

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK TRAINING TEST (3 epochs)")
    print("=" * 60)
    
    # Check device
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal) GPU available")
    elif torch.cuda.is_available():
        print("✅ CUDA GPU available")
    else:
        print("⚠️ Using CPU (slower)")
    
    # Run main training
    gaia_training.main()
