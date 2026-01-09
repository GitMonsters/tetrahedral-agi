"""
API Gateway and Deployment System for 64-Point Tetrahedron AI
Provides RESTful APIs and deployment infrastructure
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import torch
import numpy as np
import asyncio
import uuid
import json
import time
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from ..neural_network.tetrahedral_network import TetrahedralAGINetwork
from ..training.trainer import TetrahedralTrainer, TrainingConfig
from ..geometry.tetrahedral_grid import TetrahedralGrid


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    points: List[List[float]] = Field(..., description="3D points as [[x,y,z], ...]")
    features: Optional[List[List[float]]] = Field(None, description="Optional features for each point")
    model_id: Optional[str] = Field("default", description="Model identifier to use")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[List[float]] = Field(..., description="Model predictions")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class TrainingRequest(BaseModel):
    """Request model for training"""
    config: Dict[str, Any] = Field(..., description="Training configuration")
    data_path: str = Field(..., description="Path to training data")
    model_id: Optional[str] = Field(None, description="Model ID for training")


class TrainingStatus(BaseModel):
    """Training status response"""
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Training status")
    progress: float = Field(..., description="Training progress (0-1)")
    current_epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    metrics: Dict[str, float] = Field(..., description="Current metrics")


class ModelInfo(BaseModel):
    """Model information response"""
    model_id: str = Field(..., description="Model identifier")
    architecture: str = Field(..., description="Architecture type")
    parameters: int = Field(..., description="Number of parameters")
    grid_info: Dict[str, Any] = Field(..., description="Tetrahedral grid information")
    performance: Dict[str, float] = Field(..., description="Performance metrics")


class ModelManager:
    """Manages multiple model instances"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.models: Dict[str, TetrahedralAGINetwork] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.default_model_id = "default"
        
        # Initialize default model
        self._load_default_model()
    
    def _load_default_model(self):
        """Load the default model"""
        try:
            model = TetrahedralAGINetwork(device=self.device)
            model.eval()
            self.models[self.default_model_id] = model
            self.model_configs[self.default_model_id] = {
                'input_channels': 3,
                'hidden_channels': 256,
                'output_channels': 128,
                'device': self.device
            }
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load default model: {e}")
    
    def get_model(self, model_id: str) -> TetrahedralAGINetwork:
        """Get a model by ID"""
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return self.models[model_id]
    
    def add_model(self, model_id: str, model: TetrahedralAGINetwork, config: Dict):
        """Add a new model"""
        model.eval()
        self.models[model_id] = model
        self.model_configs[model_id] = config
        logger.info(f"Model {model_id} added successfully")
    
    def remove_model(self, model_id: str):
        """Remove a model"""
        if model_id in self.models:
            del self.models[model_id]
            del self.model_configs[model_id]
            logger.info(f"Model {model_id} removed")
    
    def list_models(self) -> List[str]:
        """List all available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a model"""
        model = self.get_model(model_id)
        config = self.model_configs[model_id]
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Get grid info
        grid_info = model.get_grid_info()
        
        return ModelInfo(
            model_id=model_id,
            architecture="64-Point Tetrahedron AI",
            parameters=num_params,
            grid_info=grid_info,
            performance={
                'inference_latency': 0.01,  # Placeholder
                'memory_usage': 1000,      # Placeholder
                'accuracy': 0.95           # Placeholder
            }
        )


class TrainingJobManager:
    """Manages training jobs"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
    
    def create_job(self, config: TrainingConfig, data_path: str) -> str:
        """Create a new training job"""
        job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            'id': job_id,
            'config': config,
            'data_path': data_path,
            'status': 'created',
            'progress': 0.0,
            'current_epoch': 0,
            'total_epochs': config.num_epochs,
            'metrics': {},
            'start_time': None,
            'end_time': None,
            'error': None
        }
        
        return job_id
    
    async def start_training(self, job_id: str):
        """Start training for a job"""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = self.jobs[job_id]
        job['status'] = 'running'
        job['start_time'] = time.time()
        
        try:
            # Create model and trainer
            model = TetrahedralAGINetwork(device=job['config'].device)
            trainer = TetrahedralTrainer(model, job['config'])
            
            # Setup data
            from ..training.trainer import GeometricDataset
            dataset = GeometricDataset(job['data_path'])
            trainer.setup_data_loaders(dataset)
            
            # Custom training loop with progress updates
            for epoch in range(1, job['config'].num_epochs + 1):
                # Update progress
                job['current_epoch'] = epoch
                job['progress'] = epoch / job['config'].num_epochs
                
                # Train epoch
                train_losses = trainer.train_epoch(epoch)
                
                # Validate
                val_losses = trainer.validate()
                
                # Update metrics
                job['metrics'] = {
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss'],
                    'learning_rate': trainer.optimizer.param_groups[0]['lr']
                }
                
                # Check if job should be cancelled
                if job['status'] == 'cancelled':
                    break
            
            # Mark as completed
            job['status'] = 'completed'
            job['end_time'] = time.time()
            
            # Save the trained model
            model_path = f"./models/{job_id}.pth"
            Path("./models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
        except Exception as e:
            job['status'] = 'failed'
            job['error'] = str(e)
            job['end_time'] = time.time()
            logger.error(f"Training job {job_id} failed: {e}")
    
    def get_job_status(self, job_id: str) -> TrainingStatus:
        """Get the status of a training job"""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = self.jobs[job_id]
        
        return TrainingStatus(
            job_id=job['id'],
            status=job['status'],
            progress=job['progress'],
            current_epoch=job['current_epoch'],
            total_epochs=job['total_epochs'],
            metrics=job['metrics']
        )
    
    def cancel_job(self, job_id: str):
        """Cancel a training job"""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = self.jobs[job_id]
        if job['status'] == 'running':
            job['status'] = 'cancelled'
            job['end_time'] = time.time()


# Global instances
model_manager = ModelManager()
training_manager = TrainingJobManager()


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting Tetrahedral AI API Gateway")
    yield
    # Shutdown
    logger.info("Shutting down Tetrahedral AI API Gateway")


# Create FastAPI app
app = FastAPI(
    title="64-Point Tetrahedron AI API",
    description="API Gateway for 64-Point Tetrahedron AI Model Architecture Platform",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "64-Point Tetrahedron AI API Gateway",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "models_loaded": len(model_manager.models),
        "active_training_jobs": len(training_manager.active_jobs)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the tetrahedral AI model"""
    try:
        start_time = time.time()
        
        # Get model
        model = model_manager.get_model(request.model_id)
        
        # Convert input to tensor
        points = torch.tensor(request.points, dtype=torch.float32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dimension
        
        # Handle features
        if request.features:
            features = torch.tensor(request.features, dtype=torch.float32)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            
            # Combine points and features
            input_data = torch.cat([points, features], dim=-1)
        else:
            input_data = points
        
        # Transpose for model input
        input_data = input_data.transpose(1, 2)  # [batch_size, channels, num_points]
        
        # Move to device
        input_data = input_data.to(model_manager.device)
        
        # Make prediction
        with torch.no_grad():
            model.eval()
            predictions = model(input_data)
        
        # Convert back to list
        predictions = predictions.transpose(1, 2).cpu().numpy().tolist()
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            predictions=predictions,
            processing_time=processing_time,
            model_info=model_manager.get_model_info(request.model_id).dict()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=Dict[str, str])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training job"""
    try:
        # Create training config
        config = TrainingConfig(**request.config)
        
        # Create job
        job_id = training_manager.create_job(config, request.data_path)
        
        # Start training in background
        background_tasks.add_task(training_manager.start_training, job_id)
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status"""
    return training_manager.get_job_status(job_id)


@app.delete("/train/{job_id}")
async def cancel_training(job_id: str):
    """Cancel a training job"""
    training_manager.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/models", response_model=List[str])
async def list_models():
    """List all available models"""
    return model_manager.list_models()


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    return model_manager.get_model_info(model_id)


@app.post("/models/{model_id}/load")
async def load_model(model_id: str, model_path: str):
    """Load a model from file"""
    try:
        # Create new model
        model = TetrahedralAGINetwork(device=model_manager.device)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=model_manager.device)
        model.load_state_dict(state_dict)
        model.eval()
        
        # Add to manager
        config = {
            'input_channels': 3,
            'hidden_channels': 256,
            'output_channels': 128,
            'device': model_manager.device
        }
        model_manager.add_model(model_id, model, config)
        
        return {"model_id": model_id, "status": "loaded"}
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}")
async def unload_model(model_id: str):
    """Unload a model"""
    model_manager.remove_model(model_id)
    return {"model_id": model_id, "status": "unloaded"}


@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    return {
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "models_loaded": len(model_manager.models),
        "active_training_jobs": len(training_manager.active_jobs),
        "memory_usage": {
            "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            "cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        }
    }


# WebSocket for real-time training updates
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


connection_manager = ConnectionManager()


@app.websocket("/ws/training/{job_id}")
async def websocket_training_updates(websocket: WebSocket, job_id: str):
    """WebSocket for real-time training updates"""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Get job status
            status = training_manager.get_job_status(job_id)
            
            # Send status update
            await websocket.send_text(json.dumps(status.dict()))
            
            # Check if training is complete
            if status.status in ['completed', 'failed', 'cancelled']:
                break
            
            # Wait before next update
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )