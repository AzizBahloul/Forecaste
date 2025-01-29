from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import os
import logging
from prediction_service import PredictionService
import numpy as np
from typing import Dict, Any
import pandas as pd  # Add pandas import
from datetime import datetime  # Add datetime import
from llm import analyzer  # Add llm import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger('main')

# Define model directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

class PredictionInput(BaseModel):
    base_price: float = Field(..., gt=0, description="Base price must be greater than 0")
    total_price: float = Field(..., gt=0, description="Total price must be greater than 0")
    is_featured_sku: int = Field(..., ge=0, le=1, description="Must be 0 or 1")
    is_display_sku: int = Field(..., ge=0, le=1, description="Must be 0 or 1")
    sku_id: int = Field(..., gt=0, description="SKU ID must be greater than 0")

    @validator('total_price')
    def total_price_must_be_greater_than_base_price(cls, v, values):
        if 'base_price' in values and v < values['base_price']:
            raise ValueError('total_price must be greater than or equal to base_price')
        return v

class PredictionOutput(BaseModel):
    predicted_units: int = Field(..., ge=0, description="Predicted units to be sold")
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "confidence_score": None,
            "input_data": None
        },
        description="Prediction metadata including model version and timestamp"
    )

    class Config:
        schema_extra = {
            "example": {
                "predicted_units": 49,
                "metadata": {
                    "model_version": "1.0.0",
                    "timestamp": "2025-01-29T00:09:27.544675",
                    "confidence_score": 0.85,
                    "input_data": {
                        "base_price": 100.0,
                        "total_price": 120.0,
                        "is_featured_sku": 1,
                        "is_display_sku": 1,
                        "sku_id": 9632
                    }
                }
            }
        }

# Global service instance
prediction_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global prediction_service
    try:
        prediction_service = PredictionService(MODEL_DIR)  # Changed from BASE_DIR to MODEL_DIR
        logger.info("PredictionService initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize PredictionService: {e}")
        if prediction_service:
            logger.info("PredictionService shutdown complete")
        raise RuntimeError("Application failed to start") from e
    finally:
        if prediction_service:
            logger.info("PredictionService shutdown complete")

app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make prediction based on input data"""
    global prediction_service
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Prediction service not ready")

    try:
        logger.info(f"Processing prediction request: {input_data.dict()}")
        prediction = prediction_service.predict(
            base_price=input_data.base_price,
            total_price=input_data.total_price,
            is_featured_sku=input_data.is_featured_sku,
            is_display_sku=input_data.is_display_sku,
            sku_id=input_data.sku_id
        )
        
        response = PredictionOutput(
            predicted_units=prediction,
            metadata={
                "model_version": getattr(prediction_service, "model_version", "unknown"),
                "timestamp": datetime.now().isoformat(),
                "input_data": input_data.dict()
            }
        )
        
        # Add LLM analysis
        analysis = analyzer.analyze_prediction(response.dict())
        response.metadata["analysis"] = analysis
        
        logger.info(f"Prediction complete: {response.dict()}")
        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not prediction_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        # Verify model is responsive
        prediction_service._verify_model()
        return {
            "status": "healthy",
            "model_version": getattr(prediction_service, "model_version", "unknown")
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
