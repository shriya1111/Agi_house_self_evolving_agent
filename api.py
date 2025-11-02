#!/usr/bin/env python3
"""FastAPI backend for self-evolving image generation agent."""
import sys
import os
import base64
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.orchestration.orchestrator import AgentOrchestrator
from src.observability.metrics import MetricsLogger
from src.config import config
from PIL import Image

app = FastAPI(title="Self-Evolving Image Generation Agent", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images) from outputs directory
outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)
try:
    app.mount("/images", StaticFiles(directory=str(outputs_dir)), name="images")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")


class ProcessingResponse(BaseModel):
    success: bool
    message: str
    transcription: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    scenes: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None
    virality_scores: Optional[List[Dict[str, Any]]] = None
    optimizations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string."""
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            # Determine image format
            img_format = image_path.suffix[1:].upper() if image_path.suffix else 'PNG'
            return f"data:image/{img_format.lower()};base64,{img_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def get_all_images(output_dir: Path) -> List[Dict[str, Any]]:
    """Get all generated images from output directory."""
    images = []
    
    # Look for all image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    for ext in image_extensions:
        for img_path in output_dir.glob(f"*{ext}"):
            # Convert to base64
            img_base64 = image_to_base64(img_path)
            if img_base64:
                images.append({
                    'filename': img_path.name,
                    'url': f"/images/{img_path.name}",
                    'base64': img_base64,
                    'path': str(img_path)
                })
    
    return sorted(images, key=lambda x: x['filename'])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Self-Evolving Image Generation Agent API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/process-audio", response_model=ProcessingResponse)
async def process_audio(
    audio_file: UploadFile = File(...),
    output_dir: str = Form(default="outputs")
):
    """
    Process audio file and generate images.
    
    Args:
        audio_file: Audio file (mp3, m4a, wav, etc.)
        output_dir: Output directory for generated images
        
    Returns:
        ProcessingResponse with transcription, scenes, images, and metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file temporarily
    temp_audio = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.filename).suffix) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_audio = temp_file.name
        
        print(f"Processing audio file: {audio_file.filename}")
        
        # Initialize orchestrator and agents (same as main.py)
        orchestrator = AgentOrchestrator()
        
        # Register agents
        from src.audio.transcriber import AudioTranscriber
        from src.context.extractor import ContextExtractor
        from src.generation.image_generator import ImageGenerator
        from src.evaluation.virality_evaluator import ViralityEvaluator
        from src.rl.prompt_optimizer import RLPromptOptimizer
        
        try:
            orchestrator.register_agent('transcription', AudioTranscriber(use_wandb=True))
            orchestrator.register_agent('context', ContextExtractor(use_wandb=True))
            orchestrator.register_agent('generation', ImageGenerator(use_wandb=True))
            orchestrator.register_agent('evaluation', ViralityEvaluator(use_wandb=True))
            orchestrator.register_agent('rl', RLPromptOptimizer(use_wandb=True))
        except Exception as e:
            print(f"Error registering agents: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Execute pipeline - use async workflow directly (FastAPI is async)
        results = await orchestrator.execute_workflow(temp_audio, str(output_path))
        
        # Check if processing was successful
        if not results.get('success', False):
            error_msg = results.get('error', 'Unknown error occurred')
            return ProcessingResponse(
                success=False,
                message="Processing failed",
                error=error_msg
            )
        
        # Get all generated images
        images = get_all_images(output_path)
        
        # Extract transcription text safely
        transcription_text = ''
        if results.get('transcription'):
            if isinstance(results['transcription'], dict):
                transcription_text = results['transcription'].get('text', '')
            else:
                transcription_text = str(results['transcription'])
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        # Prepare response
        response_data = {
            'success': True,
            'message': 'Processing completed successfully',
            'transcription': transcription_text,
            'context': convert_types(results.get('context', {})),
            'scenes': convert_types(results.get('scenes', [])),
            'images': images,
            'virality_scores': convert_types(results.get('virality_scores', [])),
            'optimizations': convert_types(results.get('optimizations', []))
        }
        
        return ProcessingResponse(**response_data)
        
    except Exception as e:
        error_msg = f"Error processing audio: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return ProcessingResponse(
            success=False,
            message="Processing failed",
            error=error_msg
        )
    
    finally:
        # Clean up temporary file
        if temp_audio and Path(temp_audio).exists():
            try:
                Path(temp_audio).unlink()
            except Exception as e:
                print(f"Error deleting temp file: {e}")


@app.get("/api/images")
async def list_images(output_dir: str = "outputs"):
    """List all generated images."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {"images": []}
    
    images = get_all_images(output_path)
    return {"images": images, "count": len(images)}


@app.get("/api/images/{image_name}")
async def get_image(image_name: str, output_dir: str = "outputs"):
    """Get a specific image file."""
    output_path = Path(output_dir) / image_name
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        output_path,
        media_type=f"image/{output_path.suffix[1:]}"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

