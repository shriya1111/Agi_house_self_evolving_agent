"""Image generation using Stable Diffusion or Replicate API."""
import sys
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np
import time
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.observability.metrics import MetricsLogger

# Optional Replicate import
try:
    import replicate
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False

# Optional diffusers import (only for local generation)
try:
    from diffusers import StableDiffusionXLPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class ImageGenerator:
    """Generates images using Stable Diffusion XL."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize image generator."""
        self.provider = config.get('models.image_generation.provider', 'replicate')
        self.model_name = config.get('models.image_generation.model', 'stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b')
        self.num_inference_steps = config.get('models.image_generation.num_inference_steps', 25)
        self.guidance_scale = config.get('models.image_generation.guidance_scale', 7.5)
        self.image_size = config.get('models.image_generation.image_size', [1024, 1024])
        self.use_wandb = use_wandb
        self.metrics = MetricsLogger() if use_wandb else None
        
        # Check for GPU (only needed for local)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.replicate_client = None
        self.dalle_client = None
        self.skip_generation = False
        
        if self.provider == 'local':
            if not DIFFUSERS_AVAILABLE:
                raise ValueError("Diffusers not available. Install with: pip install diffusers")
            self._load_model()
        elif self.provider == 'replicate':
            if not REPLICATE_AVAILABLE:
                raise ValueError("Replicate not available. Install with: pip install replicate")
            self._setup_replicate()
        elif self.provider == 'dall-e':
            # Use OpenAI DALL-E (already have API key)
            self._setup_dalle()
        elif self.provider == 'skip':
            # Skip image generation entirely
            print("Image generation disabled (skip mode)")
            self.skip_generation = True
        else:
            raise ValueError(f"Provider {self.provider} not supported. Use 'local', 'replicate', 'dall-e', or 'skip'.")
    
    def _load_model(self):
        """Load Stable Diffusion model for local generation."""
        try:
            print(f"Loading model {self.model_name} on {self.device}...")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                # Enable memory efficient attention
                self.pipe.enable_attention_slicing()
            
            print(f"Model loaded successfully on {self.device}")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.device = "cpu"
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
    
    def _setup_replicate(self):
        """Setup Replicate API client."""
        try:
            api_key = config.get_api_key('replicate')
            if not api_key:
                # Try environment variable
                api_key = os.getenv('REPLICATE_API_TOKEN')
            
            if api_key:
                os.environ['REPLICATE_API_TOKEN'] = api_key
                print(f"Replicate API configured (provider: {self.provider})")
            else:
                print("Warning: REPLICATE_API_TOKEN not found. Replicate may not work without it.")
            
            self.replicate_client = replicate.Client(api_token=api_key) if api_key else None
        except Exception as e:
            print(f"Warning: Replicate setup issue: {e}")
            self.replicate_client = None
    
    def _setup_dalle(self):
        """Setup OpenAI DALL-E (uses existing OpenAI API key)."""
        try:
            from openai import OpenAI
            api_key = config.get_api_key('openai')
            if not api_key:
                raise ValueError("OpenAI API key not found (needed for DALL-E)")
            
            self.dalle_client = OpenAI(api_key=api_key)
            print(f"DALL-E API configured (provider: {self.provider}, using OpenAI key)")
        except Exception as e:
            print(f"Warning: DALL-E setup issue: {e}")
            self.dalle_client = None
    
    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate an image from a prompt.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt (things to avoid)
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            
        Returns:
            Dictionary with image and metadata
        """
        start_time = time.time()
        
        if hasattr(self, 'skip_generation') and self.skip_generation:
            return {
                'success': False,
                'error': 'Image generation skipped',
                'prompt': prompt,
                'skipped': True
            }
        elif self.provider == 'replicate':
            return self._generate_with_replicate(prompt, negative_prompt, num_inference_steps, guidance_scale, start_time)
        elif self.provider == 'dall-e':
            return self._generate_with_dalle(prompt, start_time)
        elif self.provider == 'local':
            return self._generate_local(prompt, negative_prompt, num_inference_steps, guidance_scale, start_time)
        else:
            return {
                'success': False,
                'error': f'Unknown provider: {self.provider}'
            }
    
    def _generate_with_replicate(
        self, 
        prompt: str,
        negative_prompt: Optional[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """Generate image using Replicate API."""
        try:
            import replicate
            
            # Use provided parameters or defaults
            steps = num_inference_steps or self.num_inference_steps
            guidance = guidance_scale or self.guidance_scale
            neg_prompt = negative_prompt or "blurry, low quality, distorted, watermark"
            
            print(f"Generating image with Replicate (model: {self.model_name})...")
            
            # Call Replicate API
            output = replicate.run(
                self.model_name,
                input={
                    "prompt": prompt,
                    "negative_prompt": neg_prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "width": self.image_size[1],
                    "height": self.image_size[0]
                }
            )
            
            # Download image from URL
            image_url = output[0] if isinstance(output, list) else output
            import requests
            from io import BytesIO
            
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            
            generation_time = time.time() - start_time
            
            # Log metrics
            if self.metrics:
                self.metrics.log_image_generation_metrics({
                    'generation_time': generation_time,
                    'prompt_length': len(prompt),
                    'num_inference_steps': steps,
                    'guidance_scale': guidance,
                    'image_size': self.image_size,
                    'device': 'replicate-api',
                    'model': self.model_name
                })
            
            return {
                'image': image,
                'prompt': prompt,
                'negative_prompt': neg_prompt,
                'generation_time': generation_time,
                'num_inference_steps': steps,
                'guidance_scale': guidance,
                'device': 'replicate-api',
                'provider': 'replicate',
                'success': True
            }
        
        except Exception as e:
            error_msg = f"Replicate generation failed: {str(e)}"
            print(f"Error: {error_msg}")
            
            if self.metrics:
                self.metrics.log_error('image_generation', error_msg)
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_with_dalle(
        self,
        prompt: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Generate image using OpenAI DALL-E 3."""
        try:
            if not self.dalle_client:
                self._setup_dalle()
            
            if not self.dalle_client:
                raise ValueError("DALL-E client not initialized")
            
            print(f"Generating image with DALL-E 3...")
            
            # Call DALL-E API
            response = self.dalle_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=f"{self.image_size[1]}x{self.image_size[0]}",
                quality="standard",
                n=1,
            )
            
            # Get image URL
            image_url = response.data[0].url
            
            # Download image
            import requests
            from io import BytesIO
            
            img_response = requests.get(image_url)
            image = Image.open(BytesIO(img_response.content))
            
            generation_time = time.time() - start_time
            
            # Log metrics
            if self.metrics:
                self.metrics.log_image_generation_metrics({
                    'generation_time': generation_time,
                    'prompt_length': len(prompt),
                    'num_inference_steps': 0,  # DALL-E doesn't use steps
                    'guidance_scale': 0,  # DALL-E doesn't use guidance
                    'image_size': self.image_size,
                    'device': 'dall-e-api',
                    'model': 'dall-e-3'
                })
            
            return {
                'image': image,
                'prompt': prompt,
                'generation_time': generation_time,
                'device': 'dall-e-api',
                'provider': 'dall-e',
                'success': True
            }
        
        except Exception as e:
            error_msg = f"DALL-E generation failed: {str(e)}"
            print(f"Error: {error_msg}")
            
            if self.metrics:
                self.metrics.log_error('image_generation', error_msg)
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_local(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        start_time: float
    ) -> Dict[str, Any]:
        """Generate image using local Stable Diffusion."""
        if not self.pipe:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Use provided parameters or defaults
            steps = num_inference_steps or self.num_inference_steps
            guidance = guidance_scale or self.guidance_scale
            neg_prompt = negative_prompt or "blurry, low quality, distorted, watermark"
            
            # Generate image
            image = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=self.image_size[0],
                width=self.image_size[1]
            ).images[0]
            
            generation_time = time.time() - start_time
            
            # Log metrics
            if self.metrics:
                self.metrics.log_image_generation_metrics({
                    'generation_time': generation_time,
                    'prompt_length': len(prompt),
                    'num_inference_steps': steps,
                    'guidance_scale': guidance,
                    'image_size': self.image_size,
                    'device': self.device,
                    'model': self.model_name
                })
                
                # Log GPU usage if available
                if self.device == "cuda":
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    self.metrics.log_gpu_metrics({
                        'gpu_memory_used_gb': gpu_memory,
                        'gpu_memory_max_gb': torch.cuda.max_memory_allocated() / 1024**3
                    })
            
            return {
                'image': image,
                'prompt': prompt,
                'negative_prompt': neg_prompt,
                'generation_time': generation_time,
                'num_inference_steps': steps,
                'guidance_scale': guidance,
                'device': self.device,
                'provider': 'local',
                'success': True
            }
        
        except Exception as e:
            error_msg = f"Image generation failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('image_generation', error_msg)
            
            return {
                'success': False,
                'error': error_msg
            }
    
    def generate_scene_frames(
        self, 
        start_prompt: str, 
        end_prompt: str,
        scene_number: int
    ) -> Dict[str, Any]:
        """
        Generate start and end frames for a scene.
        
        Args:
            start_prompt: Prompt for start frame
            end_prompt: Prompt for end frame
            scene_number: Scene number for identification
            
        Returns:
            Dictionary with both frames and metadata
        """
        start_frame = self.generate_image(start_prompt)
        end_frame = self.generate_image(end_prompt)
        
        return {
            'scene_number': scene_number,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'success': start_frame.get('success', False) and end_frame.get('success', False)
        }
    
    def save_image(self, image: Image.Image, output_path: str):
        """Save image to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Image saved to {output_path}")

