"""Video generation using OpenAI or other services."""
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.observability.metrics import MetricsLogger

class VideoGenerator:
    """Generates short videos from prompts using OpenAI or other services."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize video generator."""
        self.provider = config.get('video_generation.provider', 'openai')
        self.use_wandb = use_wandb
        self.metrics = MetricsLogger() if use_wandb else None
        
        if self.provider == 'openai':
            api_key = config.get_api_key('openai')
            if not api_key:
                raise ValueError("OpenAI API key not found (needed for video generation)")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == 'replicate':
            # Replicate has video generation models
            try:
                import replicate
                self.replicate_client = replicate
                self.client = None
            except ImportError:
                raise ValueError("Replicate not available. Install with: pip install replicate")
        else:
            raise ValueError(f"Provider {self.provider} not supported")
    
    def generate_video(
        self,
        prompt: str,
        duration: int = 4,  # seconds, max 4 for OpenAI
        output_path: str = "outputs/video.mp4"
    ) -> Dict[str, Any]:
        """
        Generate a short video from a prompt.
        
        Args:
            prompt: Video generation prompt
            duration: Video duration in seconds (max 4 for OpenAI)
            output_path: Where to save the video
            
        Returns:
            Dictionary with video path and metadata
        """
        start_time = time.time()
        
        if self.provider == 'openai':
            return self._generate_with_openai(prompt, duration, output_path, start_time)
        elif self.provider == 'replicate':
            return self._generate_with_replicate(prompt, duration, output_path, start_time)
        else:
            return {
                'success': False,
                'error': f'Unknown provider: {self.provider}'
            }
    
    def _generate_with_openai(
        self,
        prompt: str,
        duration: int,
        output_path: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Generate video using OpenAI API."""
        try:
            print(f"Generating video with OpenAI...")
            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Duration: {duration} seconds")
            
            # Try OpenAI video generation API
            try:
                # Check if OpenAI has video generation API available
                # Try the videos API endpoint
                response = self.client.videos.generate(
                    model="sora-1.0" if duration <= 4 else "sora-1.0",  # Sora supports up to 4 seconds
                    prompt=prompt,
                    duration=duration,
                    size="1024x1024"
                )
                
                # Get video URL and download
                video_url = response.data[0].video_url if hasattr(response.data[0], 'video_url') else response.data[0].url
                
                import requests
                video_response = requests.get(video_url)
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(video_response.content)
                
                generation_time = time.time() - start_time
                
                if self.metrics:
                    self.metrics.log_image_generation_metrics({
                        'generation_time': generation_time,
                        'prompt_length': len(prompt),
                        'duration': duration,
                        'model': 'sora-1.0',
                        'provider': 'openai'
                    })
                
                return {
                    'success': True,
                    'video_path': output_path,
                    'prompt': prompt,
                    'duration': duration,
                    'generation_time': generation_time,
                    'provider': 'openai-sora'
                }
                
            except AttributeError:
                # API doesn't have videos.generate - Sora not available
                print("⚠ OpenAI Sora API is not yet available in your account.")
                print("   Using optimized frame-based generation...")
                return self._generate_video_from_frames(prompt, duration, output_path, start_time)
            except Exception as e:
                error_msg = str(e)
                if "videos" in error_msg.lower() or "sora" in error_msg.lower():
                    print("⚠ OpenAI Sora API is not yet available.")
                    print("   Using optimized frame-based generation...")
                    return self._generate_video_from_frames(prompt, duration, output_path, start_time)
                else:
                    raise e
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Video generation failed: {str(e)}"
            }
    
    def _generate_with_replicate(
        self,
        prompt: str,
        duration: int,
        output_path: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Generate video using Replicate API."""
        try:
            import replicate
            
            print(f"Generating video with Replicate...")
            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Duration: {duration} seconds")
            
            # Use Replicate's video generation models
            # Example: stability-ai/stable-video-diffusion or other video models
            model = "stability-ai/stable-video-diffusion"
            
            output = replicate.run(
                model,
                input={
                    "prompt": prompt,
                    "duration": min(duration, 4),  # Max 4 seconds
                    "fps": 8  # Frames per second
                }
            )
            
            # Download video
            video_url = output[0] if isinstance(output, list) else output
            import requests
            
            response = requests.get(video_url)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            generation_time = time.time() - start_time
            
            if self.metrics:
                self.metrics.log_image_generation_metrics({
                    'generation_time': generation_time,
                    'prompt_length': len(prompt),
                    'duration': duration,
                    'model': model,
                    'provider': 'replicate'
                })
            
            return {
                'success': True,
                'video_path': output_path,
                'prompt': prompt,
                'duration': duration,
                'generation_time': generation_time,
                'provider': 'replicate'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Replicate video generation failed: {str(e)}"
            }
    
    def _generate_video_from_frames(
        self,
        prompt: str,
        duration: int,
        output_path: str,
        start_time: float
    ) -> Dict[str, Any]:
        """Generate video by creating frames and animating them."""
        try:
            from src.generation.image_generator import ImageGenerator
            import subprocess
            
            print("  Generating video from frames...")
            
            # Generate start and end frames
            # Force enable image generation for video
            from src.config import config
            original_provider = config.get('models.image_generation.provider')
            config.set('models.image_generation.provider', 'dall-e')  # Temporarily enable DALL-E
            
            img_gen = ImageGenerator(use_wandb=False)
            
            # Restore original setting
            config.set('models.image_generation.provider', original_provider)
            
            # Generate frames for animation - simplified for 4-second video
            num_frames = duration * 8  # 8 fps for smooth video
            
            print(f"  Generating {num_frames} frames (8 fps for {duration}s video)...")
            
            # Generate key frames: start, middle, end
            print(f"  Generating key frames...")
            start_frame = img_gen.generate_image(prompt + ", cinematic, dynamic, professional video scene")
            
            if not start_frame.get('success'):
                start_frame = img_gen.generate_image(prompt)
            
            middle_prompt = prompt + ", cinematic, dynamic, mid-scene, different angle"
            middle_frame = img_gen.generate_image(middle_prompt)
            
            if not middle_frame.get('success'):
                middle_frame = start_frame  # Fallback to start frame
            
            end_prompt = prompt + ", cinematic, dynamic, end scene, different angle or movement"
            end_frame = img_gen.generate_image(end_prompt)
            
            if not end_frame.get('success'):
                end_frame = start_frame  # Fallback to start frame
            
            if not start_frame.get('success'):
                return {
                    'success': False,
                    'error': 'Failed to generate start frame'
                }
            
            # Create frames directory (clean up old frames first)
            frames_dir = Path(output_path).parent / "video_frames"
            if frames_dir.exists():
                import shutil
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            # Save frames - create smooth transition
            start_img = start_frame['image']
            middle_img = middle_frame['image'] if middle_frame.get('success') else start_img
            end_img = end_frame['image'] if end_frame.get('success') else start_img
            
            # Create frames with smooth transitions
            frames = []
            third = num_frames // 3
            for i in range(num_frames):
                frame_path = frames_dir / f"frame_{i:04d}.png"
                if i < third:
                    start_img.save(frame_path)
                elif i < 2 * third:
                    middle_img.save(frame_path)
                else:
                    end_img.save(frame_path)
                frames.append(frame_path)
            
            print(f"  ✓ Created {len(frames)} frames")
            
            # Use ffmpeg to create video
            print(f"  Creating video from {len(frames)} frames...")
            
            fps = 8  # 8 frames per second
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', str(frames_dir / 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-t', str(duration),
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f'FFmpeg failed: {result.stderr}'
                }
            
            generation_time = time.time() - start_time
            
            return {
                'success': True,
                'video_path': output_path,
                'prompt': prompt,
                'duration': duration,
                'generation_time': generation_time,
                'num_frames': len(frames),
                'provider': 'frame_animation'
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Frame-based video generation failed: {str(e)}"
            }
    
    def generate_scene_video(
        self,
        scenes: List[Dict[str, Any]],
        output_dir: str = "outputs",
        max_duration: int = 4
    ) -> Dict[str, Any]:
        """
        Generate a single video combining all scenes (max 4 seconds total).
        
        Args:
            scenes: List of scene dictionaries with prompts
            output_dir: Directory to save video
            max_duration: Maximum video duration in seconds (default 4)
            
        Returns:
            Dictionary with video path and metadata
        """
        # Combine all scene descriptions into one prompt
        scene_descriptions = []
        optimized_prompts = []
        
        for scene in scenes:
            # Prefer optimized prompt, fallback to start frame prompt
            prompt = scene.get('optimized_prompt') or scene.get('start_frame_prompt', '')
            if prompt:
                optimized_prompts.append(prompt)
            
            desc = scene.get('scene_description', '')
            if desc:
                scene_descriptions.append(desc[:100])  # Truncate descriptions
        
        # Create combined prompt for single 4-second video
        if optimized_prompts:
            # Use the first optimized prompt as base, add context from other scenes
            main_prompt = optimized_prompts[0]
            
            # Add brief context from other scenes
            if len(optimized_prompts) > 1:
                additional_context = ". ".join([p[:50] for p in optimized_prompts[1:3]])  # Max 2 more scenes
                combined_prompt = f"{main_prompt}. Scene transitions: {additional_context}. Fast-paced, cinematic video, 4 seconds total"
            else:
                combined_prompt = f"{main_prompt}, cinematic, dynamic, fast-paced, professional video, 4 seconds"
        else:
            # Fallback: combine scene descriptions
            combined_prompt = f"A cinematic video combining scenes: {' | '.join(scene_descriptions[:3])}. Fast-paced, 4 seconds total, professional quality"
        
        output_path = f"{output_dir}/final_video.mp4"
        
        print(f"\nGenerating 4-second video combining all scenes...")
        print(f"  Combined prompt: {combined_prompt[:150]}...")
        
        # Generate single 4-second video
        video_result = self.generate_video(combined_prompt, max_duration, output_path)
        
        if video_result.get('success'):
            return {
                'success': True,
                'video_path': output_path,
                'duration': max_duration,
                'prompt': combined_prompt,
                'scenes_combined': len(scenes)
            }
        else:
            return {
                'success': False,
                'error': video_result.get('error', 'Unknown error')
            }

