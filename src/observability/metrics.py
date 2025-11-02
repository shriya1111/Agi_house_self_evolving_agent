"""Weights & Biases metrics logging with Weave."""
import sys
import os
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config

# Weave integration (3 lines)
try:
    import weave
    weave.init(project_name=config.get('wandb.project', 'self-evolving-image-gen'))
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    print("Warning: Weave not available. Install with: pip install weave")

class MetricsLogger:
    """Centralized metrics logging with Weights & Biases."""
    
    def __init__(self, project_name: Optional[str] = None, entity: Optional[str] = None):
        """Initialize W&B logger."""
        self.project_name = project_name or config.get('wandb.project', 'self-evolving-image-gen')
        self.entity = entity or config.get('wandb.entity')
        
        # Initialize W&B
        api_key = config.get_api_key('wandb')
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        
        # Check if Weave is available
        self.weave_available = WEAVE_AVAILABLE
        
        wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=config.config,
            mode='online' if api_key else 'disabled'
        )
        
        print(f"W&B initialized: project={self.project_name}, entity={self.entity}")
    
    def log_transcription_metrics(self, metrics: Dict[str, Any]):
        """Log transcription metrics."""
        wandb.log({
            'transcription/length': metrics.get('transcription_length', 0),
            'transcription/language': metrics.get('language', 'unknown'),
            'transcription/confidence': metrics.get('confidence', 0.0),
            'transcription/provider': metrics.get('provider', 'unknown'),
            'transcription/model': metrics.get('model', 'unknown')
        })
    
    def log_llm_metrics(self, metrics: Dict[str, Any]):
        """Log LLM usage metrics."""
        wandb.log({
            'llm/tokens_used': metrics.get('tokens_used', 0),
            'llm/prompt_tokens': metrics.get('prompt_tokens', 0),
            'llm/completion_tokens': metrics.get('completion_tokens', 0),
            'llm/model': metrics.get('model', 'unknown'),
            'llm/cost_estimate': self._estimate_cost(metrics)
        })
    
    def log_scene_generation_metrics(self, metrics: Dict[str, Any]):
        """Log scene generation metrics."""
        wandb.log({
            'scenes/num_scenes': metrics.get('num_scenes', 0),
            'scenes/tokens_used': metrics.get('tokens_used', 0),
            'scenes/model': metrics.get('model', 'unknown')
        })
    
    def log_image_generation_metrics(self, metrics: Dict[str, Any]):
        """Log image generation metrics."""
        log_dict = {
            'image_gen/generation_time': metrics.get('generation_time', 0),
            'image_gen/prompt_length': metrics.get('prompt_length', 0),
            'image_gen/num_inference_steps': metrics.get('num_inference_steps', 0),
            'image_gen/guidance_scale': metrics.get('guidance_scale', 0),
            'image_gen/device': metrics.get('device', 'unknown'),
            'image_gen/model': metrics.get('model', 'unknown')
        }
        
        # Add image size
        img_size = metrics.get('image_size', [])
        if img_size:
            log_dict['image_gen/image_width'] = img_size[0] if len(img_size) > 0 else 0
            log_dict['image_gen/image_height'] = img_size[1] if len(img_size) > 1 else 0
        
        wandb.log(log_dict)
    
    def log_gpu_metrics(self, metrics: Dict[str, Any]):
        """Log GPU usage metrics."""
        wandb.log({
            'gpu/memory_used_gb': metrics.get('gpu_memory_used_gb', 0),
            'gpu/memory_max_gb': metrics.get('gpu_memory_max_gb', 0)
        })
    
    def log_scraping_metrics(self, metrics: Dict[str, Any]):
        """Log Firecrawl scraping metrics with Weave."""
        # Log to W&B
        wandb.log({
            'scraping/url': metrics.get('url', 'unknown'),
            'scraping/pages_scraped': metrics.get('pages_scraped', 0),
            'scraping/engagement_score': metrics.get('engagement_score', 0),
            'scraping/keywords_found': metrics.get('keywords_found', 0),
            'scraping/status': metrics.get('status', 'unknown')
        })
        
        # Also log to Weave if available
        if self.weave_available:
            try:
                import weave
                weave.log({'firecrawl_scraping': metrics})
            except Exception:
                pass
    
    def log_virality_evaluation(self, metrics: Dict[str, Any]):
        """Log virality evaluation metrics."""
        wandb.log({
            'virality/score': metrics.get('virality_score', 0),
            'virality/keywords_used': len(metrics.get('keywords_used', [])),
            'virality/prompt_preview': metrics.get('prompt', '')[:100]
        })
    
    def log_rl_metrics(self, metrics: Dict[str, Any]):
        """Log RL agent metrics."""
        wandb.log({
            'rl/action_magnitude': metrics.get('action_magnitude', 0),
            'rl/virality_score': metrics.get('virality_score', 0),
            'rl/step_count': metrics.get('step_count', 0),
            'rl/buffer_size': metrics.get('buffer_size', 0)
        })
    
    def log_rl_update_metrics(self, metrics: Dict[str, Any]):
        """Log RL policy update metrics."""
        wandb.log({
            'rl_update/loss': metrics.get('loss', 0),
            'rl_update/mean_reward': metrics.get('mean_reward', 0),
            'rl_update/std_reward': metrics.get('std_reward', 0),
            'rl_update/virality_score': metrics.get('virality_score', 0)
        })
    
    def log_system_metrics(self):
        """Log system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        log_dict = {
            'system/cpu_percent': cpu_percent,
            'system/memory_percent': memory.percent,
            'system/memory_used_gb': memory.used / (1024**3),
            'system/memory_available_gb': memory.available / (1024**3)
        }
        
        if torch.cuda.is_available():
            log_dict['system/gpu_available'] = 1
            log_dict['system/gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
        else:
            log_dict['system/gpu_available'] = 0
        
        wandb.log(log_dict)
    
    def log_error(self, component: str, error_message: str):
        """Log errors."""
        wandb.log({
            f'errors/{component}': 1,
            f'errors/{component}_message': error_message[:200]  # Truncate
        })
    
    def log_image(self, image, caption: str = ""):
        """Log image to W&B."""
        wandb.log({
            'generated_images': wandb.Image(image, caption=caption)
        })
    
    def _estimate_cost(self, metrics: Dict[str, Any]) -> float:
        """Estimate API cost for LLM calls."""
        # Rough estimates (update based on actual pricing)
        model = metrics.get('model', '')
        prompt_tokens = metrics.get('prompt_tokens', 0)
        completion_tokens = metrics.get('completion_tokens', 0)
        
        if 'gpt-4o' in model:
            # GPT-4o pricing (approximate)
            cost = (prompt_tokens / 1e6 * 2.50) + (completion_tokens / 1e6 * 10.00)
        elif 'gpt-4' in model:
            cost = (prompt_tokens / 1e6 * 30.00) + (completion_tokens / 1e6 * 60.00)
        else:
            cost = 0.0
        
        return cost
    
    def finish(self):
        """Finish W&B run."""
        wandb.finish()

