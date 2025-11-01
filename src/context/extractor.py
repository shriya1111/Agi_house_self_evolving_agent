"""Context extraction from transcriptions using GPT-4o."""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.observability.metrics import MetricsLogger

class ContextExtractor:
    """Extracts video intent, vibe, and content requirements from transcriptions."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize context extractor."""
        self.provider = config.get('models.llm.provider', 'openai')
        self.model = config.get('models.llm.model', 'gpt-4o')
        self.temperature = config.get('models.llm.temperature', 0.7)
        self.max_tokens = config.get('models.llm.max_tokens', 2000)
        self.use_wandb = use_wandb
        self.metrics = MetricsLogger() if use_wandb else None
        
        if self.provider == 'openai':
            api_key = config.get_api_key('openai')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Provider {self.provider} not supported yet")
    
    def extract_context(self, transcription: str) -> Dict[str, Any]:
        """
        Extract video context from transcription.
        
        Args:
            transcription: Transcription text
            
        Returns:
            Dictionary with extracted context (vibe, content, style, etc.)
        """
        try:
            prompt = self._build_extraction_prompt(transcription)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert video content analyst. Extract structured information about video intent, vibe, and requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            context_str = response.choices[0].message.content
            context = json.loads(context_str)
            
            # Add metadata
            context['extraction_metadata'] = {
                'model': self.model,
                'tokens_used': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            
            # Log metrics
            if self.metrics:
                self.metrics.log_llm_metrics({
                    'model': self.model,
                    'tokens_used': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'context_extracted': True
                })
            
            return context
        
        except Exception as e:
            error_msg = f"Context extraction failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('context_extraction', error_msg)
            
            return {
                'error': error_msg,
                'success': False
            }
    
    def generate_scenes(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate scene descriptions with start/end frames from context.
        
        Args:
            context: Extracted video context
            
        Returns:
            List of scene descriptions
        """
        try:
            prompt = self._build_scene_prompt(context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert video storyboarder. Generate detailed scene descriptions with start and end frame requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            scenes_str = response.choices[0].message.content
            scenes_data = json.loads(scenes_str)
            
            # Ensure scenes is a list
            scenes = scenes_data.get('scenes', [])
            if not isinstance(scenes, list):
                scenes = [scenes] if scenes else []
            
            # Add metadata to each scene
            for i, scene in enumerate(scenes):
                scene['scene_number'] = i + 1
                scene['scene_metadata'] = {
                    'model': self.model,
                    'tokens_used': response.usage.total_tokens
                }
            
            # Log metrics
            if self.metrics:
                self.metrics.log_scene_generation_metrics({
                    'num_scenes': len(scenes),
                    'model': self.model,
                    'tokens_used': response.usage.total_tokens
                })
            
            return scenes
        
        except Exception as e:
            error_msg = f"Scene generation failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('scene_generation', error_msg)
            
            return []
    
    def _build_extraction_prompt(self, transcription: str) -> str:
        """Build prompt for context extraction."""
        return f"""Analyze the following transcription from a meeting or voice note about video creation. Extract structured information about:
1. Video intent and purpose
2. Overall vibe and mood (energetic, calm, professional, casual, etc.)
3. Content style and aesthetic
4. Target audience
5. Key visual elements mentioned
6. Color palette preferences
7. Music/sound preferences if mentioned

Return the analysis as a JSON object with these fields:
- intent: string
- vibe: string (descriptive)
- content_style: string
- target_audience: string
- visual_elements: array of strings
- color_palette: array of strings
- sound_preferences: string (if mentioned)

Transcription:
{transcription}
"""
    
    def _build_scene_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for scene generation."""
        context_str = json.dumps(context, indent=2)
        
        return f"""Based on the following video context, generate a detailed scene-by-scene breakdown.
Each scene should have:
- scene_description: detailed description of what happens in this scene
- start_frame_prompt: detailed prompt for image generation of the opening frame
- end_frame_prompt: detailed prompt for image generation of the closing frame
- transition_type: type of transition (cut, fade, zoom, etc.)
- duration_estimate: estimated duration in seconds

Generate between 3-10 scenes depending on the content complexity.

Return as JSON with structure:
{{
  "scenes": [
    {{
      "scene_description": "...",
      "start_frame_prompt": "...",
      "end_frame_prompt": "...",
      "transition_type": "...",
      "duration_estimate": number
    }}
  ]
}}

Video Context:
{context_str}
"""

