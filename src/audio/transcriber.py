"""Audio transcription using Whisper."""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import whisper
import torch
import numpy as np
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config
from src.observability.metrics import MetricsLogger

class AudioTranscriber:
    """Handles audio transcription using Whisper."""
    
    def __init__(self, use_wandb: bool = True):
        """Initialize transcriber."""
        self.provider = config.get('models.transcription.provider', 'openai')
        self.model_name = config.get('models.transcription.model', 'whisper-1')
        self.use_wandb = use_wandb
        self.metrics = MetricsLogger() if use_wandb else None
        
        if self.provider == 'openai':
            api_key = config.get_api_key('openai')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
            self.local_model = None
        else:
            # Load local Whisper model
            self.local_model = whisper.load_model(self.model_name)
            self.client = None
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and metadata
        """
        try:
            if self.provider == 'openai':
                # Use OpenAI Whisper API
                audio_path_obj = Path(audio_path)
                
                # Detect actual file type (file might be M4A even if extension is MP3)
                # Check file signature to determine actual format
                with open(audio_path, 'rb') as f:
                    header = f.read(12)
                    f.seek(0)
                    
                    # M4A files start with specific bytes
                    if header[:4] == b'ftyp' or header[4:8] == b'ftyp':
                        # It's M4A/M4V/MP4 format
                        filename = audio_path_obj.stem + '.m4a'
                        mime_type = 'audio/m4a'
                    else:
                        # Assume MP3 or use original extension
                        filename = audio_path_obj.name
                        ext = audio_path_obj.suffix.lower()
                        mime_type = f"audio/{ext[1:] if ext and ext[1:] else 'mp3'}"
                    
                    # Upload with proper format
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model_name,
                        file=(filename, f, mime_type),
                        response_format="verbose_json"
                    )
                
                transcription_text = transcript.text
                language = getattr(transcript, 'language', 'unknown')
                confidence = 1.0  # API doesn't provide confidence, assume high
                
            else:
                # Use local Whisper model
                if not self.local_model:
                    self.local_model = whisper.load_model(self.model_name)
                
                result = self.local_model.transcribe(audio_path)
                transcription_text = result['text']
                language = result.get('language', 'unknown')
                
                # Estimate confidence from segments
                segments = result.get('segments', [])
                if segments:
                    confidences = [seg.get('no_speech_prob', 0.0) for seg in segments]
                    confidence = 1.0 - np.mean(confidences) if confidences else 0.8
                else:
                    confidence = 0.8
            
            # Log metrics
            if self.metrics:
                self.metrics.log_transcription_metrics({
                    'transcription_length': len(transcription_text),
                    'language': language,
                    'confidence': confidence,
                    'provider': self.provider,
                    'model': self.model_name
                })
            
            return {
                'text': transcription_text,
                'language': language,
                'confidence': confidence,
                'provider': self.provider,
                'model': self.model_name,
                'success': True
            }
        
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            
            if self.metrics:
                self.metrics.log_error('transcription', error_msg)
            
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'provider': self.provider,
                'model': self.model_name,
                'success': False,
                'error': error_msg
            }

