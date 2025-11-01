#!/usr/bin/env python3
"""
Quick test mode - skips image generation to test transcription and context extraction faster.
"""
import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config import config
from src.audio.transcriber import AudioTranscriber
from src.context.extractor import ContextExtractor
from src.evaluation.virality_evaluator import ViralityEvaluator
from src.observability.metrics import MetricsLogger
from src.orchestration.orchestrator import AgentOrchestrator

def main():
    """Quick test - transcription, context, and scenes only."""
    parser = argparse.ArgumentParser(description='Quick Test Mode (No Image Generation)')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable W&B logging')
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    use_wandb = not args.disable_wandb
    metrics = MetricsLogger() if use_wandb else None
    
    print("=" * 60)
    print("Quick Test Mode - Transcription & Context Extraction")
    print("(Image generation skipped for faster testing)")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print("=" * 60)
    
    try:
        # Initialize components (without image generator)
        print("\nInitializing components...")
        
        transcriber = AudioTranscriber(use_wandb=use_wandb)
        context_extractor = ContextExtractor(use_wandb=use_wandb)
        
        try:
            virality_evaluator = ViralityEvaluator(use_wandb=use_wandb)
        except Exception as e:
            print(f"Warning: Virality evaluator failed: {e}")
            virality_evaluator = None
        
        print("✓ Components initialized")
        
        # Step 1: Transcription
        print("\n" + "=" * 60)
        print("Step 1: Transcribing audio...")
        print("=" * 60)
        transcription_result = transcriber.transcribe(str(audio_path))
        
        if not transcription_result.get('success', False):
            print(f"✗ Transcription failed: {transcription_result.get('error', 'Unknown error')}")
            return 1
        
        transcription_text = transcription_result.get('text', '')
        print(f"\n✓ Transcription successful!")
        print(f"Language: {transcription_result.get('language', 'unknown')}")
        print(f"Confidence: {transcription_result.get('confidence', 0.0):.2f}")
        print(f"\nTranscription text:")
        print("-" * 60)
        print(transcription_text[:500] + "..." if len(transcription_text) > 500 else transcription_text)
        print("-" * 60)
        
        # Step 2: Context Extraction
        print("\n" + "=" * 60)
        print("Step 2: Extracting context...")
        print("=" * 60)
        context = context_extractor.extract_context(transcription_text)
        
        if 'error' in context:
            print(f"✗ Context extraction failed: {context['error']}")
            return 1
        
        print(f"\n✓ Context extracted successfully!")
        print(f"Intent: {context.get('intent', 'N/A')}")
        print(f"Vibe: {context.get('vibe', 'N/A')}")
        print(f"Style: {context.get('content_style', 'N/A')}")
        print(f"Target Audience: {context.get('target_audience', 'N/A')}")
        
        if context.get('visual_elements'):
            print(f"Visual Elements: {', '.join(context.get('visual_elements', []))}")
        
        if context.get('color_palette'):
            print(f"Color Palette: {', '.join(context.get('color_palette', []))}")
        
        # Step 3: Scene Generation
        print("\n" + "=" * 60)
        print("Step 3: Generating scenes...")
        print("=" * 60)
        scenes = context_extractor.generate_scenes(context)
        
        if not scenes:
            print("✗ Scene generation failed")
            return 1
        
        print(f"\n✓ Generated {len(scenes)} scenes!")
        for scene in scenes:
            scene_num = scene.get('scene_number', '?')
            desc = scene.get('scene_description', 'N/A')[:100]
            print(f"\nScene {scene_num}:")
            print(f"  Description: {desc}...")
            print(f"  Duration: ~{scene.get('duration_estimate', 0)} seconds")
            print(f"  Transition: {scene.get('transition_type', 'N/A')}")
        
        # Step 4: Virality Evaluation (if available)
        if virality_evaluator:
            print("\n" + "=" * 60)
            print("Step 4: Evaluating virality...")
            print("=" * 60)
            print("Scraping forums for virality metrics...")
            
            try:
                # Extract keywords from transcription
                keywords = virality_evaluator._extract_keywords(transcription_text)
                scraping_result = virality_evaluator.scrape_forums(keywords)
                
                if scraping_result.get('success', False):
                    virality_score = scraping_result.get('virality_score', 0.0)
                    print(f"\n✓ Virality evaluation complete!")
                    print(f"Virality Score: {virality_score:.3f}")
                    print(f"URLs Scraped: {scraping_result.get('total_urls_scraped', 0)}")
                else:
                    print(f"⚠ Virality evaluation had issues: {scraping_result.get('error', 'Unknown')}")
            except Exception as e:
                print(f"⚠ Virality evaluation error: {e}")
        
        print("\n" + "=" * 60)
        print("✓ Quick test completed successfully!")
        print("=" * 60)
        print("\nNote: Image generation was skipped (very slow on CPU).")
        print("To test full pipeline with image generation, use: python main.py")
        print("(Image generation requires GPU or will be very slow on CPU)")
        
        if metrics:
            metrics.log_system_metrics()
            metrics.finish()
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if metrics:
            metrics.finish()

if __name__ == '__main__':
    sys.exit(main())

