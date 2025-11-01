#!/usr/bin/env python3
"""
Self-Evolving Image Generation Agent
Main pipeline entry point.
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
from src.generation.image_generator import ImageGenerator
from src.evaluation.virality_evaluator import ViralityEvaluator
from src.rl.prompt_optimizer import RLPromptOptimizer
from src.observability.metrics import MetricsLogger
from src.orchestration.orchestrator import AgentOrchestrator

def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description='Self-Evolving Image Generation Agent')
    parser.add_argument('audio_path', type=str, help='Path to audio file (meeting/voice note)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory for generated images')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--disable-rl', action='store_true', help='Disable RL optimization')
    
    args = parser.parse_args()
    
    # Validate audio file
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize observability
    use_wandb = not args.disable_wandb
    metrics = MetricsLogger() if use_wandb else None
    
    print("=" * 60)
    print("Self-Evolving Image Generation Agent")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Output directory: {output_dir}")
    print(f"W&B logging: {use_wandb}")
    print(f"RL optimization: {not args.disable_rl}")
    print("=" * 60)
    
    try:
        # Initialize components
        print("\nInitializing components...")
        
        transcriber = AudioTranscriber(use_wandb=use_wandb)
        context_extractor = ContextExtractor(use_wandb=use_wandb)
        
        try:
            image_generator = ImageGenerator(use_wandb=use_wandb)
        except Exception as e:
            print(f"Warning: Image generator initialization failed: {e}")
            print("You may need to install CUDA or use a different provider.")
            image_generator = None
        
        try:
            virality_evaluator = ViralityEvaluator(use_wandb=use_wandb)
        except Exception as e:
            print(f"Warning: Virality evaluator initialization failed: {e}")
            print("Check your FIRECRAWL_API_KEY.")
            virality_evaluator = None
        
        rl_optimizer = None
        if not args.disable_rl:
            try:
                rl_optimizer = RLPromptOptimizer(use_wandb=use_wandb)
            except Exception as e:
                print(f"Warning: RL optimizer initialization failed: {e}")
                rl_optimizer = None
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        orchestrator.register_agent('transcription', transcriber)
        orchestrator.register_agent('context', context_extractor)
        orchestrator.register_agent('generation', image_generator)
        orchestrator.register_agent('evaluation', virality_evaluator)
        orchestrator.register_agent('rl', rl_optimizer)
        
        # Update config
        config.set('pipeline.enable_rl', not args.disable_rl)
        
        print("\nComponents initialized successfully!")
        print("\nStarting workflow...\n")
        
        # Execute workflow
        results = orchestrator.execute_sync(
            str(audio_path),
            str(output_dir)
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("Workflow Results")
        print("=" * 60)
        
        if results.get('success', False):
            print("✓ Workflow completed successfully!")
            
            if results.get('transcription'):
                trans = results['transcription']
                print(f"\nTranscription:")
                print(f"  Text: {trans.get('text', '')[:200]}...")
                print(f"  Confidence: {trans.get('confidence', 0.0):.2f}")
            
            if results.get('context'):
                ctx = results['context']
                print(f"\nContext Extracted:")
                print(f"  Intent: {ctx.get('intent', 'N/A')}")
                print(f"  Vibe: {ctx.get('vibe', 'N/A')}")
            
            if results.get('scenes'):
                print(f"\nScenes Generated: {len(results['scenes'])}")
                for scene in results['scenes']:
                    print(f"  Scene {scene.get('scene_number', '?')}: {scene.get('scene_description', 'N/A')[:100]}...")
            
            if results.get('images'):
                print(f"\nImages Generated: {len(results['images'])} scenes")
                for img_result in results['images']:
                    if img_result.get('success'):
                        print(f"  Scene {img_result.get('scene_number', '?')}: Start and end frames generated")
            
            if results.get('virality_scores'):
                print(f"\nVirality Scores:")
                for score_data in results['virality_scores']:
                    print(f"  Scene {score_data.get('scene_number', '?')}: {score_data.get('virality_score', 0.0):.3f}")
            
            if results.get('optimizations'):
                print(f"\nRL Optimizations Applied: {len(results['optimizations'])}")
                for opt in results['optimizations'][:3]:  # Show first 3
                    print(f"  Scene {opt.get('scene_number', '?')}: Prompt optimized")
        else:
            print("✗ Workflow failed!")
            if 'error' in results:
                print(f"Error: {results['error']}")
            return 1
        
        # Save results to file
        if results.get('success', False):
            try:
                from save_results import save_results_to_file
                output_file = save_results_to_file(results, f"{args.output_dir}/generated_content.md")
                print(f"\n✓ All results saved to: {output_file}")
                print(f"  Open this file to see all scenes, prompts, and optimizations!")
            except Exception as e:
                print(f"\n⚠ Could not save results file: {e}")
        
        # Log final metrics
        if metrics:
            metrics.log_system_metrics()
            print(f"\nW&B Dashboard: {metrics.project_name}")
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        
        if results.get('success', False):
            print("\n📄 Generated Content:")
            print(f"  - Scene descriptions and prompts: {args.output_dir}/generated_content.md")
            if results.get('images'):
                print(f"  - Generated images: {args.output_dir}/")
            else:
                print(f"  - To generate images: Switch to 'dall-e' in config.yaml")
                print(f"  - Then run again: python main.py {args.audio_path}")
            print("\n💡 Note: This system generates the video blueprint, not the video itself.")
            print("   Use the prompts/images with video editing software to create the final video.")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        if metrics:
            metrics.finish()

if __name__ == '__main__':
    sys.exit(main())

