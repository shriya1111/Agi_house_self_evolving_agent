#!/usr/bin/env python3
"""Generate video from the latest generated content."""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.generation.video_generator import VideoGenerator
from src.orchestration.orchestrator import AgentOrchestrator
from src.config import config
import yaml

def load_generated_scenes():
    """Load scenes from generated_content.md or config."""
    content_file = Path("outputs/generated_content.md")
    
    # For now, we'll load from the last run's results
    # In production, this would parse the markdown or use a JSON output
    
    # Read from config or use default scenes
    scenes = []
    
    # Try to read from a results file if it exists
    results_file = Path("outputs/last_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
            scenes = results.get('scenes', [])
    
    return scenes

def main():
    """Generate video from prompts."""
    print("=" * 60)
    print("Video Generation from Generated Prompts")
    print("=" * 60)
    
    # Initialize video generator
    try:
        video_gen = VideoGenerator(use_wandb=True)
        print("✓ Video generator initialized")
    except Exception as e:
        print(f"✗ Video generator failed: {e}")
        print("\nFalling back to frame-based animation...")
        config.set('video_generation.provider', 'frame_animation')
        video_gen = VideoGenerator(use_wandb=True)
    
    # Load generated scenes/prompts
    # For demo, we'll use the optimized prompts from the last run
    print("\nLoading generated prompts...")
    
    # Read the generated content markdown to extract prompts
    content_file = Path("outputs/generated_content.md")
    if not content_file.exists():
        print("✗ No generated content found. Run main.py first!")
        return 1
    
    # Parse scenes from markdown (simplified)
    scenes = []
    with open(content_file, 'r') as f:
        content = f.read()
        
        # Extract scene prompts (simplified parsing)
        import re
        scene_sections = re.findall(r'### Scene (\d+)(.*?)(?=### Scene|\Z)', content, re.DOTALL)
        
        for scene_match in scene_sections:
            scene_num = int(scene_match[0])
            scene_text = scene_match[1]
            
            # Extract optimized prompt
            opt_match = re.search(r'\*\*Optimized Prompt:\*\*\n```\n(.*?)\n```', scene_text, re.DOTALL)
            if opt_match:
                prompt = opt_match.group(1).strip()
            else:
                # Fallback to start frame prompt
                start_match = re.search(r'\*\*Start Frame Prompt:\*\*\n```\n(.*?)\n```', scene_text, re.DOTALL)
                prompt = start_match.group(1).strip() if start_match else ""
            
            if prompt:
                scenes.append({
                    'scene_number': scene_num,
                    'optimized_prompt': prompt,
                    'duration_estimate': 3  # 3-4 seconds per scene
                })
    
    if not scenes:
        print("✗ No prompts found in generated content!")
        print("  Make sure you've run main.py first to generate scenes.")
        return 1
    
    print(f"✓ Found {len(scenes)} scenes with prompts")
    
    # Use all scenes but combine into ONE 4-second video
    print(f"\nCombining all {len(scenes)} scenes into a single 4-second video...")
    print(f"  Target: ONE video, 4 seconds total\n")
    
    # Generate single combined video
    results = video_gen.generate_scene_video(scenes, output_dir="outputs", max_duration=4)
    
    if results.get('success'):
        print("\n" + "=" * 60)
        print("✓ Video Generation Complete!")
        print("=" * 60)
        print(f"\n📹 Generated video:")
        print(f"  Path: {results['video_path']}")
        print(f"  Duration: {results['duration']} seconds")
        print(f"  Scenes combined: {results['scenes_combined']}")
        print(f"\n✅ Video ready! Open: {results['video_path']}")
        
        return 0
    else:
        print(f"\n✗ Video generation failed: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

