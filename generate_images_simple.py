#!/usr/bin/env python3
"""Simple script to generate images from the optimized prompts."""
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.generation.image_generator import ImageGenerator
from src.config import config

def main():
    """Generate images from generated prompts."""
    print("=" * 60)
    print("Image Generation from Optimized Prompts")
    print("=" * 60)
    
    # Read generated content
    content_file = Path("outputs/generated_content.md")
    if not content_file.exists():
        print("✗ No generated content found. Run main.py first!")
        return 1
    
    # Parse optimized prompts
    with open(content_file, 'r') as f:
        content = f.read()
    
    # Extract optimized prompts
    scene_sections = re.findall(r'### Scene (\d+)(.*?)(?=### Scene|\Z)', content, re.DOTALL)
    
    scenes_with_prompts = []
    for scene_match in scene_sections:
        scene_num = int(scene_match[0])
        scene_text = scene_match[1]
        
        # Extract optimized prompt
        opt_match = re.search(r'\*\*Optimized Prompt:\*\*\n```\n(.*?)\n```', scene_text, re.DOTALL)
        if opt_match:
            prompt = opt_match.group(1).strip()
            scenes_with_prompts.append({
                'scene_number': scene_num,
                'prompt': prompt
            })
    
    if not scenes_with_prompts:
        print("✗ No optimized prompts found!")
        return 1
    
    print(f"\n✓ Found {len(scenes_with_prompts)} scenes with optimized prompts")
    print(f"  Generating images using DALL-E...\n")
    
    # Initialize image generator
    try:
        img_gen = ImageGenerator(use_wandb=True)
        print("✓ Image generator initialized (DALL-E)\n")
    except Exception as e:
        print(f"✗ Image generator failed: {e}")
        return 1
    
    # Generate images
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    generated_images = []
    
    for scene in scenes_with_prompts[:5]:  # Max 5 scenes
        scene_num = scene['scene_number']
        prompt = scene['prompt']
        
        print(f"Scene {scene_num}: Generating image...")
        print(f"  Prompt: {prompt[:80]}...")
        
        result = img_gen.generate_image(prompt)
        
        if result.get('success'):
            image = result['image']
            output_path = output_dir / f"scene_{scene_num}_optimized.png"
            img_gen.save_image(image, str(output_path))
            generated_images.append({
                'scene_number': scene_num,
                'path': str(output_path),
                'prompt': prompt
            })
            print(f"  ✓ Saved: {output_path}\n")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}\n")
    
    print("=" * 60)
    print("✓ Image Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated {len(generated_images)} image(s):")
    for img in generated_images:
        print(f"  Scene {img['scene_number']}: {img['path']}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

