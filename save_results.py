#!/usr/bin/env python3
"""Save all generated results to a readable file."""
import json
from pathlib import Path

def save_results_to_file(results: dict, output_file: str = "outputs/generated_content.md"):
    """Save all results to a markdown file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = []
    content.append("# Generated Video Content\n")
    content.append("## Your VC Ghost Video - Complete Blueprint\n")
    
    # Transcription
    if results.get('transcription'):
        trans = results['transcription']
        content.append("### Transcription\n")
        content.append(f"**Text:** {trans.get('text', 'N/A')}\n")
        content.append(f"**Language:** {trans.get('language', 'N/A')}\n")
        content.append(f"**Confidence:** {trans.get('confidence', 0.0):.2f}\n\n")
    
    # Context
    if results.get('context'):
        ctx = results['context']
        content.append("### Video Context\n")
        content.append(f"**Intent:** {ctx.get('intent', 'N/A')}\n")
        content.append(f"**Vibe:** {ctx.get('vibe', 'N/A')}\n")
        content.append(f"**Style:** {ctx.get('content_style', 'N/A')}\n")
        content.append(f"**Target Audience:** {ctx.get('target_audience', 'N/A')}\n\n")
        
        if ctx.get('visual_elements'):
            content.append(f"**Visual Elements:** {', '.join(ctx.get('visual_elements', []))}\n")
        if ctx.get('color_palette'):
            content.append(f"**Color Palette:** {', '.join(ctx.get('color_palette', []))}\n")
        content.append("\n")
    
    # Scenes
    if results.get('scenes'):
        content.append("## Scenes Generated\n")
        for scene in results['scenes']:
            scene_num = scene.get('scene_number', '?')
            content.append(f"### Scene {scene_num}\n")
            content.append(f"**Description:** {scene.get('scene_description', 'N/A')}\n\n")
            content.append(f"**Start Frame Prompt:**\n```\n{scene.get('start_frame_prompt', 'N/A')}\n```\n\n")
            content.append(f"**End Frame Prompt:**\n```\n{scene.get('end_frame_prompt', 'N/A')}\n```\n\n")
            content.append(f"**Transition:** {scene.get('transition_type', 'N/A')}\n")
            content.append(f"**Duration:** ~{scene.get('duration_estimate', 0)} seconds\n\n")
    
    # Virality Scores
    if results.get('virality_scores'):
        content.append("## Virality Evaluation\n")
        for score_data in results['virality_scores']:
            scene_num = score_data.get('scene_number', '?')
            score = score_data.get('virality_score', 0.0)
            content.append(f"**Scene {scene_num}:** {score:.3f}\n")
        content.append("\n")
    
    # RL Optimizations
    if results.get('optimizations'):
        content.append("## Self-Evolution Results (RL Optimizations)\n")
        for opt in results['optimizations']:
            scene_num = opt.get('scene_number', '?')
            content.append(f"### Scene {scene_num}\n")
            content.append(f"**Base Prompt:**\n```\n{opt.get('base_prompt', 'N/A')}\n```\n\n")
            content.append(f"**Optimized Prompt:**\n```\n{opt.get('optimized_prompt', 'N/A')}\n```\n\n")
            content.append(f"**Virality Score:** {opt.get('virality_score', 0.0):.3f}\n")
            content.append(f"**Reward:** {opt.get('reward', 0.0):.3f}\n\n")
    
    # Evolution History
    if results.get('evolution_history'):
        content.append("## Evolution History (Self-Learning Progress)\n")
        for evo in results['evolution_history']:
            content.append(f"- **Iteration {evo['iteration']}:** Virality={evo['virality_score']:.3f}, Reward={evo['reward']:.3f}\n")
    
    # Write file
    output_path.write_text('\n'.join(content))
    print(f"\n✓ All results saved to: {output_path}")
    return output_path

