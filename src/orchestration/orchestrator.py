"""Multi-agent orchestration."""
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import config

# Optional Composio import
try:
    from composio_core import Action, ComposioToolSet, App
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False
    print("Composio not available - orchestration will work without it")

class AgentOrchestrator:
    """Orchestrates multiple agents."""
    
    def __init__(self):
        """Initialize orchestrator."""
        self.toolset = None
        if COMPOSIO_AVAILABLE:
            api_key = config.get_api_key('composio')
            if api_key:
                try:
                    self.toolset = ComposioToolSet(api_key=api_key)
                    print("Composio initialized")
                except Exception as e:
                    print(f"Composio initialization failed: {e}")
                    self.toolset = None
        
        # Register agents/tools
        self.agents = {
            'transcription': None,
            'context': None,
            'generation': None,
            'evaluation': None,
            'rl': None
        }
    
    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent instance."""
        if agent_name in self.agents:
            self.agents[agent_name] = agent_instance
            print(f"Registered agent: {agent_name}")
        else:
            print(f"Warning: Unknown agent name: {agent_name}")
    
    async def execute_workflow(
        self,
        audio_path: str,
        output_dir: str = "outputs"
    ) -> Dict[str, Any]:
        """
        Execute the complete workflow.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory for output files
            
        Returns:
            Complete workflow results
        """
        results = {
            'success': False,
            'transcription': None,
            'context': None,
            'scenes': None,
            'images': None,
            'virality_scores': None,
            'optimizations': None
        }
        
        try:
            # Step 1: Transcription
            if self.agents['transcription']:
                print("Step 1: Transcribing audio...")
                transcription_result = self.agents['transcription'].transcribe(audio_path)
                results['transcription'] = transcription_result
                
                if not transcription_result.get('success', False):
                    return results
                
                transcription_text = transcription_result.get('text', '')
            else:
                return {'error': 'Transcription agent not registered', **results}
            
            # Step 2: Context Extraction
            if self.agents['context']:
                print("Step 2: Extracting context...")
                context = self.agents['context'].extract_context(transcription_text)
                results['context'] = context
                
                if 'error' in context:
                    return results
                
                # Step 3: Scene Generation
                print("Step 3: Generating scenes...")
                scenes = self.agents['context'].generate_scenes(context)
                results['scenes'] = scenes
                
            else:
                return {'error': 'Context agent not registered', **results}
            
            # Step 4: Image Generation (Preview Mode - 1 image only)
            provider = config.get('models.image_generation.provider', 'skip')
            
            if provider == 'skip':
                print("Step 4: Skipping image generation (skip mode enabled)")
                print("  All prompts and scenes are ready - images can be generated later")
                results['images'] = []
                results['prompts_ready'] = True
            elif self.agents['generation']:
                preview_mode = config.get('pipeline.preview_mode', True)
                limit_images = config.get('pipeline.limit_preview_images', 1)
                
                if preview_mode:
                    print(f"Step 4: Generating preview image (Preview Mode: {limit_images} image(s))...")
                else:
                    print("Step 4: Generating images...")
                
                generated_images = []
                scenes_to_process = scenes[:limit_images] if preview_mode else scenes
                
                for i, scene in enumerate(scenes_to_process):
                    start_prompt = scene.get('start_frame_prompt', '')
                    end_prompt = scene.get('end_frame_prompt', '')
                    scene_num = scene.get('scene_number', 0)
                    
                    try:
                        if preview_mode and i == 0:
                            # Preview mode: Generate only the start frame of the first scene
                            print(f"  Generating preview for Scene {scene_num} (start frame)...")
                            start_frame = self.agents['generation'].generate_image(start_prompt)
                            
                            if start_frame.get('success', False):
                                start_img = start_frame.get('image')
                                if start_img:
                                    output_path = f"{output_dir}/scene_{scene_num}_preview.png"
                                    self.agents['generation'].save_image(start_img, output_path)
                                    print(f"  ✓ Preview image saved: {output_path}")
                                
                                generated_images.append({
                                    'scene_number': scene_num,
                                    'start_frame': start_frame,
                                    'end_frame': {'success': False, 'skipped': 'preview_mode'},
                                    'success': True,
                                    'preview_mode': True
                                })
                            else:
                                error = start_frame.get('error', 'Unknown error')
                                if 'skipped' in error.lower() or start_frame.get('skipped'):
                                    print(f"  ⚠ Image generation skipped: {error}")
                                    results['prompts_ready'] = True
                                else:
                                    print(f"  ✗ Image generation failed: {error}")
                                generated_images.append({
                                    'scene_number': scene_num,
                                    'success': False,
                                    'error': error
                                })
                        else:
                            # Full mode: Generate both start and end frames
                            frames = self.agents['generation'].generate_scene_frames(
                                start_prompt,
                                end_prompt,
                                scene_num
                            )
                            
                            generated_images.append(frames)
                            
                            # Save images
                            if frames.get('success', False):
                                start_frame_data = frames.get('start_frame', {})
                                end_frame_data = frames.get('end_frame', {})
                                
                                start_img = start_frame_data.get('image') if isinstance(start_frame_data, dict) else None
                                end_img = end_frame_data.get('image') if isinstance(end_frame_data, dict) else None
                                
                                if start_img:
                                    output_path = f"{output_dir}/scene_{scene_num}_start.png"
                                    self.agents['generation'].save_image(start_img, output_path)
                                
                                if end_img:
                                    output_path = f"{output_dir}/scene_{scene_num}_end.png"
                                    self.agents['generation'].save_image(end_img, output_path)
                    except Exception as e:
                        print(f"Error generating frames for scene {scene_num}: {e}")
                        generated_images.append({
                            'scene_number': scene_num,
                            'success': False,
                            'error': str(e)
                        })
                
                results['images'] = generated_images
                if preview_mode:
                    print(f"  Preview mode: Generated {len(generated_images)} preview image(s)")
                    print(f"  (Full generation available - set preview_mode: false in config.yaml)")
            else:
                print("Step 4: Image generation disabled (no generator available)")
                results['images'] = []
                results['prompts_ready'] = True
            
            # Step 5: Virality Evaluation
            if self.agents['evaluation']:
                print("Step 5: Evaluating virality...")
                virality_scores = []
                
                for i, scene in enumerate(scenes):
                    prompt = scene.get('start_frame_prompt', '')
                    virality_score = self.agents['evaluation'].evaluate_prompt_virality(
                        prompt,
                        {'scene': scene}
                    )
                    virality_scores.append({
                        'scene_number': i + 1,
                        'virality_score': virality_score,
                        'prompt': prompt
                    })
                
                results['virality_scores'] = virality_scores
            
            # Step 6: RL Optimization & Self-Evolution (if enabled)
            if config.get('pipeline.enable_rl', True) and self.agents['rl']:
                print("\n" + "=" * 60)
                print("Step 6: Self-Evolving Prompt Optimization (RL Learning)...")
                print("=" * 60)
                optimizations = []
                
                context = results.get('context', {})
                
                # Track evolution over iterations
                evolution_history = []
                
                for i, (scene, virality_data) in enumerate(zip(scenes, virality_scores or [])):
                    base_prompt = scene.get('start_frame_prompt', '')
                    virality_score = virality_data.get('virality_score', 0.0) if virality_scores else 0.0
                    
                    print(f"\nIteration {i+1}/{len(scenes)}: Evolving prompt...")
                    print(f"  Base virality score: {virality_score:.3f}")
                    print(f"  Base prompt: {base_prompt[:80]}...")
                    
                    # Optimize prompt using RL
                    optimized_prompt, metadata = self.agents['rl'].optimize_prompt(
                        base_prompt,
                        context,
                        virality_score
                    )
                    
                    print(f"  Optimized prompt: {optimized_prompt[:80]}...")
                    print(f"  Action magnitude: {metadata.get('action_magnitude', 0):.3f}")
                    
                    # Calculate reward based on virality improvement
                    if i > 0 and virality_scores and len(virality_scores) > 1:
                        prev_score = virality_scores[i-1].get('virality_score', 0.0)
                        reward = virality_score - prev_score
                        print(f"  Reward: {reward:.3f} (current {virality_score:.3f} - previous {prev_score:.3f})")
                        
                        # Update policy - THIS IS THE SELF-EVOLVING PART
                        self.agents['rl'].update_policy(reward, virality_score)
                        print(f"  ✓ Policy updated - learning from feedback!")
                    else:
                        # For first iteration, use current score as baseline
                        reward = virality_score
                        print(f"  Reward: {reward:.3f} (baseline)")
                    
                    evolution_history.append({
                        'iteration': i + 1,
                        'virality_score': virality_score,
                        'reward': reward,
                        'action_magnitude': metadata.get('action_magnitude', 0)
                    })
                    
                    optimizations.append({
                        'scene_number': i + 1,
                        'base_prompt': base_prompt,
                        'optimized_prompt': optimized_prompt,
                        'virality_score': virality_score,
                        'reward': reward,
                        'metadata': metadata
                    })
                
                # Show evolution summary
                if len(evolution_history) > 1:
                    print(f"\n{'='*60}")
                    print("Evolution Summary (Self-Learning Progress):")
                    print(f"{'='*60}")
                    for evo in evolution_history:
                        print(f"  Iteration {evo['iteration']}: Virality={evo['virality_score']:.3f}, Reward={evo['reward']:.3f}, Action={evo['action_magnitude']:.3f}")
                    
                    # Show improvement trend
                    scores = [e['virality_score'] for e in evolution_history]
                    if len(scores) > 1:
                        improvement = scores[-1] - scores[0]
                        print(f"\n  Overall improvement: {improvement:+.3f}")
                        print(f"  Learning trend: {'↗ IMPROVING' if improvement > 0 else '↘ DECLINING' if improvement < 0 else '→ STABLE'}")
                
                results['optimizations'] = optimizations
                results['evolution_history'] = evolution_history
                print(f"\n✓ Self-evolution complete! RL agent has learned from {len(evolution_history)} iterations.")
            
            results['success'] = True
            
        except Exception as e:
            results['error'] = str(e)
            print(f"Workflow error: {e}")
        
        return results
    
    def execute_sync(self, *args, **kwargs):
        """Synchronous wrapper for async workflow."""
        return asyncio.run(self.execute_workflow(*args, **kwargs))

