#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

print("Testing imports...")

try:
    from src.config import config
    print("✓ Config module imported")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from src.audio.transcriber import AudioTranscriber
    print("✓ Audio transcriber imported")
except Exception as e:
    print(f"✗ Audio transcriber import failed: {e}")

try:
    from src.context.extractor import ContextExtractor
    print("✓ Context extractor imported")
except Exception as e:
    print(f"✗ Context extractor import failed: {e}")

try:
    from src.generation.image_generator import ImageGenerator
    print("✓ Image generator imported")
except Exception as e:
    print(f"✗ Image generator import failed: {e}")

try:
    from src.evaluation.virality_evaluator import ViralityEvaluator
    print("✓ Virality evaluator imported")
except Exception as e:
    print(f"✗ Virality evaluator import failed: {e}")

try:
    from src.rl.prompt_optimizer import RLPromptOptimizer
    print("✓ RL prompt optimizer imported")
except Exception as e:
    print(f"✗ RL prompt optimizer import failed: {e}")

try:
    from src.observability.metrics import MetricsLogger
    print("✓ Metrics logger imported")
except Exception as e:
    print(f"✗ Metrics logger import failed: {e}")

try:
    from src.orchestration.orchestrator import AgentOrchestrator
    print("✓ Agent orchestrator imported")
except Exception as e:
    print(f"✗ Agent orchestrator import failed: {e}")

print("\nImport test complete!")

