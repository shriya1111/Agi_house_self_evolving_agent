#!/usr/bin/env python3
"""Basic test to verify all components can be initialized."""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from src.config import config
        print("✓ Config loaded")
        
        # Check API keys (don't print actual keys)
        has_openai = bool(config.get_api_key('openai')) and config.get_api_key('openai') != 'your_openai_api_key_here'
        has_firecrawl = bool(config.get_api_key('firecrawl')) and config.get_api_key('firecrawl') != 'your_firecrawl_api_key_here'
        has_wandb = bool(config.get_api_key('wandb')) and config.get_api_key('wandb') != 'your_wandb_api_key_here'
        
        print(f"  OpenAI API Key: {'✓ Set' if has_openai else '✗ Missing'}")
        print(f"  Firecrawl API Key: {'✓ Set' if has_firecrawl else '✗ Missing'}")
        print(f"  W&B API Key: {'✓ Set' if has_wandb else '✗ Missing'}")
        
        return has_openai and has_firecrawl and has_wandb
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_imports():
    """Test all imports."""
    print("\nTesting imports...")
    modules = [
        ('Config', 'src.config'),
        ('Audio Transcriber', 'src.audio.transcriber'),
        ('Context Extractor', 'src.context.extractor'),
        ('Image Generator', 'src.generation.image_generator'),
        ('Virality Evaluator', 'src.evaluation.virality_evaluator'),
        ('RL Optimizer', 'src.rl.prompt_optimizer'),
        ('Metrics Logger', 'src.observability.metrics'),
        ('Orchestrator', 'src.orchestration.orchestrator'),
    ]
    
    results = []
    for name, module_path in modules:
        try:
            __import__(module_path)
            print(f"✓ {name}")
            results.append(True)
        except Exception as e:
            print(f"✗ {name}: {e}")
            results.append(False)
    
    return all(results)

def test_component_initialization():
    """Test component initialization (without API calls)."""
    print("\nTesting component initialization...")
    results = []
    
    try:
        # Test Config
        from src.config import config
        print("✓ Config initialized")
        results.append(True)
    except Exception as e:
        print(f"✗ Config initialization failed: {e}")
        results.append(False)
    
    try:
        # Test Metrics Logger (should work even without W&B key for disabled mode)
        from src.observability.metrics import MetricsLogger
        # This will try to initialize but may fail silently if no key
        print("✓ Metrics Logger module loaded")
        results.append(True)
    except Exception as e:
        print(f"✗ Metrics Logger failed: {e}")
        results.append(False)
    
    return all(results)

if __name__ == '__main__':
    print("=" * 60)
    print("Basic System Test")
    print("=" * 60)
    
    # Test 1: Configuration
    config_ok = test_config()
    
    # Test 2: Imports
    imports_ok = test_imports()
    
    # Test 3: Component initialization
    init_ok = test_component_initialization()
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Configuration: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Initialization: {'✓ PASS' if init_ok else '✗ FAIL'}")
    
    if config_ok and imports_ok and init_ok:
        print("\n✓ All basic tests passed!")
        print("\nNext steps:")
        print("1. Make sure all API keys are set in .env file")
        print("2. Prepare an audio file (MP3, WAV, etc.)")
        print("3. Run: python main.py path/to/audio.mp3")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

