#!/usr/bin/env python3
"""Test the workflow without an actual audio file."""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

def test_transcription_simulation():
    """Test with a simulated transcription."""
    print("=" * 60)
    print("Testing Workflow (Simulated)")
    print("=" * 60)
    
    try:
        # Initialize components
        print("\n1. Initializing components...")
        from src.audio.transcriber import AudioTranscriber
        from src.context.extractor import ContextExtractor
        from src.observability.metrics import MetricsLogger
        
        # Use W&B in disabled mode for testing
        metrics = MetricsLogger() if False else None
        
        print("✓ Components initialized")
        
        # Simulate transcription
        print("\n2. Simulating transcription...")
        simulated_transcription = """
        I want to create a fast-paced tech product demo video. 
        The video should be energetic and dynamic, with bright colors and modern aesthetics.
        Show the product features in an exciting way. Use vibrant color palette with blues and oranges.
        The vibe should be professional but fun, targeting tech-savvy professionals.
        """
        
        print(f"Simulated transcription: {simulated_transcription[:100]}...")
        print("✓ Transcription simulated")
        
        # Test context extraction
        print("\n3. Testing context extraction...")
        try:
            context_extractor = ContextExtractor(use_wandb=False)
            
            # This will make an actual API call to OpenAI
            print("   Calling OpenAI GPT-4o...")
            context = context_extractor.extract_context(simulated_transcription)
            
            if context and 'error' not in context:
                print("✓ Context extracted successfully!")
                print(f"   Intent: {context.get('intent', 'N/A')}")
                print(f"   Vibe: {context.get('vibe', 'N/A')}")
                print(f"   Style: {context.get('content_style', 'N/A')}")
                return True
            else:
                print(f"✗ Context extraction failed: {context.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"✗ Context extraction error: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing workflow components...")
    print("Note: This will make an actual API call to OpenAI\n")
    
    success = test_transcription_simulation()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Workflow test passed!")
        print("\nYou're ready to test with a real audio file!")
        print("Run: python main.py path/to/your/audio.mp3")
    else:
        print("✗ Workflow test failed")
        print("Please check the errors above")
    print("=" * 60)

