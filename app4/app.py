"""
Demo script for the Multimodal LLM Workflow.
This script demonstrates how to use the workflow with sample data.
"""

import os
import sys
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import wave
import struct

# Add the main module to path
sys.path.append('.')

def create_sample_image(filename="sample_image.jpg"):
    """Create a sample image for testing."""
    # Create a simple image with text and shapes
    img = Image.new('RGB', (400, 300), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black', width=2)
    draw.ellipse([200, 50, 300, 150], fill='green', outline='black', width=2)
    draw.polygon([(100, 200), (150, 170), (200, 200), (175, 250), (125, 250)], 
                fill='yellow', outline='black')
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        draw.text((10, 10), "Sample Image for Testing", fill='black', font=font)
        draw.text((10, 260), "Shapes: Rectangle, Circle, Star", fill='black', font=font)
    except:
        # Fallback if font loading fails
        draw.text((10, 10), "Sample Image for Testing", fill='black')
        draw.text((10, 260), "Shapes: Rectangle, Circle, Star", fill='black')
    
    img.save(filename)
    print(f"Created sample image: {filename}")
    return filename

def create_sample_audio(filename="sample_audio.wav", duration=3, text="Hello, this is a test audio file for the multimodal workflow."):
    """Create a sample audio file (simple tone with description)."""
    # Create a simple sine wave
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more complex waveform (multiple frequencies)
    frequency1 = 440  # A note
    frequency2 = 554  # C# note
    audio_data = (np.sin(2 * np.pi * frequency1 * t) * 0.3 + 
                 np.sin(2 * np.pi * frequency2 * t) * 0.2)
    
    # Add some envelope to make it sound more natural
    envelope = np.exp(-t * 0.5)
    audio_data *= envelope
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created sample audio: {filename}")
    print(f"Note: This is a synthesized tone. For speech recognition testing,")
    print(f"      you may want to record actual speech or use a TTS system.")
    return filename

def download_sample_image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png", 
                         filename="downloaded_sample.png"):
    """Download a sample image from the internet."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        print(f"Downloaded sample image: {filename}")
        return filename
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None

def run_comprehensive_demo():
    """Run a comprehensive demo of the multimodal workflow."""
    print("=" * 60)
    print("MULTIMODAL LLM WORKFLOW DEMO")
    print("=" * 60)
    
    try:
        # Import the workflow (will only work if dependencies are installed)
        from workflow import MultimodalLLMWorkflow
        
        # Create sample data
        print("\n1. Creating sample data...")
        image_file = create_sample_image()
        audio_file = create_sample_audio()
        
        # Initialize workflow
        print("\n2. Initializing multimodal workflow...")
        workflow = MultimodalLLMWorkflow()
        
        # Test text processing
        print("\n3. Testing text processing...")
        text_queries = [
            "What is machine learning?",
            "Explain computer vision in simple terms.",
            "How do neural networks work?"
        ]
        
        for query in text_queries:
            print(f"\nQuery: {query}")
            response = workflow.process_text(query, max_length=80)
            print(f"Response: {response}")
        
        # Test image processing
        print(f"\n4. Testing image processing with {image_file}...")
        image_queries = [
            "What shapes are in this image?",
            "What colors do you see?",
            "Describe this image"
        ]
        
        for query in image_queries:
            print(f"\nImage query: {query}")
            result = workflow.process_image(image_file, query)
            print(f"Caption: {result['caption']}")
            if 'similarity_score' in result:
                print(f"Query relevance: {result['similarity_score']:.3f}")
                print(f"Query match: {result['query_match']}")
        
        # Test audio processing
        print(f"\n5. Testing audio processing with {audio_file}...")
        try:
            audio_result = workflow.process_audio(audio_file)
            print(f"Transcription: {audio_result['transcription']}")
            print(f"Language: {audio_result['language']}")
            print("Note: This is a synthesized tone, so transcription may be empty or contain noise.")
        except Exception as e:
            print(f"Audio processing failed: {e}")
            print("This is expected for synthesized tones. Try with actual speech recordings.")
        
        # Test multimodal integration
        print(f"\n6. Testing multimodal integration...")
        multimodal_result = workflow.multimodal_chat(
            text_input="What do you see in this image?",
            image_path=image_file,
            # audio_path=audio_file  # Uncomment if you have actual speech audio
        )
        
        print(f"Processed modalities: {multimodal_result['modalities_processed']}")
        if 'text' in multimodal_result:
            print(f"Final response: {multimodal_result['text']['response']}")
        
        # Save results
        print(f"\n7. Saving results...")
        workflow.save_results(multimodal_result, "demo_results.txt")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("- Try with your own images and audio files")
        print("- Experiment with different queries")
        print("- Modify the workflow for your specific use case")
        print("- Check demo_results.txt for detailed output")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease install the required dependencies:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Please check your installation and try again.")

def quick_test():
    """Quick test to verify installation."""
    print("Testing basic imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
    
    try:
        import whisper
        print("✓ Whisper available")
    except ImportError:
        print("✗ Whisper not installed")
    
    try:
        import librosa
        print("✓ Librosa available")
    except ImportError:
        print("✗ Librosa not installed")
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow available")
    except ImportError:
        print("✗ PIL/Pillow not installed")
    
    print(f"\nCUDA available: {torch.cuda.is_available() if 'torch' in locals() else 'Unknown'}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal LLM Workflow Demo")
    parser.add_argument("--quick-test", action="store_true", help="Run quick installation test")
    parser.add_argument("--full-demo", action="store_true", help="Run full demo")
    parser.add_argument("--create-samples", action="store_true", help="Create sample files only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    elif args.create_samples:
        create_sample_image()
        create_sample_audio()
        print("Sample files created successfully!")
    elif args.full_demo:
        run_comprehensive_demo()
    else:
        print("Multimodal LLM Workflow Demo")
        print("Usage:")
        print("  python demo.py --quick-test     # Test installation")
        print("  python demo.py --create-samples # Create sample files")
        print("  python demo.py --full-demo      # Run complete demo")