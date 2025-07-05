import os
import torch
import numpy as np
from PIL import Image
import librosa
import whisper
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    pipeline
)
import warnings
warnings.filterwarnings("ignore")

class MultimodalLLMWorkflow:
    """
    A basic multimodal LLM workflow that handles text, images, and audio inputs.
    Uses open-source models for each modality.
    """
    
    def __init__(self, device="auto"):
        """
        Initialize the multimodal workflow with necessary models.
        
        Args:
            device: Device to run models on ("auto", "cuda", "cpu")
        """
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._load_text_model()
        self._load_vision_model()
        self._load_audio_model()
        
    def _get_device(self, device):
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_text_model(self):
        """Load text generation model."""
        print("Loading text model...")
        model_name = "microsoft/DialoGPT-medium"  # Lightweight conversational model
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        self.text_model.to(self.device)
        print("Text model loaded successfully")
    
    def _load_vision_model(self):
        """Load vision understanding models."""
        print("Loading vision models...")
        
        # CLIP for image understanding
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        
        # BLIP for image captioning
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.to(self.device)
        
        print("Vision models loaded successfully")
    
    def _load_audio_model(self):
        """Load audio processing model."""
        print("Loading audio model...")
        
        # Whisper for speech-to-text
        self.whisper_model = whisper.load_model("base")
        
        print("Audio model loaded successfully")
    
    def process_text(self, text, max_length=100):
        """
        Process text input and generate response.
        
        Args:
            text: Input text string
            max_length: Maximum length of generated response
            
        Returns:
            Generated text response
        """
        # Encode input
        inputs = self.text_tokenizer.encode(text + self.text_tokenizer.eos_token, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.text_tokenizer.pad_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input text from response
        if text in response:
            response = response.replace(text, "").strip()
        
        return response
    
    def process_image(self, image_path, query=None):
        """
        Process image input and generate description or answer query.
        
        Args:
            image_path: Path to image file
            query: Optional text query about the image
            
        Returns:
            Dictionary with image analysis results
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        results = {}
        
        # Generate image caption
        inputs = self.blip_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        results["caption"] = caption
        
        # If query provided, use CLIP for image-text matching
        if query:
            inputs = self.clip_processor(
                text=[query], 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                similarity = torch.cosine_similarity(
                    outputs.text_embeds, 
                    outputs.image_embeds
                ).item()
            
            results["query"] = query
            results["similarity_score"] = similarity
            results["query_match"] = similarity > 0.25  # Threshold for match
        
        return results
    
    def process_audio(self, audio_path):
        """
        Process audio input and convert to text.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text from audio
        """
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Transcribe using Whisper
        result = self.whisper_model.transcribe(audio_path)
        
        return {
            "transcription": result["text"],
            "language": result.get("language", "unknown"),
            "confidence": result.get("confidence", 0.0)
        }
    
    def multimodal_chat(self, text_input=None, image_path=None, audio_path=None):
        """
        Process multimodal input and generate unified response.
        
        Args:
            text_input: Text input string
            image_path: Path to image file
            audio_path: Path to audio file
            
        Returns:
            Dictionary with processed results from all modalities
        """
        results = {"modalities_processed": []}
        
        # Process audio first (convert to text)
        if audio_path:
            audio_result = self.process_audio(audio_path)
            results["audio"] = audio_result
            results["modalities_processed"].append("audio")
            
            # Use transcribed text if no text input provided
            if not text_input:
                text_input = audio_result["transcription"]
        
        # Process image
        if image_path:
            # Use text input as query for image if available
            image_result = self.process_image(image_path, text_input)
            results["image"] = image_result
            results["modalities_processed"].append("image")
            
            # Combine image caption with text input for richer context
            if text_input:
                text_input = f"Image shows: {image_result['caption']}. User asks: {text_input}"
            else:
                text_input = f"Describe this image: {image_result['caption']}"
        
        # Process text (always last to incorporate other modalities)
        if text_input:
            text_result = self.process_text(text_input)
            results["text"] = {
                "input": text_input,
                "response": text_result
            }
            results["modalities_processed"].append("text")
        
        return results
    
    def save_results(self, results, output_file="multimodal_results.txt"):
        """Save processing results to file."""
        with open(output_file, "w") as f:
            f.write("Multimodal LLM Processing Results\n")
            f.write("=" * 40 + "\n\n")
            
            for modality in results["modalities_processed"]:
                f.write(f"{modality.upper()} RESULTS:\n")
                f.write("-" * 20 + "\n")
                
                if modality == "audio":
                    f.write(f"Transcription: {results['audio']['transcription']}\n")
                    f.write(f"Language: {results['audio']['language']}\n")
                    f.write(f"Confidence: {results['audio']['confidence']:.2f}\n\n")
                
                elif modality == "image":
                    f.write(f"Caption: {results['image']['caption']}\n")
                    if "query" in results['image']:
                        f.write(f"Query: {results['image']['query']}\n")
                        f.write(f"Similarity Score: {results['image']['similarity_score']:.3f}\n")
                        f.write(f"Query Match: {results['image']['query_match']}\n")
                    f.write("\n")
                
                elif modality == "text":
                    f.write(f"Input: {results['text']['input']}\n")
                    f.write(f"Response: {results['text']['response']}\n\n")
        
        print(f"Results saved to {output_file}")


def main():
    """
    Example usage of the multimodal LLM workflow.
    """
    # Initialize workflow
    workflow = MultimodalLLMWorkflow()
    
    # Example 1: Text-only processing
    print("\n=== Text Processing Example ===")
    text_response = workflow.process_text("What is artificial intelligence?")
    print(f"Response: {text_response}")
    
    # Example 2: Image processing (requires image file)
    print("\n=== Image Processing Example ===")
    # Uncomment and provide actual image path
    # image_results = workflow.process_image("path/to/image.jpg", "What is in this image?")
    # print(f"Image results: {image_results}")
    
    # Example 3: Audio processing (requires audio file)
    print("\n=== Audio Processing Example ===")
    # Uncomment and provide actual audio path
    # audio_results = workflow.process_audio("path/to/audio.wav")
    # print(f"Audio results: {audio_results}")
    
    # Example 4: Multimodal processing
    print("\n=== Multimodal Processing Example ===")
    # This would process all available modalities
    # multimodal_results = workflow.multimodal_chat(
    #     text_input="Describe what you see and hear",
    #     image_path="path/to/image.jpg",
    #     audio_path="path/to/audio.wav"
    # )
    # workflow.save_results(multimodal_results)
    
    print("\nMultimodal LLM workflow setup complete!")
    print("To use with actual files, uncomment the examples above and provide file paths.")


if __name__ == "__main__":
    main()