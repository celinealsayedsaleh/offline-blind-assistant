"""
Multimodal AI Assistant - ViT-GPT2
Speech → Text → ViT-GPT2 → Response → Audio
"""

import argparse
import sys
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

from speech_to_text import SpeechToTextEngine
from text_to_speech import TextToSpeechEngine

VALID_KEYWORDS = [
    "what is this",
    "what's this",
    "what is that",
    "what's that",
    "what do you see",
    "describe this",
    "describe the scene",
    "what's in front of me",
    "what is in front of me",
    "tell me what this is",
    "what object",
    "identify this",
]

class ViTGPT2Assistant:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.verbose:
            print(f"[Core] Using device: {self.device}")

        print("[Core] Initializing Speech-to-Text Engine...")
        self.stt = SpeechToTextEngine(
            model_size="small",
            device="auto",
            compute_type="float16",
            verbose=self.verbose,
        )

        print("[Core] Initializing Text-to-Speech Engine...")
        self.tts = TextToSpeechEngine(
            rate=180,
            volume=1.0,
            verbose=self.verbose,
        )

        print("[Core] Loading ViT-GPT2 model...")

        try:
            model_name = "nlpconnect/vit-gpt2-image-captioning"
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            self.model.to(self.device)
            print("[ViT-GPT2] Model loaded ✅\n")
        except Exception as e:
            print(f"❌ Error loading ViT-GPT2: {e}")
            sys.exit(1)

    def is_valid_vision_request(self, user_text: str) -> bool:
        """Return True if the user asked a recognized vision question."""
        text = user_text.lower().strip()

        for kw in VALID_KEYWORDS:
            if kw in text:
                return True
        return False

    def capture_frame(self, camera_index: int = 0):
        """Capture a single frame from the specified camera."""
        if self.verbose:
            print("[Camera] Opening camera...")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[Camera] ERROR: Could not open camera.")
            return None

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("[Camera] ERROR: Failed to capture frame.")
            return None

        if self.verbose:
            print("[Camera] Frame captured ✅")

        return frame

    def frame_to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        """Convert OpenCV BGR frame to PIL RGB image."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def generate_response(self, image: Image.Image) -> str:
        """Generate an image caption using ViT-GPT2."""
        if self.verbose:
            print("\n[ViT-GPT2] Generating caption...")

        try:
            # Preprocess image
            inputs = self.processor(
                image,
                return_tensors="pt",
                padding=True,
            )

            pixel_values = inputs["pixel_values"].to(self.device)

            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=50,
                    num_beams=3,
                )

            caption = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )[0].strip()

            if self.verbose:
                print("[ViT-GPT2] Generated caption:", caption)

            return caption if caption else "I couldn't describe the image."

        except Exception as e:
            print(f"❌ Error generating caption: {e}")
            return "Error generating response."

    def run_single_analysis(self, speech_input: str | None = None, image_path: str | None = None):
        """Run one cycle of STT → image → caption → TTS."""

        # Step 1: Get user speech input
        if speech_input is None:
            print("\n🎙 Please speak (5 seconds)...")
            user_text = self.stt.record_and_transcribe(duration=5)
        else:
            user_text = speech_input.strip()

        if not user_text:
            print("❌ Could not understand.")
            return

        print(f"\n📝 You said: \"{user_text}\"")

        # Step 2: Validate request
        if not self.is_valid_vision_request(user_text):
            print("❌ Could not understand. Please try again.")
            return

        print("📌 Valid request: Capturing image and analyzing...\n")

        # Step 3: Get image
        if image_path:
            print(f"📷 Loading image from {image_path}...")
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ Could not load: {image_path}")
                return
        else:
            print("📷 Capturing image from camera...")
            frame = self.capture_frame()
            if frame is None:
                return

        image = self.frame_to_pil(frame)

        # Step 4: Generate caption
        print("🤖 Generating response...")
        response = self.generate_response(image)
        print(f"\n✅ Assistant: {response}")

        # Step 5: Speak the caption
        print("\n🔊 Speaking...")
        self.tts.speak(response, block=True)

    def run_interactive(self):
        """Run continuous interactive mode."""
        print("\n" + "="*60)
        print("ViT-GPT2 BLIND ASSISTANCE DEVICE")
        print("="*60)
        print("\nPress Enter to ask, or 'q' to quit.\n")

        while True:
            cmd = input("➡ Press Enter or 'q' to quit: ").strip().lower()

            if cmd in {"q", "quit"}:
                print("Bye! 👋")
                return

            try:
                self.run_single_analysis()
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return
            except Exception as e:
                print(f"\n❌ Error: {e}")

            print("\n" + "-"*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="ViT-GPT2 AI Assistant")
    parser.add_argument(
        "--mode",
        choices=["single", "interactive"],
        default="interactive",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ViT-GPT2 MULTIMODAL AI ASSISTANT")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda (GPU)' if torch.cuda.is_available() else 'cpu'}")
    print("="*60 + "\n")

    try:
        assistant = ViTGPT2Assistant(verbose=args.verbose)

        if args.mode == "single":
            assistant.run_single_analysis(speech_input=args.test_input, image_path=args.image)
        else:
            assistant.run_interactive()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
