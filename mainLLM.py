"""
Multimodal AI Assistant - Qwen2-VL
Speech → Text → Qwen2-VL → Response → Audio
"""

import argparse
import sys
import cv2
import torch
from PIL import Image
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from speech_to_text import SpeechToTextEngine
from text_to_speech import TextToSpeechEngine


PROMPT_TYPES = {
    "scene_description": {
        "keywords": ["what do you see", "what's in front", "describe the scene", "what's around me", "environment"],
        "prompt": "Describe what you see in this image.",
    },

    "object_identification": {
        "keywords": ["what is this", "what's that", "what is it", "identify", "what object", "what thing"],
        "prompt": "What is the main object in this image?",
    },
    "detailed_description": {
        "keywords": ["describe", "tell me about", "more details", "explain"],
        "prompt": "Describe the image in detail.",
    },
}


class Qwen2VLAssistant:

    def __init__(self, verbose: bool = False, stt_model_size: str = "tiny"):
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.verbose:
            print(f"[Core] Using device: {self.device}")

        print("[Core] Initializing Speech-to-Text Engine...")
        self.stt = SpeechToTextEngine(
            model_size=stt_model_size,
            device="auto",
            compute_type="int8",
            verbose=self.verbose,
        )

        print("[Core] Initializing Text-to-Speech Engine...")
        self.tts = TextToSpeechEngine(
            rate=180,
            volume=1.0,
            verbose=self.verbose,
        )

        print("[Core] Loading Qwen2-VL model...")

        try:
            MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

            # Use fp16 on GPU, fp32 on CPU
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(MODEL_ID)

            if self.verbose:
                print(f"[Model] Loaded: {MODEL_ID}")
            print("[Qwen2-VL] Model loaded ✅\n")
        except Exception as e:
            print(f"❌ Error loading Qwen2-VL: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def identify_prompt_type(self, user_text: str) -> str:
        """Identify which type of prompt the user asked."""
        user_lower = user_text.lower()

        for prompt_type, config in PROMPT_TYPES.items():
            for keyword in config["keywords"]:
                if keyword in user_lower:
                    if self.verbose:
                        print(f"[Prompt] Identified as: {prompt_type}")
                    return prompt_type

        return "scene_description"

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

    def generate_response(self, image: Image.Image, prompt_type: str) -> str:
        """Generate response using Qwen2-VL."""
        config = PROMPT_TYPES[prompt_type]
        prompt_text = config["prompt"]

        if self.verbose:
            print("\n[Qwen2-VL] Prompt:\n" + "-" * 40)
            print(prompt_text)
            print("-" * 40)

        try:
            # Build conversation in Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # 1) Turn messages into a text prompt with special tokens
            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            # 2) Pack text + image into model inputs
            inputs = self.processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # 3) Generate answer
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                )

            # 4) Remove the prompt tokens from the output
            generated_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
            ]

            # 5) Decode to text
            responses = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            response = responses[0].strip() if responses else ""

            if self.verbose:
                print("[Qwen2-VL] Generated response:\n", response)

            return response if response else "I couldn't generate a response."

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating response."

    def run_single_analysis(self, speech_input: str | None = None, image_path: str | None = None, duration: int = 5):
        """Run a single analysis cycle."""
        # Step 1: Get user input
        if speech_input is None:
            print("\n🎙 Please speak now...")
            user_text = self.stt.record_and_transcribe(duration=duration)
        else:
            user_text = speech_input.strip()

        if not user_text:
            print("❌ Could not understand.")
            return

        print(f"\n📝 You said: \"{user_text}\"")

        # Step 2: Identify prompt type
        prompt_type = self.identify_prompt_type(user_text)
        print(f"📌 Prompt type: {prompt_type}")

        # Step 3: Get image
        if image_path:
            print(f"\n📷 Loading image: {image_path}...")
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ Could not load image: {image_path}")
                return
        else:
            print("\n📷 Capturing image from camera...")
            frame = self.capture_frame()
            if frame is None:
                return

        image = self.frame_to_pil(frame)

        # Step 4: Generate response
        print("\n🤖 Generating response...")
        response = self.generate_response(image, prompt_type)
        print(f"\n✅ Assistant: {response}")

        # Step 5: Speak
        print("\n🔊 Speaking...")
        self.tts.speak(response, block=True)

    def run_interactive(self, duration: int = 5):
        """Run continuous loop."""
        print("\n" + "="*60)
        print("Qwen2-VL BLIND ASSISTANCE DEVICE")
        print("="*60)
        print(f"\nPress Enter to ask, or 'q' to quit.")
        print(f"Listening duration: {duration} seconds\n")

        while True:
            cmd = input("➡ Press Enter to ask (or 'q' to quit): ").strip().lower()
            if cmd in {"q", "quit"}:
                print("Bye! 👋")
                return

            try:
                self.run_single_analysis(duration=duration)
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

            print("\n" + "-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL AI Assistant")
    parser.add_argument(
        "--mode",
        choices=["single", "interactive"],
        default="interactive",
        help="Run mode: single analysis or interactive loop",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
        help="Text input for single mode (if not provided, uses microphone)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image file",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Microphone listening duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size for speech-to-text (default: tiny)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Qwen2-VL MULTIMODAL AI ASSISTANT")
    print("Speech → Text → Vision → Response → Audio")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda (GPU)' if torch.cuda.is_available() else 'cpu'}")
    print(f"STT Model: Whisper ({args.stt_model})")
    print(f"Listening Duration: {args.duration}s")
    print("="*60 + "\n")

    try:
        assistant = Qwen2VLAssistant(
            verbose=args.verbose,
            stt_model_size=args.stt_model,
        )

        if args.mode == "single":
            assistant.run_single_analysis(
                speech_input=args.test_input,
                image_path=args.image,
                duration=args.duration,
            )
        else:
            assistant.run_interactive(duration=args.duration)

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()