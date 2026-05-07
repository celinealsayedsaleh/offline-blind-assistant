"""
Multimodal AI Assistant - YOLO + OCR
Speech → Text → YOLO/OCR → Response → Audio
"""

import sys
import argparse
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr

from speech_to_text import SpeechToTextEngine
from text_to_speech import TextToSpeechEngine


PROMPT_TYPES = {
    "scene_description": {
        "keywords": ["what do you see", "what's in front", "describe the scene", "what is around me", "environment"],
        "use_yolo": True,
        "use_ocr": False,
    },
    "read_text": { 
        "keywords": ["read", "what does it say", "text", "words" ], 
        "use_yolo": False,
        "use_ocr": True,
    },
    "navigation_help": {
        "keywords": ["navigate", "help me", "which way", "path", "where", "direction", "can i go", "is it safe", "obstacle"],
        "use_yolo": True,
        "use_ocr": False,
    },
    "detailed_description": {
        "keywords": ["describe", "tell me about", "more details", "explain", "everything"],
        "use_yolo": True,
        "use_ocr": True,
    },
}

class YOLOOCRAssistant:
    """Uses YOLO for object detection and EasyOCR for text recognition."""

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

        print("[Core] Loading YOLO model...")
        try:
            # YOLOv8n is the smallest and fastest model
            self.yolo_model = YOLO('yolov8n.pt')
            print("[YOLO] Model loaded ✅")
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            sys.exit(1)

        print("[Core] Loading OCR model...")
        try:
            # EasyOCR with English support
            self.ocr_reader = easyocr.Reader(['en'], gpu=(self.device == "cuda"))
            print("[OCR] Model loaded ✅\n")
        except Exception as e:
            print(f"❌ Error loading OCR: {e}")
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

    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5):
        """Detect objects using YOLO."""
        if self.verbose:
            print("[YOLO] Running object detection...")

        results = self.yolo_model(frame, verbose=False)
        
        # Save labeled YOLO output
        labeled = results[0].plot()            # This draws boxes on the frame
        cv2.imwrite("labeled.jpg", labeled)    # Save it to file

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    detections.append({
                        'name': class_name,
                        'confidence': confidence,
                        'box': box.xyxy[0].cpu().numpy()
                    })

        if self.verbose:
            print(f"[YOLO] Found {len(detections)} objects")

        return detections

    def read_text(self, frame: np.ndarray):
        """Read text using OCR."""
        if self.verbose:
            print("[OCR] Reading text...")

        # Convert BGR to RGB for EasyOCR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.ocr_reader.readtext(rgb_frame)
        
        texts = []
        for (bbox, text, confidence) in results:
            if confidence >= 0.1:
                texts.append({
                    'text': text,
                    'confidence': confidence
                })

        if self.verbose:
            print(f"[OCR] Found {len(texts)} text regions")

        return texts

    def calculate_distances(self, detections):
        """Estimate relative positions (left, center, right, near, far)."""
        if not detections:
            return detections

        frame_width = 640  # Assume standard width
        
        for det in detections:
            box = det['box']
            x_center = (box[0] + box[2]) / 2
            box_width = box[2] - box[0]
            
            # Horizontal position
            if x_center < frame_width * 0.33:
                det['position'] = "on your left"
            elif x_center > frame_width * 0.67:
                det['position'] = "on your right"
            else:
                det['position'] = "in front of you"
            
            # Relative size/distance
            if box_width > frame_width * 0.4:
                det['distance'] = "very close"
            elif box_width > frame_width * 0.2:
                det['distance'] = "nearby"
            else:
                det['distance'] = "in the distance"

        return detections

    def generate_response(self, frame: np.ndarray, prompt_type: str) -> str:
        """Generate response using YOLO and/or OCR."""
        config = PROMPT_TYPES[prompt_type]
        use_yolo = config["use_yolo"]
        use_ocr = config["use_ocr"]

        response_parts = []

        # Object detection
        if use_yolo:
            detections = self.detect_objects(frame)
            detections = self.calculate_distances(detections)
            
            if detections:
                if prompt_type == "navigation_help":
                    # Focus on obstacles
                    obstacles = [d for d in detections if d['name'] in 
                                ['person', 'chair', 'bench', 'car', 'bicycle', 'dog', 'cat']]
                    
                    if obstacles:
                        response_parts.append("Warning! Obstacles detected:")
                        for obj in obstacles[:3]:  # Top 3 obstacles
                            response_parts.append(
                                f"{obj['name']} {obj['distance']} {obj['position']}"
                            )
                    else:
                        response_parts.append("The path appears clear.")
                
                elif prompt_type == "object_identification":
                    # Identify main object
                    main_object = max(detections, key=lambda x: x['confidence'])
                    response_parts.append(
                        f"I see a {main_object['name']} {main_object['position']}"
                    )
                
                else:
                    # Scene description
                    object_counts = {}
                    for det in detections:
                        name = det['name']
                        object_counts[name] = object_counts.get(name, 0) + 1
                    
                    if object_counts:
                        response_parts.append("I can see:")
                        for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                            if count > 1:
                                response_parts.append(f"{count} {obj}s")
                            else:
                                response_parts.append(f"a {obj}")
            else:
                response_parts.append("I don't see any recognizable objects.")

        # Text recognition
        if use_ocr:
            texts = self.read_text(frame)
            
            if texts:
                if use_yolo:
                    response_parts.append("Text detected:")
                
                for text_info in texts: 
                    response_parts.append(text_info['text'])
            elif not use_yolo:
                response_parts.append("I couldn't find any readable text.")

        if not response_parts:
            return "I couldn't analyze the image."

        return " ".join(response_parts)

    def run_single_analysis(self, speech_input: str | None = None, image_path: str | None = None):
        """Run a single analysis cycle."""
        # Step 1: Get user input
        if speech_input is None:
            print("\n🎙 Please speak (5 seconds)...")
            user_text = self.stt.record_and_transcribe(duration=5)
        else:
            user_text = speech_input.strip()

        if not user_text:
            print("❌ Could not understand.")
            return

        print(f"\n🔍 You said: \"{user_text}\"")
        # Step 2: Identify prompt type
        prompt_type = self.identify_prompt_type(user_text)
        print(f"📌 Prompt type: {prompt_type}")

        # Step 3: Get image (from camera or file)
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
            cv2.imwrite("captured.jpg", frame)
        # Step 4: Generate response
        print("\n🤖 Analyzing image...")
        response = self.generate_response(frame, prompt_type)
        print(f"\n✅ Assistant: {response}")

        # Step 5: Speak
        print("\n🔊 Speaking...")
        self.tts.speak(response, block=True)

    def run_interactive(self):
        """Run continuous loop."""
        print("\n" + "="*60)
        print("YOLO + OCR BLIND ASSISTANCE DEVICE")
        print("="*60)
        print("\nPress Enter to ask, or 'q' to quit.\n")

        while True:
            cmd = input("➡ Press Enter or 'q' to quit: ").strip().lower()
            if cmd in {"q", "quit"}:
                print("Bye! 👋")
                break

            try:
                self.run_single_analysis()
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

            print("\n" + "-"*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="YOLO + OCR AI Assistant")
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
    print("YOLO + OCR MULTIMODAL AI ASSISTANT")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Device: {'cuda (GPU)' if torch.cuda.is_available() else 'cpu'}")
    print("="*60 + "\n")

    try:
        assistant = YOLOOCRAssistant(verbose=args.verbose)

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
