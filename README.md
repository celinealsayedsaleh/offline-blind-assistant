# Offline Blind Assistant

An offline AI assistive system designed to support visually impaired users through speech recognition, text-to-speech, object detection, and local language model modules.

## Project Overview

This project combines multiple AI and accessibility components into one assistive system. The goal is to help users interact with their environment and receive spoken feedback without depending on an internet connection.

The system includes separate modules for:

- Speech-to-text input
- Text-to-speech output
- Object detection using YOLO
- Local language model interaction
- GPT-2-based text generation/response handling

## Technologies Used

- Python
- Speech Recognition
- Text-to-Speech
- YOLO Object Detection
- Local Language Models
- GPT-2
- Machine Learning
- Assistive Technology

## Features

- Converts spoken input into text
- Converts system responses into spoken audio
- Detects objects using a YOLO-based module
- Supports offline/local AI processing
- Organizes the system into separate Python modules
- Designed for low-resource and accessibility-focused use cases

## Repository Structure

```text
offline-blind-assistant/
│
├── mainGPT2.py
├── mainLLM.py
├── mainYOLO.py
├── speech_to_text.py
├── text_to_speech.py
├── .gitignore
└── README.md
```
