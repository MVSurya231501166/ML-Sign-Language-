import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
import json
from datetime import datetime
import os
import time
from torch.utils.data import Dataset, DataLoader

class AdvancedSignLanguageModel(nn.Module):
    def __init__(self, num_classes=26):
        super(AdvancedSignLanguageModel, self).__init__()
        
        # CNN Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),
        )
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(512 * 4 * 4, 256, num_layers=2, batch_first=True)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.features(x)
        x = x.view(batch_size, -1)
        
        # LSTM sequence processing
        x = x.unsqueeze(1)  # Add sequence dimension
        x, hidden = self.lstm(x, hidden)
        
        # Classification
        x = self.classifier(x.squeeze(1))
        return x, hidden

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def track_hands(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks.append(hand_landmarks)
        
        return frame, landmarks

class EnhancedSignLanguageRecognizer:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.IMG_SIZE = 64
        self.tts_engine = pyttsx3.init()
        self.hand_tracker = HandTracker()
        
        # Initialize model
        self.model = AdvancedSignLanguageModel().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Initialize sequence detection
        self.sequence_buffer = []
        self.word_buffer = []
        self.last_prediction_time = time.time()
        
        # Load common words dictionary
        self.word_dict = self.load_word_dictionary()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_word_dictionary(self):
        """Load or create a dictionary of common words and their letter sequences"""
        try:
            with open('sign_language_words.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Basic dictionary - expand this
            return {
                "HELLO": ["H", "E", "L", "L", "O"],
                "WORLD": ["W", "O", "R", "L", "D"],
                "THANK": ["T", "H", "A", "N", "K"],
                "YOU": ["Y", "O", "U"]
            }

    def detect_word(self):
        """Detect if current sequence matches any known words"""
        sequence_str = ''.join(self.sequence_buffer)
        for word, letters in self.word_dict.items():
            if ''.join(letters) == sequence_str:
                return word
        return None

class SignLanguageGUI:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.root = tk.Tk()
        self.root.title("Sign Language Recognition System")
        self.setup_gui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.queue = queue.Queue()
        self.is_running = True
        
        # Start video capture thread
        self.thread = threading.Thread(target=self.video_capture)
        self.thread.daemon = True
        self.thread.start()
        
        # Configure text-to-speech
        self.tts_queue = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker)
        self.tts_thread.daemon = True
        self.tts_thread.start()

    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2)
        
        # Prediction display
        self.pred_var = tk.StringVar(value="Prediction: ")
        pred_label = ttk.Label(main_frame, textvariable=self.pred_var)
        pred_label.grid(row=1, column=0, columnspan=2)
        
        # Word sequence display
        self.word_var = tk.StringVar(value="Word: ")
        word_label = ttk.Label(main_frame, textvariable=self.word_var)
        word_label.grid(row=2, column=0, columnspan=2)
        
        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=3, column=0, columnspan=2)
        
        ttk.Button(controls_frame, text="Toggle Speech", 
                  command=self.toggle_speech).grid(row=0, column=0, padx=5)
        ttk.Button(controls_frame, text="Clear Sequence", 
                  command=self.clear_sequence).grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="Save Word", 
                  command=self.save_word).grid(row=0, column=2, padx=5)
        
        # Settings
        self.setup_settings_panel(main_frame)

    def setup_settings_panel(self, parent):
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        settings_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0)
        self.conf_threshold = tk.DoubleVar(value=0.8)
        conf_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                             variable=self.conf_threshold, orient=tk.HORIZONTAL)
        conf_scale.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Sequence timeout
        ttk.Label(settings_frame, text="Sequence Timeout (s):").grid(row=1, column=0)
        self.seq_timeout = tk.DoubleVar(value=2.0)
        timeout_scale = ttk.Scale(settings_frame, from_=0.5, to=5.0, 
                                variable=self.seq_timeout, orient=tk.HORIZONTAL)
        timeout_scale.grid(row=1, column=1, sticky=(tk.W, tk.E))

    def video_capture(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Process frame with hand tracking
                frame, landmarks = self.recognizer.hand_tracker.track_hands(frame)
                
                # Convert frame for display
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.queue.put(imgtk)
                
                # Debug print
                print("Frame captured and put in queue")
                
                # Update display
                self.root.after(0, self.update_frame)

    def update_frame(self):
        try:
            imgtk = self.queue.get_nowait()
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            # Debug print
            print("Frame updated in GUI")
        except queue.Empty:
            print("Queue empty, no frame to update")

    def tts_worker(self):
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=1)
                self.recognizer.tts_engine.say(text)
                self.recognizer.tts_engine.runAndWait()
            except queue.Empty:
                continue

    def toggle_speech(self):
        # Toggle text-to-speech functionality
        pass

    def clear_sequence(self):
        self.recognizer.sequence_buffer.clear()
        self.recognizer.word_buffer.clear()
        self.word_var.set("Word: ")

    def save_word(self):
        # Save current sequence as a new word
        pass

    def run(self):
        self.root.mainloop()
        self.is_running = False
        self.cap.release()

def main():
    recognizer = EnhancedSignLanguageRecognizer()
    gui = SignLanguageGUI(recognizer)
    gui.run()

if __name__ == "__main__":
    main()