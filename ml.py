import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pyttsx3
from datetime import datetime
import os
import time

class SignLanguageModel(nn.Module):
    def __init__(self, num_classes=26):
        super(SignLanguageModel, self).__init__()
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SignLanguageRecognizer:
    def __init__(self, model_path=None, camera_source=0, camera_api=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.IMG_SIZE = 64
        self.camera_source = camera_source
        self.camera_api = camera_api
        
        # Initialize text-to-speech engine with error handling
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            self.tts_engine = None
        
        # Initialize model
        self.model = SignLanguageModel().to(self.device)
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model weights from {model_path}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print("No model path provided or file doesn't exist. Using untrained model.")
            print("Predictions won't be accurate without a trained model.")
            
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                       'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """Preprocess a single image for prediction"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img.unsqueeze(0).to(self.device)

    def test_camera_connection(self, camera_source, api=None):
        """Test if a camera can be opened and return a frame"""
        if api is not None:
            cap = cv2.VideoCapture(camera_source, api)
        else:
            cap = cv2.VideoCapture(camera_source)
            
        if not cap.isOpened():
            cap.release()
            return False, None
            
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        return ret, frame

    def predict_real_time(self):
        """Real-time prediction from webcam feed"""
        self.model.eval()
        
        # Try to connect to the camera
        connection_successful = False
        cap = None
        
        # If camera_source is a string, assume it's an IP camera URL
        if isinstance(self.camera_source, str):
            print(f"Trying to connect to IP camera: {self.camera_source}")
            cap = cv2.VideoCapture(self.camera_source)
            connection_successful = cap.isOpened()
        else:
            # Try with specified API if provided
            if self.camera_api is not None:
                print(f"Trying camera index {self.camera_source} with specific API...")
                cap = cv2.VideoCapture(self.camera_source, self.camera_api)
                connection_successful = cap.isOpened()
            
            # If that fails, try without API
            if not connection_successful:
                print(f"Trying camera index {self.camera_source} with default API...")
                cap = cv2.VideoCapture(self.camera_source)
                connection_successful = cap.isOpened()
        
        # If still not connected, try additional camera indices
        if not connection_successful:
            print("Failed to connect with given settings. Trying other camera indices...")
            for idx in range(5):  # Try first 5 camera indices
                print(f"Trying camera index {idx}...")
                ret, _ = self.test_camera_connection(idx)
                if ret:
                    print(f"Successfully connected to camera index {idx}")
                    cap = cv2.VideoCapture(idx)
                    connection_successful = True
                    break
        
        # If still not connected, try DroidCam common URLs
        if not connection_successful:
            print("Still not connected. Trying common DroidCam URLs...")
            common_ips = [
                "http://192.168.29.192:4747/video",  # Your previous attempt
                "http://192.168.29.192:4747/mjpegfeed",  # Alternative feed type
                "http://192.168.29.192:4747/videofeed",  # Alternative feed type
                "http://127.0.0.1:4747/video",  # Localhost
                "http://localhost:4747/video"   # Localhost alternative
            ]
            
            for ip in common_ips:
                print(f"Trying DroidCam URL: {ip}")
                cap = cv2.VideoCapture(ip)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Successfully connected to {ip}")
                        connection_successful = True
                        break
                    else:
                        cap.release()
        
        # If still not connected, ask for manual input
        if not connection_successful:
            print("\nAll automatic connection attempts failed.")
            print("\nPlease ensure DroidCam is running and note the IP address and port.")
            print("In the DroidCam app, check the WiFi IP address (like 192.168.x.x:4747)")
            print("\nEnter DroidCam URL (or type 'exit' to quit):")
            print("Format examples:")
            print("- http://192.168.x.x:4747/video")
            print("- http://192.168.x.x:4747/videofeed")
            print("- http://192.168.x.x:4747/mjpegfeed")
            
            user_input = input().strip()
            if user_input.lower() == 'exit':
                return
            
            print(f"Trying to connect to: {user_input}")
            cap = cv2.VideoCapture(user_input)
            connection_successful = cap.isOpened()
            
            if connection_successful:
                # Test that we can actually read frames
                ret, frame = cap.read()
                if not ret:
                    print("Connected but couldn't read frames. Please check your DroidCam setup.")
                    cap.release()
                    return
            else:
                print("Could not connect to camera. Exiting.")
                return
        
        print("\nCamera connection successful!")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to speak the detected letter")
        print("- Press 'r' to recenter the ROI")
        
        prediction_queue = []
        roi_x, roi_y, roi_size = 100, 100, 300  # Default ROI settings
        
        # Skip creating named window with WINDOW_NORMAL flag
        # Instead, use the default window creation that happens when using imshow
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    time.sleep(0.5)  # Wait a bit before trying again
                    continue
                
                # Get frame dimensions for ROI calculation
                height, width = frame.shape[:2]
                
                # Make sure ROI is within frame bounds
                if roi_x + roi_size > width:
                    roi_x = max(0, width - roi_size)
                if roi_y + roi_size > height:
                    roi_y = max(0, height - roi_size)
                
                # Create ROI for hand gestures
                if roi_x + roi_size <= width and roi_y + roi_size <= height and roi_size > 0:
                    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
                else:
                    # Fallback if ROI is out of bounds
                    roi_size = min(width, height) // 2
                    roi_x = (width - roi_size) // 2
                    roi_y = (height - roi_size) // 2
                    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
                
                # Ensure ROI has content
                if roi.size == 0:
                    print("Warning: ROI has no content. Adjusting...")
                    roi = frame[0:min(300, height), 0:min(300, width)]
                    if roi.size == 0:
                        continue  # Skip this frame if still no content
                
                # Add instruction text
                cv2.putText(frame, "Place hand in green box", 
                          (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                try:
                    # Preprocess and predict
                    processed_img = self.preprocess_image(roi)
                    outputs = self.model(processed_img)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[0][predicted_class].item()
                    
                    predicted_letter = self.classes[predicted_class]
                    
                    # Add prediction to queue for smoothing
                    prediction_queue.append(predicted_letter)
                    if len(prediction_queue) > 10:
                        prediction_queue.pop(0)
                    
                    # Get most common prediction from queue
                    if len(prediction_queue) >= 5:
                        final_prediction = max(set(prediction_queue), key=prediction_queue.count)
                        
                        # Display prediction and confidence
                        cv2.putText(frame, f"Prediction: {final_prediction}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                
                # Use standard imshow without WINDOW_NORMAL flag
                try:
                    cv2.imshow('Sign Language Recognition', frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    print("Continuing to process frames without display...")
                    # If display fails, we'll still continue processing
                    # This can happen with some OpenCV builds or remote connections
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and len(prediction_queue) >= 5 and self.tts_engine:
                    try:
                        self.tts_engine.say(final_prediction)
                        self.tts_engine.runAndWait()
                    except Exception as e:
                        print(f"TTS error: {e}")
                elif key == ord('r'):
                    # Reset ROI to center
                    roi_size = min(width, height) // 2
                    roi_x = (width - roi_size) // 2
                    roi_y = (height - roi_size) // 2
                    print("ROI recentered")
        
        cap.release()
        cv2.destroyAllWindows()

def list_available_cameras():
    """List all available camera indices by trying to open them"""
    available_cameras = []
    # Try several camera indices
    for i in range(5):  # Check first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera index {i} is available and working")
                available_cameras.append(i)
            else:
                print(f"Camera index {i} is available but not returning frames")
            cap.release()
        else:
            print(f"Camera index {i} is not available")
    
    return available_cameras

# Example usage
if __name__ == "__main__":
    model_path = None
    camera_source = 0
    camera_api = None
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].startswith("http"):
            camera_source = sys.argv[1]  # Use as URL
        else:
            model_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        if sys.argv[2].startswith("http"):
            camera_source = sys.argv[2]  # Use as URL
        else:
            try:
                camera_source = int(sys.argv[2])
            except ValueError:
                if sys.argv[2].startswith("http"):
                    camera_source = sys.argv[2]  # Use as URL
                else:
                    print(f"Invalid camera index argument: {sys.argv[2]}, defaulting to 0")
    
    # Add option to list available cameras
    print("\n=== SIGN LANGUAGE RECOGNITION SYSTEM ===")
    print("Checking for available cameras...")
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("\nNo regular cameras found. You might need to use DroidCam.")
        print("Choose an option:")
        print("1. Try DroidCam connection")
        print("2. Try another camera index manually")
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            print("\nFor DroidCam, enter the IP address shown in the app:")
            print("(e.g., http://192.168.1.100:4747/video)")
            droidcam_url = input("URL: ").strip()
            if droidcam_url:
                camera_source = droidcam_url
            else:
                print("No URL entered, will try automatic detection")
        elif choice == "2":
            try:
                idx = int(input("Enter camera index to try: ").strip())
                camera_source = idx
            except ValueError:
                print("Invalid input, using default camera index 0")
    else:
        print(f"\nAvailable cameras: {available_cameras}")
        print("Choose an option:")
        print("1. Use detected camera")
        print("2. Try DroidCam connection")
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            if 1 in available_cameras:  # Your logs showed camera 1 was available
                camera_source = 1
                print(f"Using camera index {camera_source}")
            else:
                camera_source = available_cameras[0]
                print(f"Using camera index {camera_source}")
        elif choice == "2":
            print("\nFor DroidCam, enter the IP address shown in the app:")
            print("(e.g., http://192.168.1.100:4747/video)")
            droidcam_url = input("URL: ").strip()
            if droidcam_url:
                camera_source = droidcam_url
            else:
                print("No URL entered, will try automatic detection")
    
    # Try different backends if available and using a numeric camera index
    if isinstance(camera_source, int):
        print("\nSelect camera backend:")
        print("0: Default")
        print("1: DirectShow (Windows)")
        print("2: Media Foundation (Windows)")
        print("3: V4L2 (Linux)")
        
        try:
            backend_choice = input("Enter choice (0-3), or just press Enter for default: ").strip()
            if backend_choice == "":
                backend_choice = "0"
            
            backend_choice = int(backend_choice)
            backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
            backend_names = ["Default", "DirectShow (Windows)", "Media Foundation (Windows)", "V4L2 (Linux)"]
            
            if 0 <= backend_choice < len(backends):
                camera_api = backends[backend_choice]
                print(f"Using {backend_names[backend_choice]} backend")
        except ValueError:
            print("Invalid choice, using default backend")
    
    print("\nStarting sign language recognition system...")
    recognizer = SignLanguageRecognizer(model_path=model_path, camera_source=camera_source, camera_api=camera_api)
    recognizer.predict_real_time()