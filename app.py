import torch
import cv2
import pyttsx3
import json
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
import time
import pandas as pd  # Ensure pandas is installed

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Log function
def log(message):
    print(f"[INFO] {message}")

# Speak function
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLOv5 model
log("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
log("YOLOv5 model loaded.")

# Load gender detection model (or use an alternative pre-trained model for gender classification)
log("Loading gender detection model...")
gender_model = models.resnet18(pretrained=True)
gender_model.eval()

# Preprocessing for gender detection
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to detect gender
def detect_gender(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = gender_model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted = torch.max(probabilities, 1)
        return "Female" if predicted.item() == 0 else "Male"

# Function to detect objects and save data
def detect_objects(video_url, output_file):
    log("Starting object detection...")
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        log("Error: Unable to open video.")
        return

    full_screen = False
    fps_counter = time.time()
    frame_id = 0
    all_detections = []  # Store all detection data

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_detections = {"frame": frame_id, "detections": []}

        # Perform object detection
        results = model(frame)
        detections = results.pandas().xyxy[0]  # pandas DataFrame for detection results

        # Annotate the frame and collect data
        for _, row in detections.iterrows():
            x1, y1, x2, y2, conf, cls, label = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class']), row['name']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Prepare detection data
            detection_data = {
                "label": label,
                "confidence": float(conf),
                "bounding_box": [x1, y1, x2, y2]
            }

            # Gender detection for 'person' class
            if label == 'person':
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    face_image = Image.fromarray(cropped)
                    gender = detect_gender(face_image)
                    detection_data["gender"] = gender
                    speak(f"Detected {gender} person")

            # Add detection to frame
            frame_detections["detections"].append(detection_data)

            # Voice feedback for other objects
            if label in ['dog', 'cat', 'bird']:
                speak(f"Detected {label}")

        # Save frame data
        all_detections.append(frame_detections)

        # Display FPS
        current_time = time.time()
        fps = int(1 / (current_time - fps_counter))
        fps_counter = current_time
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display the frame
        cv2.imshow("DixitCoder Software Testing", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):  # 'f' key to toggle full-screen mode
            full_screen = not full_screen
            cv2.setWindowProperty("Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full_screen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

    # Save detections to JSON file
    with open(output_file, "w") as f:
        json.dump(all_detections, f, indent=4)
    log(f"Detection data saved to {output_file}")
    log("Object detection completed.")

# Main function
if __name__ == "__main__":
    video_url = input("Enter video URL or path: ")
    output_file = "dataFormat.json"
    try:
        detect_objects(video_url, output_file)
    except Exception as e:
        log(f"An error occurred: {e}")
