import torch
import cv2
import numpy as np
import ssl
from datetime import datetime
from PIL import Image, PngImagePlugin
import os

ssl._create_default_https_context = ssl._create_unverified_context

# Load the YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Initialize the video capture (0 for default webcam, 1 for external webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream. Check your camera index and connections.")
    exit()

cap.set(cv2.CAP_PROP_FPS, 60)  # This might not be supported by your webcam
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# Create a folder to save captured frames
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# Function to add metadata to image using Pillow
def add_metadata(image_path, date_time):
    try:
        img = Image.open(image_path)
        meta = PngImagePlugin.PngInfo()

        # Add metadata
        meta.add_text("DateTime", date_time)

        # Save with metadata
        img.save(image_path, pnginfo=meta)
    except Exception as e:
        print(f"Error adding metadata: {e}")

# Start video capture
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get original frame dimensions
    orig_h, orig_w = frame.shape[:2]

    # Convert frame to RGB (for YOLOv5 model input)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to 320x320 for the YOLOv5 model
    img_resized = cv2.resize(img_rgb, (320, 320))

    # Run YOLOv5 model
    results = model(img_resized)
    print(results.xyxy[0])  # Debugging: Print detection results

    # Calculate scaling factors to map back to original frame size
    scale_w = orig_w / 320
    scale_h = orig_h / 320

    detected_person = False  # Flag to capture frame only once per detection loop

    # Iterate through detections
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  # Confidence threshold
            label = model.names[int(cls)]  # Get label of the detected object

            # Only proceed if the detected object is a "person"
            if label == "person":
                detected_person = True

                # Scale back to the original image size
                x1, y1, x2, y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h

                # Draw rectangle around the person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add label and confidence on the frame
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Capture and save the frame only if a person is detected
    if detected_person:
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"person_{current_time}.png"  # Using PNG for metadata support
        file_path = os.path.join(output_folder, filename)

        # Save the frame
        cv2.imwrite(file_path, frame)

        # Add the date and time metadata to the saved image
        add_metadata(file_path, current_time)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
