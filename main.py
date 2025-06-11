import cv2
from ultralytics import YOLO
import os
from datetime import datetime
from PIL import Image, PngImagePlugin

# Load your custom-trained YOLOv8 model
# This path is from your successful training run
model = YOLO('/Users/camdencrace/Desktop/rto/Real-Time-Object-Detection/runs/detect/train/weights/best.pt')

# --- IP Camera Stream ---
# Connect to the Reolink camera's RTSP stream.
# The password has been URL-encoded to handle the special '@' character.
stream_url = "rtsp://admin:Iw0rk%40tp13dp1p3r!@10.96.74.137:554/h265Preview_01_sub"
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Check the RTSP URL, username, password, and network connection.")
    exit()

# creating the folder to save the captured frames
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

# function to add date and time metadata to captures
def add_metadata(image_path, date_time):
    try:
        img = Image.open(image_path)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("DateTime", date_time)
        img.save(image_path, pnginfo=meta)
    except Exception as e:
        print(f"Error adding metadata: {e}")

# By default, capture mode is turned off, to turn on press the 'c' key
capture_enabled = False

# video capture loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # by default detected object bool is set to false, this changes when yolo detects an object in frame
    object_detected = False

    # The results object is a list of Results objects. We take the first one.
    if results:
        result = results[0]
        # Iterate over each detected object's bounding box
        for box in result.boxes:
            conf = box.conf[0]
            # confidence threshold
            if conf > 0.5: # We can use a higher threshold now that the model is better
                x1, y1, x2, y2 = box.xyxy[0]
                cls_id = box.cls[0]
                label = model.names[int(cls_id)]
                
                object_detected = True
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # if capture mode is enabled and an object is detected
    if object_detected and capture_enabled:
        # grab current date and time
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # set filename to include date and time and write the file to the folder
        filename = f"detection_{current_time}.png"
        file_path = os.path.join(output_folder, filename)
        cv2.imwrite(file_path, frame)
        # add metadata to folder
        add_metadata(file_path, current_time)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # program waitkeys: 'q' quits the progra, 'c' turns on and off capture mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        capture_enabled = not capture_enabled
        mode = "Capture Mode" if capture_enabled else "View-Only Mode"
        print(f"Mode switched to: {mode}")

# release resources
cap.release()
cv2.destroyAllWindows()
