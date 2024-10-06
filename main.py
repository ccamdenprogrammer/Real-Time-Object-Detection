import torch
import cv2
import numpy as np
import ssl
from datetime import datetime
from PIL import Image, PngImagePlugin
import os

#THIS LINE IS NOT SECURE: ONLY FOR TESTING--REMOVES SSL VERIFICATION
ssl._create_default_https_context = ssl._create_unverified_context

#Loading the model, if can't be loaded, error is thrown
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

#setting the video capture camera. CHANGE FOR YOU SYSTEM. By default 0 works, but for me, 1 is the correct camera as it recognises 0 as my cellphone camera
cap = cv2.VideoCapture(1)
#error if camera can't be connected
if not cap.isOpened():
    print("Error: Could not open video stream. Check your camera index and connections.")
    exit()

#creating the folder to save the captured frames
output_folder = "captured_frames"
os.makedirs(output_folder, exist_ok=True)

#function to add date and time metadata to captures
def add_metadata(image_path, date_time):
    try:
        img = Image.open(image_path)
        meta = PngImagePlugin.PngInfo()
        meta.add_text("DateTime", date_time)
        img.save(image_path, pnginfo=meta)
    except Exception as e:
        print(f"Error adding metadata: {e}")

#By default, capture mode is turned off, to turn on press the 'c' key
capture_enabled = False 

#video capture loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    orig_h, orig_w = frame.shape[:2]
 
    #converting frame to RGB for yolo
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #resizing from for yolo
    img_resized = cv2.resize(img_rgb, (320, 320))

    #creating results
    results = model(img_resized)
    print(results.xyxy[0])  

    #image rescale
    scale_w = orig_w / 320
    scale_h = orig_h / 320
   
    #by default detected person bool is set to false, this changes when yolo detects a person in frame
    detected_person = False 

   
   #loop for detecting a person
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        #confidence threshhold change to your liking but default is 0.5
        if conf > 0.5:
            #getting the label for detected object
            label = model.names[int(cls)] 
            #if the label is "person"...
            if label == "person":
                #dected person is set to true and a rectangle is drawn around them in the frame.
                detected_person = True
                x1, y1, x2, y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h            
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #if capture mode is enabled and a person is detected
    if detected_person and capture_enabled:
        #grab current date and time
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #set filename to include date and time and write the file to the folder
        filename = f"person_{current_time}.png" 
        file_path = os.path.join(output_folder, filename)     
        cv2.imwrite(file_path, frame)       
        #add metadata to folder
        add_metadata(file_path, current_time)
    #frame
    cv2.imshow('Object Detection', frame)

    #program waitkeys: 'q' quits the progra, 'c' turns on and off capture mode
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        capture_enabled = not capture_enabled
        mode = "Capture Mode" if capture_enabled else "View-Only Mode"
        print(f"Mode switched to: {mode}")

#release resources
cap.release()
cv2.destroyAllWindows()
