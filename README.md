# Real-Time-Object-Detection
This program uses a pre-trained YOLOv5 model to detect "person" objects in real-time video streams. When a person is detected: a bounding box is drawn around them, the frame is captured and saved with the bounding box and label, and the date and time of the capture are saved as metadata in the image file. This tool can be used for tasks such as monitoring environments, capturing specific individuals, or collecting datasets of human activities.

Requirements: Ensure that the following dependencies and tools are installed before running the program:

Python 3.7+
OpenCV (cv2)
PyTorch (torch)
YOLOv5 model (auto-downloaded)
Pillow (PIL for metadata handling)
SSL package (for secure loading of the model)
Installation:

Clone the repository:
Copy code
git clone https://github.com/your-username/person-detection-frame-capture.git
cd person-detection-frame-capture
Install the required Python packages:
Copy code
pip install torch torchvision torchaudio
pip install opencv-python
pip install pillow
Download the YOLOv5 model: The script automatically downloads the YOLOv5 model when executed. Make sure you are connected to the internet for the first run.
Usage:

Run the program:
Copy code
python person_detection.py
The script will attempt to open the webcam. If successful, it will start detecting objects in the video stream.
If a "person" is detected: A green rectangle will be drawn around them, the frame will be saved as a .png file in the captured_frames/ folder, and metadata containing the current date and time will be added to the image.
To stop the program, press the q key.

How It Works:

YOLOv5 Model: 
The program uses a pre-trained YOLOv5 small model (yolov5s) from the Ultralytics repository. The model detects multiple objects in each frame, and the program filters for "person" detections.

Video Capture: 
Video frames are captured using OpenCV. The script is set to use an external webcam (VideoCapture(1)), but it can be modified to use the default webcam (VideoCapture(0)).

Frame Processing: 
Each frame is resized to 320x320 pixels to match the YOLOv5 model input size. After the model processes the frame, the detection results (bounding boxes and class labels) are scaled back to the original frame size.

Bounding Boxes: 
If a person is detected (YOLOv5 class "person"), a green rectangle is drawn around the person, and the label (person) along with the confidence score is displayed.

Saving Frames:
When a person is detected, the current frame is saved in the captured_frames/ folder. The image file is saved as a PNG file, and the filename contains the timestamp (person_YYYY-MM-DD_HH-MM-SS.png).

Metadata: 
The script uses the Pillow library to add the current date and time as metadata (DateTime field) to the saved image files.
File Structure:

bash
Copy code
person-detection-frame-capture/
│
├── person_detection.py          # Main program file
├── captured_frames/             # Folder where captured images are saved
└── README.md                    # This README file
Metadata: Each saved image has metadata that includes the capture time. The metadata can be viewed using image viewers that support EXIF metadata or by using Python. Example metadata:

yaml
Copy code
DateTime: 2024-09-30_12-45-02
This indicates that the image was captured on September 30, 2024, at 12:45:02.


Notes:

Video Source: 
By default, the script uses the external webcam (cap = cv2.VideoCapture(1)). If your computer has only one webcam, you may need to change this to cap = cv2.VideoCapture(0).

Saving Frequency: 
The script captures and saves a frame whenever a person is detected. If multiple people are detected in a frame, only one frame will be saved per detection event.

Confidence Threshold: 
The program uses a confidence threshold of 0.5 for person detection. This can be adjusted by modifying the if conf > 0.5 condition.

Frame Rate: 
The script sets the webcam to capture at 60 FPS using cap.set(cv2.CAP_PROP_FPS, 60). However, actual FPS may vary depending on hardware performance.
Troubleshooting:

Video Stream Fails to Open: 
If the webcam doesn’t start, ensure that your webcam is connected and not being used by another application. Try changing the video capture device from cap = cv2.VideoCapture(1) to cap = cv2.VideoCapture(0).

Performance Issues: 
Processing each frame through the YOLOv5 model may introduce latency. To reduce lag, you can resize the input image to a smaller size or use a more powerful GPU if available.

No Person Detected: 
If the program doesn’t detect people correctly, check the lighting and angle of your webcam. YOLOv5 is sensitive to poor lighting and image clarity.

Conclusion: 
This project captures frames with detected "persons" in real-time, saving the frames with metadata. It's designed for simple surveillance tasks, research, or personal projects. The modular nature of the program allows for easy customization, such as adjusting the model, changing the detection confidence threshold, or integrating with other systems.