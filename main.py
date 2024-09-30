import torch
import cv2
import numpy as np
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    orig_h, orig_w = frame.shape[:2]


    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    img_resized = cv2.resize(img_rgb, (640, 640))


    results = model(img_resized)


    scale_w = orig_w / 640
    scale_h = orig_h / 640


    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection
        if conf > 0.5:  
            label = model.names[int(cls)]  
            
        
            x1, y1, x2, y2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
            
       
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
      
            cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  
    cv2.imshow('Object Detection', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
