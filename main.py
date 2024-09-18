import cv2
from ultralytics import YOLO
from cvzone.SerialModule import SerialObject
from time import sleep
import numpy as np


#aurdino = SerialObject()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

def result(event, x, y, flags, param):
    
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)
cv2.namedWindow('result')
cv2.setMouseCallback('result', result)
        

#webcam ip address
cap = cv2.VideoCapture("highway.mp4")
#cap = cv2.VideoCapture(0)

area=[(2,410),(1016,406)]

while cap.isOpened(): 
    success, frame = cap.read()

    frame=cv2.resize(frame,(1020,500))
    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,0),3) 

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame,verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        for r in results:
              
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                c = box.cls.item()
                
        cv2.imshow("result", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
