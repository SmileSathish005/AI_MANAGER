import cv2
from ultralytics import YOLO, solutions
from cvzone.SerialModule import SerialObject
from time import sleep
import numpy as np

#aurdino = SerialObject()

model = YOLO('yolov8n.pt')

 
cap = cv2.VideoCapture("highway.mp4")



line_points = [(20, 400), (1080, 400)]  # line or region points
classes_to_count = [0,2]  # person and car classes for count

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count,verbose=False)
    im0=counter.start_counting(im0, tracks)
    count= counter.in_counts
    print(count)
    
    
    
    

cap.release()
cv2.destroyAllWindows()
