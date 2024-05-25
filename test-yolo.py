from ultralytics import YOLO
import cv2
import cvzone
import math



#cap =cv2.VideoCapture(0)
cap =cv2.VideoCapture('/Users/roshdyhamdy/Desktop/BME8/RSA3/code/YOLO/test/5.MP4')
cap.set(3,640)
cap.set(4, 720)
model=YOLO('/Users/roshdyhamdy/Desktop/BME8/RSA3/code/YOLO/yolo_weights/yolov8n.pt')
classNames=['BVessels']
while True :
    success ,img = cap.read()
    results =model(img, stream=True, conf=0.45, iou=0.60)
    for r in results:
        boxes =r.boxes
        for box in boxes:
            x1,y1,x2,y2 =box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            conf =math.ceil(box.conf[0]*100)/100
            cls=box.cls[0]
            cls=int(cls)
            cv2.putText(img,f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,0,0))
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),3)
    cv2.imshow('stream',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            




