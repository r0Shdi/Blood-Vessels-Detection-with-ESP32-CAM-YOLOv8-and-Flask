from ultralytics import YOLO
import cv2
import cvzone
import math
import requests
import numpy as np

#assume marker object is rectangle shaped
#in pixels
x0=150
y0=250
x00=200
y00=250
mmid=(x0+x00)/2, (y0+y00)/2
# area0= (x00-x0)*(y00-y0)
# kdist=10 #in cm
# prod=area0*kdist
url = "http://192.168.1.5:81/stream"
response = requests.get(url, stream=True)

if response.status_code == 200:
    bytes = bytes()
    for chunk in response.iter_content(chunk_size=4096):
        bytes += chunk
        a = bytes.find(b'\xff\xd8')
        b = bytes.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]
            bytes = bytes[b+2:]
            if len(jpg) > 0:
                img_np = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                model = YOLO('/Users/roshdyhamdy/Desktop/BME8/RSA3/code/YOLO/yolo_weights/yolov8n.pt')
                classNames = ['BV']
                results = model(img_np, conf=0.35, iou=0.60)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        omid=(x1+x2)/2, (y1+y2)/2
                        dist = math.dist(mmid, omid)
                        # area1 =(x2-x1)*(y2-y1)
                        # dist= prod/area1
                        conf = math.ceil(box.conf[0] * 100) / 100
                        cls = box.cls[0]
                        cls = int(cls)
                        cv2.putText(img_np, f'{classNames[cls]},{conf}, dis={dist}px', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255))
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.imshow('ESP32-CAM Stream with YOLO', img_np)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
else:
    print("Failed to retrieve stream")





