import cv2
import cvzone
import math
import requests
import numpy as np
from ultralytics import YOLO




# Object marker dimensions (in pixels)
x0 = 150
y0 = 250
x00 = 200
y00 = 250
marker_center = (x0 + x00) / 2, (y0 + y00) / 2


# URL for the video stream
url = "http://192.168.1.6:81/stream"


# Define YOLO model and class names
model = YOLO('/Users/roshdyhamdy/Desktop/BME8/RSA3/code/YOLO/yolo_weights/yolov8n.pt')
classNames = ['BV']



def process_stream(url):
  results_list = []
  response = requests.get(url, stream=True)
  if response.status_code == 200:
      bytes = b''
      for chunk in response.iter_content(chunk_size=4096):
          bytes += chunk
          a = bytes.find(b'\xff\xd8')
          b = bytes.find(b'\xff\xd9')
          if a != -1 and b != -1:
              jpg = bytes[a:b+2]
              bytes = bytes[b+2:]
              if len(jpg) > 0:
                  img_np = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                  # YOLO object detection
                  results = model(img_np, conf=0.35, iou=0.60)
                  for r in results:
                      boxes = r.boxes
                      for box in boxes:
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                          object_center = (x1 + x2) / 2, (y1 + y2) / 2
                          distance = math.dist(marker_center, object_center)

                          conf = math.ceil(box.conf[0] * 100) / 100
                          cls = box.cls[0]
                          cls = int(cls)
                            # Create a dictionary with desired information
                          result_dict = {"distance": distance, "conf": conf}
                          results_list.append(result_dict)
                          # Draw bounding box, class name, confidence, and distance
                          cv2.putText(img_np, f'{classNames[cls]},{conf}, dis={distance}px', (max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255, 255, 255))
                          cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 255, 0), 3)
                  yield (b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img_np)[1].tobytes() + b'\r\n', results_list)
  else:
      print("Failed to retrieve stream")



def generate():
    for frame, detection_results in process_stream(url):
        # Only yield the frame; avoid session operations here
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Session operations outside the generator







