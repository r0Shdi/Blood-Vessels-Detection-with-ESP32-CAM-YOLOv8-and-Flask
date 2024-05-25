from flask import Flask, Response, request, jsonify, render_template, redirect,url_for,session
import cv2
import cvzone
import math
import requests
import numpy as np
from ultralytics import YOLO
import json
import time
res =None
# Object marker dimensions (in pixels)
x0 = 150
y0 = 250
x00 = 200
y00 = 250
marker_center = (x0 + x00) / 2, (y0 + y00) / 2

# URL for the video stream
#url = "http://192.168.1.2:81/stream"

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
                         b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img_np)[1].tobytes() + b'\r\n')
  else:
      print("Failed to retrieve stream")



def get_frame(url):
    video = cv2.VideoCapture(url)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, img = video.read()
        if not success:
            break
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
        ret, jpeg = cv2.imencode('.jpg', img)   
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 









# Flask application
app = Flask(__name__)
app.secret_key = "mona"
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/startstream", methods=["POST"])
def getadress():
    url= request.form.get("StreamAdress")
    url = str(url)
    session["url"] = url
    return  redirect("/#stream")

@app.route("/stopstream", methods=["POST"])
def stopstream():
    if "url" in session:
        session.pop("url")
    return redirect("/#stream")


@app.route("/esp_feed")
def esp_feed():
    if "url" in session:
        url = session["url"]
        if 'mp4' in url:
            return Response(get_frame(url), mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return Response(process_stream(url), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(open("/Users/roshdyhamdy/Desktop/rsaUI/static/images/RSA03.jpeg", 'rb').read(), mimetype='image/jpeg') 




@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)









