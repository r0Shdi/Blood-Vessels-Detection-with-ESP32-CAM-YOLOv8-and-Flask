import requests
import cv2
import numpy as np

url = "http://192.168.1.7:81/stream"
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
                try:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('ESP32-CAM Stream', frame)
                except cv2.error as e:
                    print("Error decoding frame:", e)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
else:
    print("Failed to retrieve stream")



