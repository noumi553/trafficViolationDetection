import os
import cv2
from ultralytics import YOLO
from Module.wrongsideviolationdetection import process_wrong_way_frame
from Module.helmatdetection import helmet_detection
from Module.speed import SpeedDetector
from Module.red_and_green import process_red_light_frame

os.makedirs("violationsReport", exist_ok=True)

model = YOLO("yolov8n.pt")
detector = SpeedDetector()

modules_data = [
    {
        "name": "1. Wrong Way Violation Detection",
        "video": "videoPhotages/wrongway.mp4",
        "func": lambda f: process_wrong_way_frame(f, model)
    },
    {
        "name": "2. Helmet Detection",
        "video": "videoPhotages/demo.mp4",
        "func": lambda f: helmet_detection(f)
    },
    {
        "name": "3. Professional Speed Detection",
        "video": "videoPhotages/st.mp4",
        "func": lambda f: detector.process(f)
    },
    {
        "name": "4. Red Light & License Plate Detection",
        "video": "videoPhotages/VID_20260612_135203.mp4",
        "func": lambda f, res: process_red_light_frame(f, res)
    }
]

current_index = 0

def load_video(index):
    video_path = modules_data[index]["video"]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Video open nahi ho saki -> {video_path}")
    if index == 2:  
        detector.fps = cap.get(cv2.CAP_PROP_FPS) or 30
    return cap

current_module = modules_data[current_index]
cap = load_video(current_index)
window_name = "Traffic Monitoring System (Press 'c' to Change, 'q' to Quit)"

print(f"Current Running: {current_module['name']}")

while True:
    ret, frame = cap.read()
    
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    
    if current_index == 2:
        frame = cv2.resize(frame, (640, 360))
    else:
        frame = cv2.resize(frame, (900, 500))
    
    if current_index == 3:
        result = model.track(frame, persist=True, verbose=False)
        frame = current_module["func"](frame, result)
    else:
        frame = current_module["func"](frame)
    
    cv2.putText(frame, f"Mode: {current_module['name']}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.imshow(window_name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('c'):
        cap.release()
        current_index = (current_index + 1) % len(modules_data)
        current_module = modules_data[current_index]
        cap = load_video(current_index)
        print(f"Switched to: {current_module['name']}")

cap.release()
cv2.destroyAllWindows()
