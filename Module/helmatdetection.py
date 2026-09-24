import cv2
from ultralytics import YOLO
from Module.licensePlateDetection import licensePlateDetector
import os

os.makedirs("violationsReport", exist_ok=True)
helmet_model = YOLO("runs/detect/train-3/weights/best.pt")

violation_count = 0
violation_active = False

def helmet_detection(frame):
    global violation_count
    global violation_active
    results = helmet_model(frame, conf=0.5)
    no_helmet_detected = False
    objtwo = licensePlateDetector
    frame = objtwo.license_plate_detection(frame)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = helmet_model.names[cls_id].lower()
            if name in ["no_helmet", "no-helmet", "no helmet"]:
                no_helmet_detected = True
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    1
                )
                cv2.rectangle(
                    frame,
                    (x1, max(0, y1 - 25)),
                    (x1 + 195, y1),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    frame,
                    f"Helmet violation {confidence:.2f}",
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )
                if not violation_active:
                    violation_count += 1
                    filename = (
                        f"violationsReport/"
                        f"violation_{violation_count}.jpg"
                    )
                    cv2.imwrite(filename, frame)
                    print(
                        f"Violation saved: {filename}"
                    )
                    violation_active = True
                    
            elif name in ["helmet", "helmat"]:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )
                cv2.rectangle(
                    frame,
                    (x1, max(0, y1 - 25)),
                    (x1 + 140, y1),
                    (0, 255, 0),
                    -1
                )
                cv2.putText(
                    frame,
                    f"Helmet {confidence:.2f}",
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )
                
    if not no_helmet_detected:
        violation_active = False
        
    return frame
