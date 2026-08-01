import cv2
from ultralytics import YOLO
from Module.licensePlateDetection import licensePlateDetector
import os

os.makedirs("violations", exist_ok=True)
saved_images = set()

helmat_Modle = YOLO("../runs/detect/best.pt")
cap = cv2.VideoCapture("videoPhotages/demo.mp4")
frame_count = 0

def helmat_detection(frame):
    results = helmat_Modle(frame)
    
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            
            cls_id = int(box.cls[0])
            name = helmat_Modle.names[cls_id]

            if "no-helmet" in name:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2), 
                    (0, 0, 255),
                    1
                )

                cv2.rectangle(frame, (x1, y1-20), (x1+180, y1), (0,0,255), -1)
                cv2.putText(frame,"voilation No-Helmet" , (x1+2, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                filename = f"violationsReport/frame.jpg"
                cv2.imwrite(filename, frame)
            elif 'numberplate' in name:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            else:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )

                cv2.rectangle(frame, (x1, y1-20), (x1+80, y1), (0,255,0), -1)
                cv2.putText(frame,"Helmet" , (x1+2, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

def Main():
    global frame_count
    while True:
        ret,frame = cap.read()
        
        if not ret:
            break
        frame_count += 1

        if frame_count % 2 != 0:
            continue
        frame = cv2.resize(frame,(900,500))
        
        helmat_detection(frame)
        
        obj = licensePlateDetector
        frame = obj.license_plate_detection(frame)
        cv2.imshow("license plate detection",frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
                break

Main()
cap.release()
cv2.destroyAllWindows()