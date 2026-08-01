import cv2
from ultralytics import YOLO
from Module.red_green_time import redAndGreen
from Module.licensePlateDetection import licensePlateDetector
import os

os.makedirs("violations", exist_ok=True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videoPhotages/VID_20260612_135203.mp4")
area = [(526, 286), (74, 291)]
#area = [(69, 172), (454, 171)]
obj = redAndGreen
frame_count = 0

violator = set()
car_positions = {}

def red_green_light_voilation(frame,result):
    
    if result[0].boxes.id is not None:
        
        boxes = result[0].boxes.xyxy.cpu().numpy()
        id = result[0].boxes.id.cpu().numpy().astype(int)
        classes = result[0].boxes.cls.cpu().numpy().astype(int)
        
        red_light = obj.red_and_green_light()
        
        cv2.line(
            frame,
            area[0],
            area[1],
            (0,0,255),
            2
        )
        
        if result[0].boxes.id is not None:
            boxes = result[0].boxes.xyxy.cpu().numpy()
            ids = result[0].boxes.id.cpu().numpy().astype(int)
            classes = result[0].boxes.cls.cpu().numpy().astype(int)
            for box,track_id,cls_id in zip(boxes,ids,classes):
                
                name = model.names[cls_id]
                
                if name not in ["car", "truck", "bus", "motorcycle"]:
                        continue
                
                x1,y1,x2,y2 = map(int,box)
                
                cx = (x1 + x2) // 2
                cy = y2
                
                cv2.circle(
                frame,
                (cx, cy),
                5,
                (255, 0, 0),
                -1
            )
                
                line_y = area[0][1]

                if track_id in car_positions:
                    prev_y = car_positions[track_id]

                    if (
                        red_light and
                        prev_y < line_y and
                        cy >= line_y and
                        track_id not in violator
                    ):
                        violator.add(track_id)
                        print(f"RED LIGHT VIOLATION: {track_id}")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                        cv2.rectangle(frame, (x1, y1-20), (x1+110, y1), (0,0,255), -1)
                        cv2.putText(frame, "Violation", (x1+2, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        filename = f"violationsReport/frame_{track_id}.jpg"
                        cv2.imwrite(filename, frame)
                    
                    elif (
                        red_light and
                        prev_y > line_y and
                        cy <= line_y and
                        track_id not in violator
                    ):
                        violator.add(track_id)
                        print(f"RED LIGHT VIOLATION: {track_id}")
                    
                    elif (
                        red_light and
                        prev_y > line_y and
                        cy <= line_y and
                        track_id not in violator
                    ):
                        violator.add(track_id)
                        print(f"RED LIGHT VIOLATION: {track_id}")

                car_positions[track_id] = cy
                
                if track_id in violator:
                    color = (0,0,255)
                    cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    1
                )
                    text = f"Violation"
                    cv2.rectangle(frame, (x1, y1-20), (x1+110, y1), (0,0,255), -1)
                    cv2.putText(frame,text , (x1+2, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(frame,
                        f"ID {track_id}",
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)
                    
                
                

        if red_light:
                cv2.rectangle(
                    frame,
                    (0, 0),   
                    (150, 65),     
                    (0, 0, 255),   
                    -1            
                )
                
                cv2.putText(
                    frame,
                    "RED SIGNALE",
                    (0, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
        else:
                cv2.rectangle(
                    frame,
                    (0, 0),   
                    (170, 65),
                    (0, 255, 0),       
                    -1
                )

                cv2.putText(
                    frame,
                    "GREEN SIGNALE",
                    (0, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


def main():
    while True:
        global frame_count
        ret,frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame,(550,300))
        frame_count += 1
        if frame_count % 2 != 0:
            continue
        
        result = model.track(
            frame,
            persist=True,
            verbose=False
        )
        objtwo = licensePlateDetector
        frame=objtwo.license_plate_detection(frame)
        red_green_light_voilation(frame,result)
        cv2.imshow("red_green_light_voilation",frame)
        
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

main()
cap.release()
cv2.destroyAllWindows()