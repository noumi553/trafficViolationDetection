import os
import math
import cv2 
from ultralytics import YOLO

os.makedirs("violations", exist_ok=True)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("videoPhotages/wrongway.mp4")

area = [(891, 236), (322, 427)]
point = [(861, 167), (792, 192)]
violator = set()
car_positions = {}
up_counter = {}
down_counter = {}


def point_line_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    numerator = abs(
        (y2 - y1) * px -
        (x2 - x1) * py +
        x2 * y1 -
        y2 * x1
    )
    denominator = math.sqrt(
        (y2 - y1) ** 2 +
        (x2 - x1) ** 2
    )
    if denominator == 0:
        return 9999
    return numerator / denominator

class WrongWayViolation:
    def Main():
        while True:
            ret,frame = cap.read()
            
            if not ret:
                break
            
            frame = cv2.resize(frame,(900,500))
            
            cv2.line(frame,area[0],area[1],(55, 164, 255),2)
            cv2.line(frame,point[0],point[1],(255,0,255),2)
            
            results = model.track(frame,persist=True,verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, track_id, cls in zip(boxes, ids, classes):
                    label = model.names[cls]
                    if label not in ["car", "truck", "bus", "motorcycle"]:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = y2
                    cv2.circle(frame,(cx, cy),5,(255, 0, 0),-1)
                    distance = point_line_distance(cx,cy,area[0],area[1])
                    cv2.putText(
                        frame,
                        f"{distance:.1f}",(cx + 5, cy - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 255),1)
                    if track_id not in car_positions:
                        car_positions[track_id] = (cx, cy)
                        up_counter[track_id] = 0
                    else:
                        prev_x, prev_y = car_positions[track_id]
                        dy = cy - prev_y
                        if dy < -3:
                            up_counter[track_id] += 1
                        else:
                            up_counter[track_id] = 0
                        if (
                            up_counter[track_id] >= 5 and distance <= 8 and track_id not in violator):
                            violator.add(track_id)
                    car_positions[track_id] = (cx, cy)
                    
                    if track_id not in car_positions:
                        car_positions[track_id] = (cx, cy)
                        down_counter[track_id] = 0
                    else:
                        prev_x, prev_y = car_positions[track_id]

                        dy = cy - prev_y

    # Moving Down (Top -> Bottom)
                        if dy > 3:
                            down_counter[track_id] += 1
                        else:
                            down_counter[track_id] = 0

                        if (
                            down_counter[track_id] >= 5 and
                            distance <= 8 and
                            track_id not in violator
                        ):
                            violator.add(track_id)

                    car_positions[track_id] = (cx, cy)

                    if track_id in violator:
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(frame, (x1, y1 - 22), (x1 + 120, y1), color, -1)
                        cv2.putText(
                            frame,
                            "WrongWay Violation",(x1 + 10, y1 - 6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
                        filename = f"violationsReport/vehicle_{track_id}.jpg"
                        cv2.imwrite(filename, frame)
                    else:
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"ID:{track_id}",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )
            
            cv2.imshow("Wrong way violation detection",frame)
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()