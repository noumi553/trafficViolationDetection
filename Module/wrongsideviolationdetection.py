import math
import cv2 

# Tracking variables
violator = set()
car_positions = {}

area = [(891, 236), (322, 427)]
point = [(861, 167), (792, 192)]

def point_line_distance(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    numerator = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if denominator == 0:
        return 9999
    return numerator / denominator

def process_wrong_way_frame(frame, model):
    
    cv2.line(frame, area[0], area[1], (55, 164, 255), 2)
    cv2.line(frame, point[0], point[1], (255, 0, 255), 2)
    
    results = model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls_id in zip(boxes, ids, classes):
            label = model.names[cls_id]
            if label not in ["car", "truck", "bus", "motorcycle"]:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, y1  # Top point (circle box ke upar banega)
            
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            distance = point_line_distance(cx, cy, area[0], area[1])
            
            cv2.putText(frame, f"{distance:.1f}", (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if track_id not in car_positions:
                car_positions[track_id] = (cx, cy)
            else:
                prev_x, prev_y = car_positions[track_id]
                dy = cy - prev_y  # Agar negative hai matlab gadi upar ja rahi hai
                
                # Distance threshold ko 20 rakha hai aur upar jane wali movement par violation count hogi
                if distance <= 20 and dy < 0 and track_id not in violator:
                    violator.add(track_id)
                    print(f"WRONG WAY VIOLATION: ID {track_id}")

            car_positions[track_id] = (cx, cy)
            
            # UI & Bounding Boxes
            if track_id in violator:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y1 - 22), (x1 + 170, y1), color, -1)
                cv2.putText(frame, "WrongWay Violation", (x1 + 10, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imwrite(f"violationsReport/vehicle_{track_id}.jpg", frame)
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
            cv2.putText(frame, f"ID:{track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
    return frame
