
import cv2
from ultralytics import YOLO
import os

os.makedirs("violations", exist_ok=True)

class SpeedDetector:
    def __init__(self, line1_y=180, line2_y=280, distance=10,
                speed_limit=60, offset=5):
        self.model = YOLO("yolov8n.pt")
        self.line1_y = line1_y
        self.line2_y = line2_y
        self.distance = distance
        self.speed_limit = speed_limit
        self.offset = offset

        self.vehicle_data = {}
        self.violators = set()
        self.fps = None
        self.frame_no = 0

    def process(self, frame):
        if self.fps is None:
            self.fps = 30

        self.frame_no += 1
        results = self.model.track(frame, persist=True, verbose=False)

        cv2.line(frame,(0,self.line1_y),(frame.shape[1],self.line1_y),(0,0,255),2)
        cv2.line(frame,(0,self.line2_y),(frame.shape[1],self.line2_y),(255,0,0),2)

        if results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box,tid,cls in zip(boxes,ids,classes):
            name=self.model.names[cls]
            if name not in ["car","truck","bus","motorcycle"]:
                continue

            x1,y1,x2,y2=map(int,box)
            cy=y2
            cx=(x1+x2)//2

            if tid not in self.vehicle_data:
                self.vehicle_data[tid]={
                                    "prev_y":cy,
                                    "start_frame":None,
                                    "speed":None,
                                    "cross1":False,
                                    "cross2":False,
                                    "direction":None
                                }

            d=self.vehicle_data[tid]
            prev=d["prev_y"]

            if (not d["cross1"]) and prev < self.line1_y <= cy:
                d["cross1"] = True
                d["direction"] = "DOWN"
                d["start_frame"] = self.frame_no

            if (
                d["cross1"]
                and not d["cross2"]
                and d["direction"] == "DOWN"
                and prev < self.line2_y <= cy
            ):
                d["cross2"] = True

                frames = self.frame_no - d["start_frame"]

                if frames > 0:
                    sec = frames / self.fps
                    d["speed"] = (self.distance / sec) * 3.6
                
                d["cross1"] = False
                d["cross2"] = False
                d["direction"] = None
                d["start_frame"] = None

            if (not d["cross2"]) and prev > self.line2_y >= cy:
                d["cross2"] = True
                d["direction"] = "UP"
                d["start_frame"] = self.frame_no

            if (
                d["cross2"]
                and not d["cross1"]
                and d["direction"] == "UP"
                and prev > self.line1_y >= cy
            ):
                d["cross1"] = True

                frames = self.frame_no - d["start_frame"]

                if frames > 0:
                    sec = frames / self.fps
                    d["speed"] = (self.distance / sec) * 3.6
                
                d["cross1"] = False
                d["cross2"] = False
                d["direction"] = None
                d["start_frame"] = None


            d["prev_y"]=cy

            color=(0,255,0)
            label=f"ID {tid}"

            if d["speed"] is not None:
                if d["speed"]>self.speed_limit:
                    color=(0,0,255)
                    self.violators.add(tid)
                    label=f"OVER {d['speed']:.0f} km/h"
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.circle(frame,(cx,cy),4,(0,255,255),-1)
                    cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                    cv2.imwrite(f"violationsReport/speed_{tid}.jpg", frame)
                else:
                    label=f"{d['speed']:.0f} km/h"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.circle(frame,(cx,cy),4,(0,255,255),-1)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            

        cv2.putText(frame,f"Violators: {len(self.violators)}",
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        return frame

class SpeedDetectorMain:
    def main():
            cap=cv2.VideoCapture("videoPhotages/st.mp4")
            detector=SpeedDetector()

            detector.fps=cap.get(cv2.CAP_PROP_FPS) or 30
            while True:
                ret,frame=cap.read()
                if not ret:
                    break
                frame=cv2.resize(frame,(900,500))
                out=detector.process(frame)
                cv2.imshow("Professional Speed Detection",out)
                if cv2.waitKey(1)&0xFF==ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()