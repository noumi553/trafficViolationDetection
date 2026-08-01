from ultralytics import YOLO

model = YOLO(r"runs\detect\train-2\weights\best.pt")

class licensePlateDetector:
    def license_plate_detection(frame):
            results = model(frame)
            annotated_frame = results[0].plot()
            return annotated_frame