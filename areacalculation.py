import cv2

# Mouse callback function jo click kiye gaye points ke coordinates nikaly gi
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Point coordinates: ({x}, {y})")
        
        # Frame par point mark kar dein taake nazar aaye
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"({x},{y})", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Get Coordinates", frame)

# Yahan apni video ka sahi path dein (misal ke tor par "video.mp4" ya webcam ke liye 0)
video_path = "videoPhotages/VID_20260612_135203.mp4"  # Agar webcam hai to yahan 0 likhein
cap = cv2.VideoCapture(video_path)

# Check karein ke video open hui ya nahi
if not cap.isOpened():
    print(f"Error: '{video_path}' file nahi mili ya open nahi ho rahi!")
else:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (550, 310))
        cv2.imshow("Get Coordinates", frame)
        cv2.setMouseCallback("Get Coordinates", click_event)
        
        print("--- Instructions ---")
        print("1. Video window par kahin bhi click karein.")
        print("2. Terminal mein coordinates show ho jayen ge.")
        print("3. Window band karne ke liye keyboard par 'q' press karein.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    else:
        print("Error: Video ka pehla frame read nahi ho saka.")

cap.release()
cv2.destroyAllWindows()
