# Traffic Violation Detection System

An AI-powered Traffic Violation Detection System that automatically detects traffic rule violations from images or videos using Computer Vision and Deep Learning techniques The system identifies vehicles detects violations and license plate numbers detection and stores the results as a image in violationReport folder

---

## 📌 Features

- Vehicle Detection
- Traffic Signal Detection
- Red Light Violation Detection
- License Plate Detection
- Number Plate Recognition (OCR)
- Store Violation Records
- Real-time Video Processing

---

## Technologies Used

- Python
- OpenCV
- YOLO
- NumPy
- EasyOCR / Tesseract OCR
- TensorFlow / PyTorch (if applicable)
- Flask (if web application)
- SQLite / MySQL (if database used)

---

## 📂 Project Structure

```
trafficViolationDetection/
│
├── modules/               
├── violationReport/        
├── images/                
├── videos/               
├── main.py              
├── requirements.txt
└── README.md

---

##  Installation

### 1. Clone Repository

```bash
git clone https://github.com/noumi553/trafficViolationDetection.git
cd trafficViolationDetection
```

### 2. Create Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

If using a webcam:

```bash
python main.py --source 0
```

If using a video:

```bash
python main.py --source videos/test.mp4
```

---

## Output

The system will:

- Detect vehicles
- Detect traffic violations
- Identify license plates
- Display results in real time
- Save detected violation images
- Store violation information in the database (if configured)

---

## Sample Output

Add screenshots here.

```
Input Video
      │
      ▼
Vehicle Detection
      │
      ▼
Violation Detection
      │
      ▼
License Plate Recognition
      │
      ▼
Result Saved
```

---

## Requirements

Example packages:

```
opencv-python
numpy
ultralytics
easyocr
torch
torchvision
pandas
matplotlib
```

Install using

```bash
pip install -r requirements.txt
```

---

## Future Improvements

- Helmet Detection
- Speed Estimation
- Wrong Way Detection
- 
- Web Dashboard
- Live CCTV Or Video/photoges Integration

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## Author

**Nouman**

GitHub: https://github.com/noumi553

---

## Support

If you found this project helpful, please give it a on GitHub.
# trafficViolationDetection
