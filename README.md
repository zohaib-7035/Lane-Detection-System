
````markdown
# 🚗 Road Lane Violation Detection System

A real-time lane detection and violation monitoring system built using classical computer vision techniques with OpenCV. This project processes dashcam footage to detect lane boundaries and identify when a vehicle deviates from its lane.

---

## 📌 Features

- Canny Edge Detection for lane highlighting  
- Region of Interest (ROI) masking for focused processing  
- Hough Transform for line detection  
- Lane separation (left & right) using slope filtering  
- Temporal smoothing using moving averages  
- Lane violation detection based on vehicle offset  
- Road frame filtering (removes noisy/non-road frames)  
- Real-time performance (up to ~48 FPS on low resolution)  
- Annotated output video with:
  - Lane boundaries  
  - Lane area fill  
  - Vehicle center vs lane center  
  - Violation alerts  
  - Debug edge window  

---

## 🧠 Methodology

The system follows a step-by-step pipeline:

1. Frame Filtering  
   Detects if frame contains road (gray asphalt + edges + lines)

2. Edge Detection  
   Gaussian blur + Canny edge detection

3. ROI Masking  
   Focus on lower region where lanes exist

4. Line Detection  
   Probabilistic Hough Transform

5. Lane Separation  
   Based on slope (left = negative, right = positive)

6. Smoothing  
   Moving average across multiple frames

7. Lane Center Calculation  
   Midpoint of left & right lanes

8. Violation Detection  
   Compare vehicle center with lane center

---

## 📊 Performance

| Resolution | FPS | Accuracy |
|------------|-----|---------|
| 960x540    | ~48 FPS | 100% |
| 1280x720   | ~31 FPS | 100% |
| 1920x1080  | ~14 FPS | 100% |

Performance decreases with higher resolution due to increased computation.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lane-violation-detection.git
cd lane-violation-detection
````

### 2. Install dependencies

```bash
pip install opencv-python numpy
```

---

## ▶️ Usage

```bash
python lane-detecto.py --input video.mp4 --output output.mp4
```

### Optional arguments:

| Argument      | Description              |
| ------------- | ------------------------ |
| --input       | Input video file         |
| --output      | Output video file        |
| --save-frames | Save each frame as image |
| --slow        | Slow down output video   |

### Example:

```bash
python lane-detecto.py -i highway.mp4 -o result.mp4 --slow 2 --save-frames
```

---

## 📁 Output

* Annotated video (output.mp4)
* Optional frame images (output_frames/)
* Console statistics:

  * Total frames
  * Road vs non-road frames
  * Violations detected
  * Processing speed

---

## 📸 Visualization

The output includes:

* Green lane area
* Yellow & white lane lines
* Vehicle center (blue)
* Lane center (yellow)
* Violation alerts (red)

---

## ⚙️ Configuration

All parameters are configurable in the `Config` class:

```python
class Config:
    CANNY_LOW = 50
    CANNY_HIGH = 150
    HOUGH_THRESHOLD = 50
    VIOLATION_TOLERANCE_PX = 30
```

You can tune:

* Edge detection sensitivity
* Lane detection accuracy
* Violation threshold
* ROI region

---

## 🚀 Future Improvements

* Deep Learning-based lane detection (CNN / Segmentation)
* Night-time & rain robustness
* Multi-lane detection
* Real-time deployment on embedded systems
* Integration with autonomous driving systems

---

## 📚 Tech Stack

* Python
* OpenCV
* NumPy

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch
3. Commit your changes
4. Submit a pull request

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Zohaib Shahid
Data Science Student | ML Enthusiast

```


