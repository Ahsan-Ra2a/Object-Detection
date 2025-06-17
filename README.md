# YOLOv3-Object-Detection-Cpp-OpenCV-VisualStudio

This project demonstrates real-time object detection using the YOLOv3 deep learning model integrated with OpenCV's DNN module in C++. Built and executed entirely in Visual Studio on Windows.

## 🚀 Features

- Real-time object detection with YOLOv3
- OpenCV DNN module integration
- Bounding box drawing with confidence scores
- Color/motion-based object filtering (optional)
- Optimized for video stream and webcam

---

## 🧰 Requirements

- Visual Studio (2019 or later)
- OpenCV (4.x preferred, with DNN module)
- YOLOv3 weights and config files
- C++17 (or compatible)

---

## 📁 Setup Instructions (Windows - Visual Studio)

### 1. Clone the Repository
```bash
git clone https://github.com/Ahsan-Ra2a/Object-Detection-OpenCV-C++.git
```

### 2. Install OpenCV

- Download OpenCV from https://opencv.org/releases/
- Extract and note the path

### 3. Configure Visual Studio Project

- In Visual Studio, create a new C++ Console App
- Add `main.cpp` in `src/` to your project
- Configure Project Properties:
  - **C/C++ → General → Additional Include Directories:** `C:\opencv\build\include`
  - **Linker → General → Additional Library Directories:** `C:\opencv\build\x64\vc15\lib`
  - **Linker → Input → Additional Dependencies:**
    ```
    opencv_world4xx.lib
    ```

> Replace `4xx` with your OpenCV version (e.g., `opencv_world455.lib`)

---

## 🔗 Download YOLOv3 Files

- [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Place them in the following folders:
```
/weights/yolov3.weights
/cfg/yolov3.cfg
/coco.names
```

---

## 🧠 Code Highlights

```cpp
cv::dnn::Net net = cv::dnn::readNetFromDarknet("cfg/yolov3.cfg", "weights/yolov3.weights");
net.setInput(blob);
std::vector<cv::Mat> outputs;
net.forward(outputs, net.getUnconnectedOutLayersNames());
```

- YOLO output parsing
- Non-Max Suppression (NMS)
- Confidence score filtering
- Bounding box drawing

---

## 📷 Sample Output
  check uploaded video is the sample of the program
---

## 📌 Notes

- You can switch to webcam stream by changing video source in `cv::VideoCapture`
- Use `confidenceThreshold` and `nmsThreshold` for tuning accuracy

---

## 🤝 Credits

- YOLOv3 by Joseph Redmon
- OpenCV library
- Developed by Ahsan Raza

---

## 🪪 License

This project is for learning and educational purposes only.
