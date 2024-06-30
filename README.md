YOLO Object Detection with OpenCV and Python
This project demonstrates real-time object detection using YOLO ("You Only Look Once") deep learning algorithm implemented in OpenCV. It detects objects in live webcam feed and draws bounding boxes with class labels.

Prerequisites
Python 3.x
OpenCV (cv2)
NumPy
Pillow (PIL)
(Optional) CUDA and cuDNN for GPU acceleration
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install dependencies:

bash
Copy code
pip install opencv-python numpy pillow
Download YOLO model weights and configuration files:

Download yolov3.cfg and yolov3.weights from official YOLO website.
Usage
Place yolov3.cfg and yolov3.weights in your project directory.
Create a coco_classes.txt file containing the COCO dataset class names, one per line.
Edit the paths in the script (yolov3.cfg, yolov3.weights, coco_classes.txt) to match your local paths.
Run the script:

bash
Copy code
python yolo_object_detection.py
Press q to quit the application.

Customization
Adjust conf_threshold and nms_threshold in yolo_object_detection.py to change confidence and non-maximum suppression thresholds.
Customize fonts and colors by editing verdana.ttf and color generation in the script.
Credits
YOLO: YOLO: Real-Time Object Detection
OpenCV: OpenCV
License
This project is licensed under the MIT License - see the LICENSE file for details.
