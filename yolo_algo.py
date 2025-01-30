from ultralytics import YOLO  # Imports the YOLO class from the ultralytics library, which provides an easy-to-use interface for YOLOv8
import cv2 # imports the OpenCV library for image and video processing.
model = YOLO('yolov8n.pt') 
#now this model is perform object detection on image and videos
results=model('/datasetss/train/cat/photo-removebg-preview.png')

for result in results:
    result.show()
    result.save('output.jpg')

