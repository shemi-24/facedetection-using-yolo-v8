# from ultralytics import YOLO

# model=YOLO("yolov8n.yaml")
# results=model.train(data="data.yaml",epochs=2)
#--------------------------------------------------------
# from ultralytics import YOLO

# # Load the YOLOv8n model (you can also use yolov8s.pt, yolov8m.pt, etc.)
# model = YOLO('yolov8n.pt')  # Load the pre-trained YOLOv8n model

# # Train the model on your custom dataset
# results = model.train(data='C:\incident_detection\detect\data.yaml', epochs=2, imgsz=640)

# # Print training results (metrics)
# print(results)
#--------------------------------
# from ultralytics import YOLO
# model = YOLO("runs/detect/train16/weights/best.pt")    # output person1 and person2 verunnund
# print(model.names)  # Check the detected class names
#-----------------------------------------------------
# from ultralytics import YOLO
# model=YOLO("runs/detect/train16/weights/best.pt")
# results=model('C:/incident_detection/detect/dataset/images/train\img2.jpg',show=True)
# for r in results:
#     r.save(filename="outputss.jpg")  # Save the output image


from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO('yolov8n.pt')  # Load the pre-trained YOLOv8n model

# Train the model on your custom dataset
results = model.train(
    data='data.yaml',  # Path to your dataset configuration file
    epochs=100,                  # Number of training epochs
    imgsz=640,                   # Image size
    batch=16,                    # Batch size
    name='yolov8n_custom_dataset'        # Name of the training run
)



