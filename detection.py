from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')  # Use YOLOv8 Nano

# Train the model
results = model.train(data='dataset/data.yaml', epochs=50, imgsz=640)
# Validate the model
metrics = model.val()
print(metrics)
# Perform detection on an image
results = model('path_to_new_image.jpg')

# Display the results
for result in results:
    result.show()  # Show the image with bounding boxes
    result.save('outputs.jpg')  # Save the output image