from ultralytics import YOLO
import cv2
import json

# Load trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# List of authorized persons (as per training labels)
authorized_persons = ['Person1', 'Person2']

def check_person(image_path):
    """Detect person in the image and return JSON response."""
    
    # Read image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Process detection results
    detected_persons = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)  # Get class ID
            confidence = float(box.conf)  # Confidence score
            class_name = model.names[class_id]  # Get class name

            # Save detected person
            detected_persons.append(class_name)

    # Check if the detected persons are authorized
    if any(person in authorized_persons for person in detected_persons):
        response = {
            "status": "success",
            "message": "Authorized person detected",
            "person": detected_persons
        }
    else:
        response = {
            "status": "failed",
            "message": "Unauthorized person detected",
            "person": None
        }

    return json.dumps(response, indent=4)

# Test the function
image_path = "test_images/new_person.jpg"  # Change to the image you want to check
print(check_person(image_path))
