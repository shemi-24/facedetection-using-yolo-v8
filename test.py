import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('C:\\incident_detection\\detect\\runs\\detect\\yolov8n_custom_dataset\\weights\\best.pt')

# Define authorized persons (class names from your dataset)
authorized_persons = ['person1', 'person2']  # Update this list with your trained class names

# Detection threshold
threshold = 0.6  # Lowered to detect more objects

# Open the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop through the webcam frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform detection on the frame
    results = model(frame)  # Run YOLO detection

    # Process the results (bounding boxes and class names)
    for result in results:
        boxes = result.boxes

        for box in boxes:
            try:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if confidence >= threshold:  # Only consider detections above threshold
                    # Get the class name from the model
                    class_name = model.names[class_id] if class_id < len(model.names) else "unknown"

                    # Check if detected person is authorized
                    if class_name in authorized_persons:
                        label = f"{class_name} {confidence:.2f}"
                        color = (0, 255, 0)  # Green for authorized persons
                    else:
                        label = f"unknown {confidence:.2f}"
                        color = (0, 0, 255)  # Red for unauthorized persons

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print("Error processing a detection:", e)

    # Display the current frame with detections
    cv2.imshow('Live Webcam Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()
