# last cheythath like shameer bro debug
import cv2
from ultralytics import YOLO

# Set the absolute path to your YOLO model file
model = YOLO('C:\\incident_detection\\detect\\runs\\detect\\yolov8n_custom\\weights\\last.pt')  # Change this path based on your folder structure

# Load the YOLO model
# model = YOLO('yolov8n.pt')  # Load the YOLO model

# Define authorized persons (class names from your dataset)
authorized_persons = ['Person1', 'Person2']

# Detection threshold
threshold = 0.0  # Confidence threshold for detections

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

    
    print("EDA MONE FRAME?????")
    print(frame)
    # Perform detection on the frame
    results = model(frame)  # Run YOLO detection

    # print(results)

    print("RESULTS VANNU>>>>>>>>")
    print(len(results))

    # Process the results (bounding boxes and class names)
    for result in results:
        boxes = result.boxes
        print("MONEEE BOXES???/")
        print(boxes)
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if confidence >= threshold:  # Only consider detections above the threshold
                # Get the class name from the model
                class_name = model.names[class_id]


                print("EDA HASHIRE>>>>>>>>")

                # Check if the detected person is authorized
                if class_name in authorized_persons:
                    label = f"{class_name} {confidence:.2f}"
                    color = (0, 255, 0)  # Green color for authorized persons
                else:
                    label = f"Unknown {confidence:.2f}"
                    color = (0, 0, 255)  # Red color for unauthorized persons

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            else:
                print("Nothing found")
        print("ORU KOPPUM ILLA???????")
    # Display the current frame with detections
    cv2.imshow('Live Webcam Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close any open windows
cap.release()
cv2.destroyAllWindows()




#latest code yolov8
# from ultralytics import YOLO
# import cv2
# import json
# import time

# # Load custom-trained YOLO model
# model_path = "C:/incident_detection/detect/runs/detect/train7/weights/best.pt"
# model = YOLO(model_path)

# # List of authorized persons (as per training labels)
# authorized_persons = ['Person1', 'Person2']  # Replace with your authorized persons' labels

# # Open the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()


# # Time lag for JSON response (in seconds)
# response_delay = 5  # Adjust this value as needed
# last_response_time = time.time()

# def detect_person(frame):
#     global last_response_time

#     # Run YOLO model on the frame
#     results = model(frame)

#     detected_persons = []
#     unauthorized_persons = []

#     for result in results:
#         boxes = result.boxes

#         for box in boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             class_name = model.names[class_id]

#             # Only consider "person" class (faces of Person1 and Person2)
#             if class_name in authorized_persons:
#                 detected_persons.append(class_name)

#                 # Color set to green for authorized persons
#                 color = (0, 255, 0)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Create a text label to display on the frame
#                 label = f"{class_name} {confidence:.2f}"

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#             # If an unauthorized person is detected
#             else:
#                 unauthorized_persons.append(class_name)

#                 # Color set to red for unauthorized persons
#                 color = (0, 0, 255)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Create a text label to display on the frame
#                 label = f"Unauthorized {confidence:.2f}"

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     # Print JSON response only if unauthorized persons are detected
#     current_time = time.time()
#     if unauthorized_persons and (current_time - last_response_time) >= response_delay:
#         response = {
#             "status": "unauthorized_person detected",
#             "persons": unauthorized_persons
#         }
#         print(json.dumps(response, indent=4))  # Print JSON response
#         last_response_time = current_time  # Update the last response time

#     return frame

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame from webcam.")
#         break

#     # Process frame for person detection
#     frame = detect_person(frame)

#     # Display frame
#     cv2.imshow('Live Face Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
#last cheytha officenn-----------------
# from ultralytics import YOLO
# import cv2
# import json
# import time

# # Load trained YOLO model
# model = YOLO("C:\\incident_detection\\detect\\runs\\detect\\train7\\weights\\best.pt")

# # List of authorized persons (as per training labels)
# authorized_persons = ['Person1', 'Person2']  # Replace with your authorized persons' labels

# # Open the webcam
# cap = cv2.VideoCapture(0)

# # Time lag for JSON response (in seconds)
# response_delay = 5  # Adjust this value as needed
# last_response_time = time.time()

# def detect_person(frame):
#     global last_response_time

#     # Get detection results
#     results = model(frame)
#     detected_persons = []

#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             class_name = model.names[class_id]

#             print(f"Detected: {class_name} with Confidence: {confidence}")  # Debug print

#             # Set a confidence threshold
#             if confidence > 0.1:
#                 detected_persons.append(class_name)

#                 # Color based on authorization
#                 color = (0, 255, 0) if class_name in authorized_persons else (0, 0, 255)
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 label = f"{class_name} {confidence:.2f}"

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#                 # If an unauthorized person is detected, send a JSON response
#                 if class_name not in authorized_persons and time.time() - last_response_time > response_delay:
#                     last_response_time = time.time()
#                     response = {
#                         "status": "Unauthorized",
#                         "person_detected": class_name,
#                         "confidence": confidence,
#                         "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#                     }
#                     print("Sending JSON response:", json.dumps(response))

#     return frame

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Process frame for person detection
#     frame = detect_person(frame)

#     # Display frame
#     cv2.imshow('Live Person Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

#-------------------------
# #this is global state but not corect
# from ultralytics import YOLO
# import cv2
# import json
# import time

# # Load trained YOLO model
# model = YOLO("C:\\incident_detection\\detect\\runs\\detect\\train7\\weights\\best.pt")

# # List of authorized persons (as per training labels)
# authorized_persons = ['Person1', 'Person2']

# # Open the webcam
# cap = cv2.VideoCapture(0)

# # Time lag for JSON response (in seconds)
# response_delay = 5  # Adjust this value as needed
# last_response_time = time.time()

# def detect_person(frame):
#     """Detect person in the frame and return response if unauthorized detected."""
#     global last_response_time

#     results = model(frame)
#     detected_persons = []

#     for result in results:
#         boxes = result.boxes

#         for box in boxes:
#             class_id = int(box.cls)
#             confidence = float(box.conf)
#             class_name = model.names[class_id]

#             # Only consider "person" class
#             if class_name == "person":
#                 detected_persons.append(class_name)

#                 # Color set to green if authorized, else red
#                 color = (0, 255, 0) if class_name in authorized_persons else (0, 0, 255)
#                 # x1, y1 - top left corner, x2, y2 - bottom right coordinate
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])

#                 # Create a text label to display on the frame
#                 label = f"{class_name} {confidence:.2f}"  # e.g., Person1 0.92

#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 2 - Thickness
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)

#     # Check for unauthorized persons
#     unauthorized_detected = any(person not in authorized_persons for person in detected_persons)

#     # Print JSON response with a delay
#     current_time = time.time()
#     if unauthorized_detected and (current_time - last_response_time) >= response_delay:
#         response = {
#             "status": "unauthorized_person detected",
#             "persons": detected_persons
#         }
#         print(json.dumps(response, indent=4))  # Print JSON response
#         last_response_time = current_time  # Update the last response time

#     return frame

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Process frame for person detection
#     frame = detect_person(frame)

#     # Display frame
#     cv2.imshow('Live Person Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
#------------------------------------------------
#first code but not global state
# from ultralytics import YOLO
# import cv2
# import json

# # Load trained YOLO model
# model = YOLO("C:\\incident_detection\\detect\\runs\\detect\\train3\\weights\\best.pt")

# # List of authorized persons (as per training labels)
# authorized_persons = ['Person1', 'Person2']

# # Open the webcam
# cap = cv2.VideoCapture(0)
# # print(cap)

# def detect_person(frame):
#     """Detect person in the frame and return response if unauthorized detected."""

#     results=model(frame)
#     detected_persons=[]
#     for result in results:
#         boxes=result.boxes

#         for box in boxes:
#             class_id=int(box.cls)
#             confidence=float(box.conf)
#             class_name=model.names[class_id]

#             detected_persons.append(class_name)

#             #color setakanam like green and red
#             color=(0,255,0) if class_name in authorized_persons else (0,0,255)
#             # x1,y1-top left corner x2,y2-bottom right coordinate
#             x1, y1, x2, y2=map(int,box.xyxy[0])

#             #This creates a text label to display on the frame.
#             label=f"{class_name} {confidence:.2f}"  #eg-  Person1 0.92

#             #             Draw bounding box and label

#             # This draws a rectangle (bounding box) around the detected object.
#             cv2.rectangle(frame,(x1,x2),(x2,y2),color,2)  #2- Thickness 

#             # This displays the label text above the bounding box.
#             cv2.putText(frame,label,(x1, y1 - 10),cv2.FONT_HERSHEY_COMPLEX,0.9,color,2)

#             unauthorized_detected=any(person not in authorized_persons for person in detected_persons)

#             if unauthorized_detected:
#                 response={
#                     "status":"unauthorized_person detected",
#                     "person":detected_persons

#                 }
#                 print(json.dumps(response, indent=4))  # Print JSON response
#             return frame
        
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Process frame for person detection
#     frame = detect_person(frame)

#     # Display frame
#     cv2.imshow('Live Face Detection', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()        