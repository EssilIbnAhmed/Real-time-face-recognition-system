import face_recognition
import cv2
import os
import numpy as np

# Path to dataset
dataset_path = "dataset"

# Initialize known face data
known_face_names = []
known_face_encodings = []

# Load dataset encodings
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_folder):
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            try:
                # Load image and find face encodings
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(person_name)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

print("Training complete. Starting webcam...")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        
        # Compare with known encodings
        distances = [np.linalg.norm(face_encoding - known_encoding) for known_encoding in known_face_encodings]
        min_distance = min(distances)
        best_match_index = distances.index(min_distance)

        # Set name if the distance is below a threshold
        if min_distance < 0.6:  # Adjust threshold as needed
            name = known_face_names[best_match_index]

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display the name
        cv2.putText(frame, f"{name} ({min_distance:.2f})", (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the resulting frame
    cv2.imshow('Webcam Face Recognition', frame)

    # Quit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
