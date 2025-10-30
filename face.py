import cv2
import os
from deepface import DeepFace
import numpy as np
from scipy.spatial.distance import cosine

# Paths to datasets
dataset_path_Essil = "C:/Users/sirin/Desktop/face rec 2/dataset/Essil"
dataset_path_Chada = "C:/Users/sirin/Desktop/face rec 2/dataset/Chada"
dataset_path_Hajer = "C:/Users/sirin/Desktop/face rec 2/dataset/Hajer"

# Load embeddings
known_encodings_Essil = []
known_encodings_Chada = []
known_encodings_Hajer = []

# Function to load face encodings
def load_face_encodings(dataset_path, encodings_list):
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        try:
            embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face")[0]["embedding"]
            encodings_list.append(embedding)
        except Exception as e:
            print(f"Skipping {image_path}: {e}")

# Load face encodings
load_face_encodings(dataset_path_Essil, known_encodings_Essil)
load_face_encodings(dataset_path_Chada, known_encodings_Chada)
load_face_encodings(dataset_path_Hajer, known_encodings_Hajer)

print(f"Loaded faces: Essil({len(known_encodings_Essil)}), Chada({len(known_encodings_Chada)}), Hajer({len(known_encodings_Hajer)})")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access webcam.")
    exit()

# Parameters
frame_width, frame_height = 640, 480
circle_radius = 80
trackers = []  # Store face trackers

# Colors and names
people_data = {
    "Essil": {"color": (0, 255, 0), "verified": False, "position": None},
    "Chada": {"color": (0, 255, 255), "verified": False, "position": None},
    "Hajer": {"color": (255, 0, 0), "verified": False, "position": None}
}

# Face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Reset verification states
    for person in people_data:
        people_data[person]["verified"] = False
        people_data[person]["position"] = None

    # Detect faces
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face_center = (x + w//2, y + h//2)
        face_roi = frame[y:y+h, x:x+w]

        try:
            # Directly use the face ROI for embedding calculation
            embedding = DeepFace.represent(img_path=face_roi, model_name="VGG-Face")[0]["embedding"]
            
            # Compare with each person's known encodings
            for person in people_data:
                encodings = locals()[f"known_encodings_{person.lower()}"]
                for known_encoding in encodings:
                    similarity = 1 - cosine(known_encoding, embedding)
                    if similarity > 0.6:  # Similarity threshold
                        people_data[person]["verified"] = True
                        people_data[person]["position"] = face_center
                        break
                        
        except Exception as e:
            print(f"Error in recognition: {e}")

    # Display
    for person, data in people_data.items():
        if data["position"]:
            # Circle around detected face
            cv2.circle(frame, data["position"], circle_radius, data["color"], 2)
            
            # Text with name
            status = person if data["verified"] else "UNKNOWN"
            color = data["color"] if data["verified"] else (0, 0, 255)
            
            cv2.putText(frame, status, 
                       (data["position"][0] - 40, data["position"][1] + circle_radius + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display video with face recognition
    cv2.imshow('Dynamic Face Recognition', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
