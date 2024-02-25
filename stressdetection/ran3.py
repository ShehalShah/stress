import cv2
import joblib
import numpy as np
import mediapipe as mp
import time

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Function to extract pose keypoints from an image
def extract_keypoints(image):
    if image is None:
        print("Error loading image")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        return [landmark.x for landmark in results.pose_landmarks.landmark] + [landmark.y for landmark in results.pose_landmarks.landmark]
    else:
        return None

# Load the trained model
model_filename = 'posture_classifier_model.pkl'
loaded_model = joblib.load(model_filename)

# List of posture types
posture_types = ['slouch', 'headforward', 'tilting', 'shoulders', 'leaning', 'normal']

# Initialize webcam
cap = cv2.VideoCapture(0)

last_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    if not ret:
        break
    
    current_time = time.time()
    
    if current_time - last_time >= 2: 
        last_time = current_time
        
        keypoints = extract_keypoints(frame)
        
        if keypoints:
            predicted_label = loaded_model.predict([keypoints])[0]
            predicted_posture = posture_types[predicted_label]
            print(f"Predicted Posture: {predicted_posture}")
    
    cv2.imshow('Real-time Pose Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
