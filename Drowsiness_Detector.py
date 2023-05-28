import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# Constants for eye landmarks
LEFT_EYE_LANDMARKS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_LANDMARKS = [42, 43, 44, 45, 46, 47]
MOUTH_LANDMARKS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

# Constants for PERCLOS thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.4
PERCLOS_THRESHOLD = 0.15

# Count of consecutive frames where eye is closed
consecutive_frames = 0
blink_counter = 0

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate EAR
def calculate_ear(eye_landmarks):
    a = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    b = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    c = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (a + b) / (2.0 * c)
    return ear

# Function to calculate MAR
def calculate_mar(mouth_landmarks):
    top_lip = np.mean(mouth_landmarks[2:6], axis=0)
    bottom_lip = np.mean(mouth_landmarks[8:12], axis=0)
    mar = dist.euclidean(top_lip, bottom_lip)
    return mar

# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Get the eye landmarks
        left_eye_landmarks = np.array([(landmarks.part(idx).x, landmarks.part(idx).y) for idx in LEFT_EYE_LANDMARKS])
        right_eye_landmarks = np.array([(landmarks.part(idx).x, landmarks.part(idx).y) for idx in RIGHT_EYE_LANDMARKS])

        # Calculate EAR for left and right eyes
        left_ear = calculate_ear(left_eye_landmarks)
        right_ear = calculate_ear(right_eye_landmarks)

        # Get the mouth landmarks
        mouth_landmarks = np.array([(landmarks.part(idx).x, landmarks.part(idx).y) for idx in MOUTH_LANDMARKS])

        # Calculate MAR
        mar = calculate_mar(mouth_landmarks)

        # Calculate PERCLOS
        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
            consecutive_frames += 1
        else:
            if consecutive_frames >= 15:
                blink_counter += 1
            consecutive_frames = 0

        if mar > MAR_THRESHOLD:
            consecutive_frames += 1
        else:
            if consecutive_frames >= 10:
                blink_counter += 1
            consecutive_frames = 0

        perclos = consecutive_frames / 15.0

        # Display the eye and mouth aspect ratios on the frame
        cv2.putText(frame, f"EAR: {round((left_ear + right_ear) / 2, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {round(mar, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"PERCLOS: {round(perclos, 2)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # Draw landmarks on the frame
        for (x, y) in np.concatenate((left_eye_landmarks, right_eye_landmarks, mouth_landmarks)):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Drowsiness Detector', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
