import cv2

# Load pre-trained Haar cascades for eye and mouth detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#print(cv2.CascadeClassifier.empty(eye_cascade))

# Function to detect eyes and mouth in a frame
def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected eyes and mouth
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in mouth:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


# Open a video capture object
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Detect eyes and mouth in the frame
    frame = detect_features(frame)

    # Display the resulting frame
    cv2.imshow('Face Features Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
