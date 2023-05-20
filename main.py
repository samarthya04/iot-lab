# Import required libraries
import cv2
import numpy as np
import dlib

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)

# Detect the coordinates
hog = cv2.HOGDescriptor()
detector = hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# detector = dlib.get_frontal_face_detector()

def detect(frame):
    bounding_box_cordinates, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1
        if person > 10:
            cv2.putText(frame, 'Please go to other door ', (40, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, 'Q : Quit', (500, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow('output', frame)

    return frame


# Capture frames continuously
while True:

    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = detect(frame)

    # This command let's us quit with the "q" button on a keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
