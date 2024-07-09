import cv2
import numpy as np

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)


# Function to draw the game elements
def draw_game_elements(frame, eye_center):
    # Example puzzle elements
    height, width, _ = frame.shape
    piece_size = 50
    piece_color = (0, 255, 0)

    # Draw puzzle piece
    cv2.rectangle(frame, (eye_center[0] - piece_size // 2, eye_center[1] - piece_size // 2),
                  (eye_center[0] + piece_size // 2, eye_center[1] + piece_size // 2), piece_color, -1)

    return frame


# Main loop
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eye_center = (frame.shape[1] // 2, frame.shape[0] // 2)  # Default center position

    for (x, y, w, h) in faces:
        # Region of interest for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Calculate the eye center
            eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            cv2.circle(frame, eye_center, 10, (255, 0, 0), 2)
            break  # Only track one eye for simplicity

    # Draw game elements based on the eye center
    frame = draw_game_elements(frame, eye_center)

    # Display the frame
    cv2.imshow('Eye Tracking Game', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
