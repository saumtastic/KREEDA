import cv2
import numpy as np
import pygame
import mediapipe as mp

# Initialize Pygame mixer
pygame.mixer.init()

# Load drum sounds
sounds = {
    'kick': pygame.mixer.Sound('kick.wav'),
    'snare': pygame.mixer.Sound('snare.wav'),
    'hihat': pygame.mixer.Sound('hihat.wav')
}

# Define the color ranges for detection (HSV)
color_ranges = {
    'kick': ((20, 100, 100), (30, 255, 255)),  # Yellow
    'snare': ((110, 100, 100), (130, 255, 255)),  # Blue
    'hihat': ((50, 100, 100), (70, 255, 255))  # Green
}

# Define drum areas on the screen
drum_areas = {
    'kick': ((50, 100), (200, 250)),
    'snare': ((250, 100), (400, 250)),
    'hihat': ((450, 100), (600, 250))
}

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)


def detect_and_play_gesture(hand_landmarks, frame):
    # Get the center of the palm (average of wrist and middle finger MCP)
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    h, w, _ = frame.shape
    hand_center = (
        int((wrist.x + middle_finger_mcp.x) / 2 * w),
        int((wrist.y + middle_finger_mcp.y) / 2 * h)
    )

    # Draw a circle at the hand center
    cv2.circle(frame, hand_center, 10, (0, 255, 0), -1)

    # Check which drum area the hand center is in
    for drum, ((x1, y1), (x2, y2)) in drum_areas.items():
        if x1 < hand_center[0] < x2 and y1 < hand_center[1] < y2:
            sounds[drum].play()
            cv2.putText(frame, f'{drum} hit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    result = hands.process(rgb_frame)

    # Draw drum areas
    for drum, ((x1, y1), (x2, y2)) in drum_areas.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, drum, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Draw hand landmarks and detect gestures
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_and_play_gesture(hand_landmarks, frame)

    # Display the frame
    cv2.imshow('Virtual Drum Kit', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
