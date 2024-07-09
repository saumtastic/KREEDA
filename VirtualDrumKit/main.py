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

# MediaPipe hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Function to determine hand gesture and play corresponding sound
def detect_and_play_gesture(hand_landmarks, frame):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Convert normalized coordinates to pixel coordinates
    h, w, _ = frame.shape
    thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
    middle_tip_coords = (int(middle_tip.x * w), int(middle_tip.y * h))

    # Draw circles on finger tips
    cv2.circle(frame, thumb_tip_coords, 10, (0, 255, 0), -1)
    cv2.circle(frame, index_tip_coords, 10, (0, 255, 0), -1)
    cv2.circle(frame, middle_tip_coords, 10, (0, 255, 0), -1)

    # Determine gesture based on finger positions
    if thumb_tip.y < index_tip.y < middle_tip.y:
        sounds['kick'].play()
        cv2.putText(frame, 'Kick', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif thumb_tip.y > index_tip.y > middle_tip.y:
        sounds['snare'].play()
        cv2.putText(frame, 'Snare', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
        sounds['hihat'].play()
        cv2.putText(frame, 'HiHat', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
