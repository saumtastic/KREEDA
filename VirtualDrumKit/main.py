import cv2
import numpy as np
import pygame
import mediapipe as mp
pygame.mixer.init()

sounds = {
    'kick': pygame.mixer.Sound('kick.wav'),
    'snare': pygame.mixer.Sound('snare.wav'),
    'hihat': pygame.mixer.Sound('hihat.wav')
}
color_ranges = {
    'kick': ((20, 100, 100), (30, 255, 255)),  # Yellow
    'snare': ((110, 100, 100), (130, 255, 255)),  # Blue
    'hihat': ((50, 100, 100), (70, 255, 255))  # Green
}

drum_areas = {
    'kick': ((50, 100), (200, 250)),
    'snare': ((250, 100), (400, 250)),
    'hihat': ((450, 100), (600, 250))
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def detect_and_play_gesture(hand_landmarks, frame):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    h, w, _ = frame.shape
    hand_center = (
        int((wrist.x + middle_finger_mcp.x) / 2 * w),
        int((wrist.y + middle_finger_mcp.y) / 2 * h)
    )

    cv2.circle(frame, hand_center, 10, (0, 255, 0), -1)

    for drum, ((x1, y1), (x2, y2)) in drum_areas.items():
        if x1 < hand_center[0] < x2 and y1 < hand_center[1] < y2:
            sounds[drum].play()
            cv2.putText(frame, f'{drum} hit', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    for drum, ((x1, y1), (x2, y2)) in drum_areas.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, drum, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_and_play_gesture(hand_landmarks, frame)

    cv2.imshow('Virtual Drum Kit', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.quit()
