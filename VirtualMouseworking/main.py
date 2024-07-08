import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
import ctypes

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

video = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image_height, image_width, _ = image.shape

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                index_finger_tip = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    image_width, image_height)

                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    landmarks[mp_hands.HandLandmark.THUMB_TIP].x,
                    landmarks[mp_hands.HandLandmark.THUMB_TIP].y,
                    image_width, image_height)

                if index_finger_tip and thumb_tip:
                    try:
                        cv2.circle(image, (index_finger_tip[0], index_finger_tip[1]), 25, (0, 200, 0), 5)
                        cv2.circle(image, (thumb_tip[0], thumb_tip[1]), 25, (0, 0, 200), 5)

                        screen_x = int(index_finger_tip[0] * screen_width / image_width)
                        screen_y = int(index_finger_tip[1] * screen_height / image_height)

                        pyautogui.moveTo(screen_x, screen_y)

                        distance = calculate_distance(index_finger_tip, thumb_tip)

                        click_threshold = 40

                        if distance < click_threshold:
                            pyautogui.click()
                        else:
                            pyautogui.mouseUp(button='left')

                    except Exception as e:
                        print(f"Error: {e}")

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing the ESC key.
            break

video.release()
cv2.destroyAllWindows()
