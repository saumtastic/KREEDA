import cv2
import numpy as np
import random
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


cap = cv2.VideoCapture(0)


piece_size = 50
options = [(0, 0), (0, 0), (0, 0)]
shapes = ['Square', 'Circle', 'Triangle','Rhombus']
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
correct_option_index = 0
correct_shape = ""
score = 0
rounds = 5
time_limit = 5
round_start_time = time.time()
piece_selected = False



def draw_game_elements(frame, options, shapes, colors, correct_shape, piece_selected):
    height, width, _ = frame.shape
    piece_color = (0, 255, 0) if not piece_selected else (0, 0, 255)


    cv2.putText(frame, f'Select: {correct_shape}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    for i, pos in enumerate(options):
        color = colors[i]
        shape = shapes[i]
        if shape == 'Square':
            cv2.rectangle(frame, (pos[0] - piece_size // 2, pos[1] - piece_size // 2),
                          (pos[0] + piece_size // 2, pos[1] + piece_size // 2), color, -1)
        elif shape == 'Circle':
            cv2.circle(frame, pos, piece_size // 2, color, -1)
        elif shape == 'Triangle':
            points = np.array([[pos[0], pos[1] - piece_size // 2],
                               [pos[0] - piece_size // 2, pos[1] + piece_size // 2],
                               [pos[0] + piece_size // 2, pos[1] + piece_size // 2]], np.int32)
            cv2.fillPoly(frame, [points], color)


    cv2.putText(frame, f'Score: {score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Round: {6 - rounds}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

def generate_options():
    global correct_option_index, correct_shape
    height, width = cap.read()[1].shape[:2]

    option1 = (random.randint(piece_size, width - piece_size), random.randint(piece_size, height - piece_size))
    option2 = (random.randint(piece_size, width - piece_size), random.randint(piece_size, height - piece_size))
    option3 = (random.randint(piece_size, width - piece_size), random.randint(piece_size, height - piece_size))

    correct_option_index = random.randint(0, 2)
    correct_shape = shapes[correct_option_index]

    return [option1, option2, option3]



while rounds > 0:
    piece_selected = False
    round_start_time = time.time()
    options = generate_options()

    while time.time() - round_start_time < time_limit:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_center = None

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]


            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:

                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                cv2.circle(frame, eye_center, 10, (255, 0, 0), 2)
                break
        if eye_center:
            for i, pos in enumerate(options):
                if (pos[0] - piece_size // 2 < eye_center[0] < pos[0] + piece_size // 2 and
                        pos[1] - piece_size // 2 < eye_center[1] < pos[1] + piece_size // 2):
                    piece_selected = True
                    if i == correct_option_index:
                        score += 2
                    else:
                        score -= 1
                    break

        frame = draw_game_elements(frame, options, shapes, colors, correct_shape, piece_selected)


        cv2.imshow('Eye Tracking Puzzle Game', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        if piece_selected:
            break

    if cv2.waitKey(1) & 0xFF == 27:
        break

    rounds -= 1

cap.release()
cv2.destroyAllWindows()

print(f'Final Score: {score}')
