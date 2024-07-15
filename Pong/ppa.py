import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np


class HandballGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.imgBackground = cv2.imread("Resources/Background.png")
        self.imgGameOver = cv2.imread("Resources/gameOver.png")
        self.imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
        self.imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
        self.imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

        self.detector = HandDetector(detectionCon=0.8, maxHands=2)

        self.ballPos = [100, 100]
        self.speedX = 15
        self.speedY = 15
        self.gameOver = False
        self.score = [0, 0]

    def play_game(self, mode="local", player1="Player 1", player2="Player 2"):
        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            imgRaw = img.copy()

            hands, img = self.detector.findHands(img, flipType=False)

            img = cv2.addWeighted(img, 0.2, self.imgBackground, 0.8, 0)

            if hands:
                for hand in hands:
                    x, y, w, h = hand['bbox']
                    h1, w1, _ = self.imgBat1.shape
                    y1 = y - h1 // 2
                    y1 = np.clip(y1, 20, 415)

                    if hand['type'] == "Left":
                        img = cvzone.overlayPNG(img, self.imgBat1, (59, y1))
                        if 59 < self.ballPos[0] < 59 + w1 and y1 < self.ballPos[1] < y1 + h1:
                            self.speedX = -self.speedX
                            self.ballPos[0] += 30
                            self.score[0] += 1

                    if hand['type'] == "Right":
                        img = cvzone.overlayPNG(img, self.imgBat2, (1195, y1))
                        if 1195 - 50 < self.ballPos[0] < 1195 and y1 < self.ballPos[1] < y1 + h1:
                            self.speedX = -self.speedX
                            self.ballPos[0] -= 30
                            self.score[1] += 1

            if self.ballPos[0] < 40 or self.ballPos[0] > 1200:
                self.gameOver = True

            if self.gameOver:
                img = self.imgGameOver
                cv2.putText(img, f"{player1}: {self.score[0]}", (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3,
                            (255, 255, 255), 5)
                cv2.putText(img, f"{player2}: {self.score[1]}", (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3,
                            (255, 255, 255), 5)
                break
            else:
                if self.ballPos[1] >= 500 or self.ballPos[1] <= 10:
                    self.speedY = -self.speedY

                self.ballPos[0] += self.speedX
                self.ballPos[1] += self.speedY

                img = cvzone.overlayPNG(img, self.imgBall, self.ballPos)

                cv2.putText(img, str(self.score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
                cv2.putText(img, str(self.score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

            img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))

            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            if key == ord('r'):
                self.ballPos = [100, 100]
                self.speedX = 15
                self.speedY = 15
                self.gameOver = False
                self.score = [0, 0]
            elif key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return self.score

