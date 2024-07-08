import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/Score.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)
imgBackground = cv2.resize(imgBackground, (1280, 720))

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

while True:

    _, img = cap.read()
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img, draw = False, flipType=False)
    img = cv2.addWeighted(img, 0.1, imgBackground, 0.8, 0)


    img = cvzone.overlayPNG(img, imgBall, (59, 100))

    #if hands:
        #for hand in hands:
            #if hand['type'] == "left":
                #img = cvzone.overlayPNG(img, imgBall, (59, 100))

    #img = cvzone.overlayPNG(img,0.2, imgBackground,0.8,0)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()