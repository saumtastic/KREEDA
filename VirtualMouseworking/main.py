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
