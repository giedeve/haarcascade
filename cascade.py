import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread("img/trump.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.blur(face, ((w // 5), (h // 5)))
    img[y:y+h, x:x+w] = face


cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
