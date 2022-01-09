import numpy as np
import cv2 

cap = cv2.VideoCapture(0) 

face_cascade = cv2.CascadeClassifier('haarcascade\haarcascade_profileface.xml')

def detect_face(img):
    face_img = img.copy()
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray) 
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,0,255), 2) 
    return face_img
while(1):
    frame = cv2.imread('image\image11.jpg')
    #ret, frame = cap.read(0) 
    frame = detect_face(frame)
    cv2.imshow('Image Face Detection', frame) 
    cv2.imwrite('detected(haar)\image11_p.jpg', frame)
    c = cv2.waitKey(1) 
    if c == 27:
        break 
cv2.destroyAllWindows()
cap.release()
